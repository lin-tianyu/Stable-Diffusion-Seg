"""
The original version of SDSeg, built by disgustingly modify `LatentDiffusion`. Err.
I want to reimplement SDSeg using OOP! Elegance is the key!

In this original version:
    SDSeg == (modificaions of) LatentDiffusion <-- (modifications of) DDPM

Note that:
1. In `ddpm.py` (this file): tons of modifications have been made on both `DDPM` and `LatentDiffusion`, quite ugly...
2. In `SDSeg.py`: the `DDPM` and `LatentDiffusion` is 99% from https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py
    (only add a `only_model` parameter in `LatentDiffusion` when calling `self.init_from_ckpt`!)

Tianyu
Oct 31, 2024
"""

import sys
import os

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch import autocast
import tqdm
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from pytorch_lightning.utilities.distributed import rank_zero_only
from PIL import Image
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.slice2seg import prepare_for_first_stage, dice_score, iou_score
from ldm.data.synapse import colorize

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l1",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=True,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 original_elbo_weight=0.,
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=True):  # modified, only load unet
        """load only pretrained unet in training phase, load the entire model in testing phase"""
        sd = self.model.diffusion_model.state_dict()
        self.unet_sd_keys = set(map(lambda x: x.split(".")[0], sd.keys()))
        
        if not only_model:  
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"\033[32mRestored Diffusion, Cond-stage and First-stage model from {path} with "
                  f"{len(missing)} missing and {len(unexpected)} unexpected keys\033[0m")
            if len(missing) > 0:
                print(f"\033[31m[Missing Keys]\033[0m: {missing}\n")
            if len(unexpected) > 0:
                print(f"\033[31m[Unexpected Keys]\033[0m: {unexpected}\n")
        else:
            pretrain_sd = torch.load(path, map_location="cpu")
            
            if "label_emb" in self.unet_sd_keys:
                label_emb_keys = [key for key in sd.keys() if "label_emb" in key]
                label_emb_tmp = [sd.pop(key) for key in label_emb_keys]
            else:
                label_emb_tmp = None
            if "state_dict" in list(pretrain_sd.keys()):
                pretrain_sd = pretrain_sd["state_dict"]
            keys = list(pretrain_sd.keys())
            # deleting non-unet parameters
            for k in keys:  # only load unet parameters!
                if not k.startswith("model."):
                    del pretrain_sd[k]
                else:
                    v = pretrain_sd.pop(k)
                    new_k = k.replace("model.diffusion_model.", "")
                    pretrain_sd[new_k] = v
            # check incompatible parameters and fill with zeros (only 1 layer)
            for pk, k in zip(sorted(pretrain_sd.keys()), sorted(sd.keys())):
                assert pk == k and len(pretrain_sd[pk].shape) == len(sd[k].shape), \
                    ((pk, k) , (len(pretrain_sd[pk].shape) , len(sd[k].shape)))
                pshape, shape = pretrain_sd[pk].shape, sd[k].shape
                if pshape != shape:
                    if len(pretrain_sd[pk].shape) == 4:
                        # note: simply repeat is not working
                        sd[k] = torch.cat((pretrain_sd[pk], torch.zeros(pshape)), dim=1)
                        print(f"\033[31m[ATT]: filling zeros to initialize "
                              f"pretrained weight '{pk}' from {pshape} to {shape}\033[0m")
                else:
                    sd[k] = pretrain_sd[pk]

            # random init label_emb, will be ignored if not needed, so don't worry
            if label_emb_tmp is not None:
                for key, val in zip(label_emb_keys, label_emb_tmp):
                    sd[key] = val

            missing, unexpected = self.model.diffusion_model.load_state_dict(sd, strict=False)
            print(f"\033[32mRestored only Diffusion Model from {path} with "
                  f"{len(missing)} missing and {len(unexpected)} unexpected keys\033[0m")
            if len(missing) > 0:
                print(f"\033[31m[Missing Keys]\033[0m: {missing}\n")
            if len(unexpected) > 0:
                print(f"\033[31m[Unexpected Keys]\033[0m: {unexpected}\n")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        if batch_idx % 10 == 0:
            self.log("train/input_layer_pretrained_4",
                     self.model.diffusion_model.state_dict()["input_blocks.0.0.weight"][:, :4].mean().item())
            self.log("train/input_layer_zero_initiate_4",
                     self.model.diffusion_model.state_dict()["input_blocks.0.0.weight"][:, 4:].mean().item())
        loss_dict.pop("train/loss_vlb")

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=False)

        self.log("step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('param/lr_abs', lr.item(), prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 num_classes=2,
                 load_only_unet=True,
                 *args, **kwargs):
        self.num_classes = num_classes
        self.load_only_unet = load_only_unet
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=load_only_unet)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:  # and not self.restarted_from_ckpt:
            if self.scale_by_std:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                x = super().get_input(batch, self.first_stage_key)
                x = x.to(self.device)
                encoder_posterior = self.encode_first_stage(x)
                z = self.get_first_stage_encoding(encoder_posterior).detach()   # range roughly in (-18, 18)
                # print(z.shape, z.flatten().shape, z.min(), z.max(), z.flatten().std())
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())    # sts3d: 1/7.8957, synapse: 1/8.7146
            print(f"setting self.scale_factor to {self.scale_factor}")
            print(f"### USING STD-RESCALING: \033[31m{self.scale_by_std}\033[0m ###")
            self.log("val_avg_dice", 0, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        denoise_row_latent = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
            denoise_row_latent.append(zd.to(self.device))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)

        n_imgs_per_row_latent = len(denoise_row_latent)
        denoise_row_latent = torch.stack(denoise_row_latent)  # n_log_step, n_row, C, H, W
        denoise_grid_latent = rearrange(denoise_row_latent, 'n b c h w -> b n c h w')
        denoise_grid_latent = rearrange(denoise_grid_latent, 'b n c h w -> (b n) c h w')
        denoise_grid_latent = make_grid(denoise_grid_latent, nrow=n_imgs_per_row_latent)
        return denoise_grid, denoise_grid_latent

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        # print(batch["class_id"], batch["class_id"].shape)
        cls_id = batch["class_id"][:, 0]  
        # print(cls_id, cls_id.shape)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c, x, cls_id]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c, seg_label, cls_id = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, cls_id, seg_label)
        return loss

    def forward(self, x, c, cls_id, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        assert t.shape[0] == cls_id.shape[0], (t.shape, cls_id.shape, cls_id.shape[0])
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        c = dict(c_concat=[c], c_crossattn=[cls_id])  # hybrid mode requires a dict
        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def get_loss_seg_regression(self, x_start, x_noisy, t, model_output, seg_label=None):
        x_recon = self.predict_start_from_noise(x_noisy, t, noise=model_output)
        return self.get_loss(x_recon, x_start, mean=False)  # loss type according to `self.loss_type`


    def p_losses(self, x_start, cond, t, seg_label, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
    

        # get latent segmentation loss
        loss_seg = self.get_loss_seg_regression(x_start, x_noisy, t, model_output, seg_label).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_seg": loss_seg.mean().item()})

        # get noise loss 
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean().item()})

        # learn logvar (useless)
        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_seg / torch.exp(logvar_t) + logvar_t
        # loss = (loss_simple + loss_seg) / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean().item()})
            loss_dict.update({'logvar': self.logvar.data.mean().item()})

        # create loss
        loss = self.l_simple_weight * loss.mean()

        # get vlb loss (useless)
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.item()})

        # add vlb (useless) and latent seg loss
        loss += (self.original_elbo_weight * loss_vlb) 
        loss += loss_seg.mean() # weight == 1 is good enough! doesn't need to change this, 
        loss_dict.update({f'{prefix}/loss': loss.item()})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                        shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_dice(self, data=None, save_dir=None, ddim_steps=50):
        
        if data is None: # if dataset is not None, means the call comes from inference script.
            dataset = self.trainer.datamodule.datasets["test"]
            data = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

        # self.model.eval()     # ImageLogger will handle this
        metrics_dict = dict()
        seg_label_dict = dict()

        def get_dice(data, used_sampler="ddim", save_dir=None, ddim_steps=50):
            """
            Args:
                used_sampler: "direct", "ddim", "plms" ( "direct" -> self.predict_start_from_noise() )

            Returns:
                return ema_dice_list
            """

            def get_dice_loop(data, sampler, use_direct=False, noise=None, save_dir=None, ddim_steps=50):
                dice_list = 0.
                iou_list = 0.
                label_latent_list, samples_latent_list, cond_latent_list = list(), list(), list()
                label_image_list, samples_image_list = list(), list()
                samples_logits_list, samples_cond_list = list(), list()
                pbar = tqdm(data, desc="Validating Segmentation")   # volume-wise
                for prompts in pbar:
                    slice_path = prompts["file_path_"]
                    image = prompts["image"]  # 1 256 256 3  (1, H, W, D)
                    label = prompts["segmentation"]  # 1 256 256 3  (1, H, W, D)
                    assert image.shape == label.shape
                    assert label.max() == self.num_classes-1, label.max()
                    _, x, y, _ = label.shape
                    image = torch.from_numpy(zoom(image, (1, 256 / x, 256 / y, 1), order=1))
                    label = torch.from_numpy(zoom(label, (1, 256 / x, 256 / y, 1), order=0))
                    # print(image.device, image.shape, label.device, label.shape)
                    volume_name = slice_path[0].split("/")[-1].split("_")[0]
                    image, label = image.squeeze(0).numpy(), label.squeeze(0).numpy()
                    prediction = np.zeros_like(image)   

                    if image.shape[-1] > 3:     # for gray-scale 3D dataset inference
                        # prediction = np.zeros_like(image)   # 256 256
                        pbar_sub = tqdm(iterable=range(image.shape[2]), desc=f"[Inferring volume {volume_name.split('.')[0]}]")
                        for idx in range(image.shape[2]):
                            slice = image[:, :, idx]    # 256 256
                            slice_label = label[:, :, idx]  # H W
                            input = torch.from_numpy(slice).unsqueeze(2).unsqueeze(0).repeat((1, 1, 1, 3)).float().cuda()

                            c = dict(
                                c_concat=[self.get_learned_conditioning(prepare_for_first_stage(input))],
                                c_crossattn=[None]
                            )
                            samples_pred = list()
                            if use_direct:
                                noise = default(noise, lambda: torch.randn_like(c["c_concat"][0]))
                                final_t = torch.tensor([self.num_timesteps - 1], device=self.device).long()
                                if self.num_classes > 2:    # multi class segmentation
                                    for cls in range(0, self.num_classes):  # predict once for each class
                                        c["c_crossattn"] = [torch.tensor([cls], device=self.device)]   # cls_id
                                        model_output = self.apply_model(noise, final_t, c)
                                        pred_tmp = self.predict_start_from_noise(noise, final_t, noise=model_output)
                                        samples_pred.append(pred_tmp)
                                else:
                                    model_output = self.apply_model(noise, final_t, c)
                                    pred_tmp = self.predict_start_from_noise(noise, final_t, noise=model_output)
                                    samples_pred.append(pred_tmp)
                            else:
                                # for cls in range(0, self.num_classes):
                                #     c["c_crossattn"] = [torch.tensor([cls], device=self.device)]
                                pred_tmp, _ = sampler.sample(
                                    S=ddim_steps,
                                    conditioning=c,
                                    shape=(self.channels, self.image_size, self.image_size),
                                    batch_size=1,
                                    verbose=False,
                                    unconditional_guidance_scale=1.0,  # CT slice takes control
                                    unconditional_conditioning=None,  # dont need unconditional result
                                    eta=1.,
                                    x_T=None
                                )
                                samples_pred.append(pred_tmp)

                            out = torch.zeros((256, 256, self.num_classes))     # h w num_classes
                            if self.num_classes > 2:
                                for cls in range(0, self.num_classes):
                                    x_samples_ddim = self.decode_first_stage(samples_pred[cls])
                                    if cls == 0:
                                        x_samples_ddim *= -1    # for softmax
                                    x_samples_ddim = torch.mean(x_samples_ddim, dim=1, keepdim=False)    # b h w
                                    out[:, :, cls] = x_samples_ddim[0, ...]
                                out = out.softmax(dim=2).argmax(dim=2, keepdim=True).repeat(1, 1, 3).numpy()    # h w c==3
                            else:
                                x_samples_ddim = self.decode_first_stage(samples_pred[0])
                                x_samples_ddim = torch.clamp(
                                    (x_samples_ddim + 1.0) / 2.0 , min=0.0, max=1.0
                                )  # [-1, 1] -> [0, 13 or 255]     (b, c, h, w)
                                # channel-wise average, can not use for colored mode:
                                x_samples_ddim = x_samples_ddim.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                                out_p = rearrange(x_samples_ddim.squeeze(0).cpu().numpy(), 'c h w -> h w c')
                                out = (out_p > 0.5)
                            pbar_sub.set_postfix(dict(
                                label_cls=set(list(slice_label.flatten().astype(int))),
                                pred_cls=set(list(out.flatten().astype(int)))
                            )
                            )
                            prediction[:, :, idx] = out[:, :, 0]

                            if save_dir is not None:
                                slice_name = slice_path[0].split("/")[-1].split(".")[0] + f"_{idx}" + ".png"  # ori: .nii.gz
                                save_pred_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-gts", slice_name.split(".")[-1]]))
                                save_logits_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-logits", slice_name.split(".")[-1]]))
                                save_all_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-all", slice_name.split(".")[-1]]))
                                
                                save_pred = (out*255).astype(np.uint8)
                                save_logits = (out_p*255).astype(np.uint8)
                                save_gt = np.expand_dims((slice_label*255), 2).repeat(3, axis=2).astype(np.uint8)
                                save_cond = np.expand_dims((slice+1)/2*255, 2).repeat(3, axis=2).astype(np.uint8)
                                save_all = np.concatenate((save_cond, save_gt, save_pred, save_logits), axis=1)
                                
                                # WARNING: only *-all.png == exact test set size, the others only have size of non-empty slices. 
                                if slice_label.max() > 0:
                                    Image.fromarray(save_pred).save(save_pred_path)
                                    Image.fromarray(save_logits).save(save_logits_path)
                                Image.fromarray(save_all).save(save_all_path)


                            # # log non-empty examples for debugging
                            if idx % (image.shape[2] // 8) == 0:
                                slice_label = np.expand_dims(slice_label, 2).repeat(3, axis=2)  # H W C
                                # slice_label = zoom(slice_label, (256 / x, 256 / y, 1), order=0)
                                if self.num_classes > 2:
                                    for cls in range(0, self.num_classes):
                                        encoder_posterior = self.encode_first_stage(
                                            torch.from_numpy(slice_label == cls).unsqueeze(0).permute((0, 3, 1, 2)).half().cuda()
                                        )
                                        label_latent = self.get_first_stage_encoding(encoder_posterior).detach()
                                        label_latent_list.append(label_latent)  # 1 4 32 32
                                else:
                                    encoder_posterior = self.encode_first_stage(
                                        torch.from_numpy(slice_label).unsqueeze(0).permute((0, 3, 1, 2)).half().cuda()
                                    )
                                    label_latent = self.get_first_stage_encoding(encoder_posterior).detach()
                                    label_latent_list.append(label_latent)  # 1 4 32 32
                                    
                                samples_latent_list.extend(samples_pred[0:])    # 1 4 32 32
                                label_image_list.append(
                                    torch.from_numpy(colorize(slice_label, num_classes=self.num_classes))
                                    .unsqueeze(0).permute(0, 3, 1, 2))    # 1 3 256 256
                                samples_image_list.append(
                                    torch.from_numpy(colorize(out, num_classes=self.num_classes))
                                    .unsqueeze(0).permute(0, 3, 1, 2))  # 1 3 256 256
                                
                                samples_logits_list.append(torch.from_numpy(out_p * 255).unsqueeze(0).permute(0, 3, 1, 2))
                                samples_cond_list.append(torch.from_numpy((slice+1)/2*255).unsqueeze(0).unsqueeze(1).repeat((1, 3, 1, 1)))

                            pbar_sub.update()
                        pbar_sub.close()

                    else:   # for 2D slices inference
                        slice = image
                        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
                        label = label[:, :, 0]

                        c = dict(
                            c_concat=[self.get_learned_conditioning(prepare_for_first_stage(input))],
                            c_crossattn=[None]
                        )
                        samples_pred = list()
                        if use_direct:
                            noise = default(noise, lambda: torch.randn_like(c["c_concat"][0]))
                            final_t = torch.tensor([self.num_timesteps - 1], device=self.device).long()
                            if self.num_classes > 2:    # multi class segmentation
                                for cls in range(0, self.num_classes):  # predict once for each class
                                    c["c_crossattn"] = [torch.tensor([cls], device=self.device)]   # cls_id
                                    model_output = self.apply_model(noise, final_t, c)
                                    pred_tmp = self.predict_start_from_noise(noise, final_t, noise=model_output)
                                    samples_pred.append(pred_tmp)
                            else:
                                model_output = self.apply_model(noise, final_t, c)
                                pred_tmp = self.predict_start_from_noise(noise, final_t, noise=model_output)
                                samples_pred.append(pred_tmp)
                        else:
                            # for cls in range(0, self.num_classes):
                            #     c["c_crossattn"] = [torch.tensor([cls], device=self.device)]
                            pred_tmp, _ = sampler.sample(
                                S=ddim_steps,
                                conditioning=c,
                                shape=(self.channels, self.image_size, self.image_size),
                                batch_size=1,
                                verbose=False,
                                unconditional_guidance_scale=1.0,  # CT slice takes control
                                unconditional_conditioning=None,  # dont need unconditional result
                                eta=1.,
                                x_T=None
                            )
                            samples_pred.append(pred_tmp)

                        out = torch.zeros((256, 256, self.num_classes))     # h w num_classes
                        if self.num_classes > 2:
                            for cls in range(0, self.num_classes):
                                x_samples_ddim = self.decode_first_stage(samples_pred[cls])
                                if cls == 0:
                                    x_samples_ddim *= -1    # for softmax
                                x_samples_ddim = torch.mean(x_samples_ddim, dim=1, keepdim=False)    # b h w
                                out[:, :, cls] = x_samples_ddim[0, ...]
                            out_p = out.softmax(dim=2)
                            out = out_p.argmax(dim=2, keepdim=True).repeat(1, 1, 3).numpy()    # h w c==3
                        else:
                            x_samples_ddim = self.decode_first_stage(samples_pred[0])
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0 , min=0.0, max=1.0
                            )  
                            # x_samples_ddim = (x_samples_ddim + 1.0) / 2.0
                            # channel-wise average, can not use for colored mode:
                            x_samples_ddim = x_samples_ddim.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                            out_p = rearrange(x_samples_ddim.squeeze(0).cpu().numpy(), 'c h w -> h w c')
                            out = (out_p > 0.5)
                        pbar.set_postfix(dict(
                            label_cls=set(list(label.flatten().astype(int))),
                            pred_cls=set(list(out.flatten().astype(int)))
                        )
                        )
                        prediction = out[:, :, 0]

                        if save_dir is not None:
                            slice_name = slice_path[0].split("/")[-1]
                            save_pred_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-gts", slice_name.split(".")[-1]]))
                            save_logits_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-logits", slice_name.split(".")[-1]]))
                            save_all_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-all", slice_name.split(".")[-1]]))
                            
                            save_pred = (out*255).astype(np.uint8)
                            save_logits = (out_p*255).astype(np.uint8)
                            save_gt = np.expand_dims((label*255).astype(np.uint8), 2).repeat(3, axis=2)
                            save_cond = ((slice+1)/2*255).astype(np.uint8)
                            save_all = np.concatenate((save_cond, save_gt, save_pred, save_logits), axis=1)
                            
                            # Image.fromarray(save_pred).save(save_pred_path)
                            # Image.fromarray(save_all).save(save_all_path)
                            Image.fromarray(save_logits).save(save_logits_path)
                            

                        # log non-empty examples for debugging
                        if  pbar.n % (pbar.total // 8) == 0:
                            slice_label = np.expand_dims(label, 2).repeat(3, axis=2)  # H W C
                            # slice_label = zoom(slice_label, (256 / x, 256 / y, 1), order=0)
                            if self.num_classes > 2:
                                for cls in range(0, self.num_classes):
                                    encoder_posterior = self.encode_first_stage(
                                        torch.from_numpy(label == cls).unsqueeze(0).permute((0, 3, 1, 2)).half().cuda()
                                    )
                                    label_latent = self.get_first_stage_encoding(encoder_posterior).detach()
                                    label_latent_list.append(label_latent)  # 1 4 32 32
                            else:
                                encoder_posterior = self.encode_first_stage(
                                    torch.from_numpy(slice_label).unsqueeze(0).permute((0, 3, 1, 2)).half().cuda()
                                )
                                label_latent = self.get_first_stage_encoding(encoder_posterior).detach()
                                label_latent_list.append(label_latent)  # 1 4 32 32
                                
                            samples_latent_list.extend(samples_pred[0:])    # 1 4 32 32
                            label_image_list.append(
                                torch.from_numpy(colorize(slice_label, num_classes=self.num_classes))
                                .unsqueeze(0).permute(0, 3, 1, 2))    # 1 3 256 256
                            samples_image_list.append(
                                torch.from_numpy(colorize(out, num_classes=self.num_classes))
                                .unsqueeze(0).permute(0, 3, 1, 2))  # 1 3 256 256
                            
                            samples_logits_list.append(torch.from_numpy(out_p * 255).unsqueeze(0).permute(0, 3, 1, 2))
                            samples_cond_list.append(torch.from_numpy((slice+1)/2*255).unsqueeze(0).permute(0, 3, 1, 2))

                    # prediction = zoom(prediction, (x / 256, y / 256), order=0)   # H W
                    # label = zoom(label, (x / 256, y / 256), order=0)   # H W
                            
                    metrics_list = [[], []]
                    label = label.round().astype(int)
                    for idx in range(1, self.num_classes):
                        metrics_list[0].append(dice_score(prediction == idx, label == idx))
                        metrics_list[1].append(iou_score(prediction == idx, label == idx))
                    dice_list += np.array(metrics_list[0])
                    iou_list += np.array(metrics_list[1])
                pbar.close()

                try:
                    seg_label_pair = [
                        self.prepare_latent_to_log(
                            torch.cat((
                                torch.cat(samples_latent_list, dim=0),
                                torch.cat(label_latent_list, dim=0)),
                                dim=1).float()
                        ),
                            (torch.cat((
                                torch.cat(samples_logits_list, dim=3),
                                torch.cat(samples_image_list, dim=3),
                                torch.cat(label_image_list, dim=3),
                                torch.cat(samples_cond_list, dim=3),),
                                dim=2)/255.*2.).float()-1.
                        ]
                except NotImplementedError:     # inference stage doesn't log
                    seg_label_pair = [list(), list()]

                dice_list = dice_list / len(data)
                for idx in range(1, self.num_classes):
                    print(f"\033[31m[Mean Dice][cls {idx}]: {dice_list[idx-1]}\033[0m")

                iou_list = iou_list / len(data)
                for idx in range(1, self.num_classes):
                    print(f"\033[31m[Mean  IoU][cls {idx}]: {iou_list[idx-1]}\033[0m")

                return dice_list, iou_list, seg_label_pair

            if used_sampler == "plms":
                sampler = PLMSSampler(self)
            elif used_sampler == "ddim":
                sampler = DDIMSampler(self)
            elif used_sampler == "direct":
                sampler = None
            else:
                raise NotImplementedError()

            precision_scope = autocast
            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.ema_scope(f"EMA Seg Validation ({used_sampler})"):
                        ema_dice_list, ema_iou_list, seg_label_pair = get_dice_loop(data, sampler,
                                                           use_direct=True if used_sampler == "direct" else False,
                                                           save_dir=save_dir, ddim_steps=ddim_steps)
            return ema_dice_list, ema_iou_list, seg_label_pair

        # try dice methods
        ema_dice, ema_iou, seg_label_pair = get_dice(data, used_sampler="direct", save_dir=save_dir)
        multi_dice, multi_iou = np.array(ema_dice), np.array(ema_iou)
        metrics_dict.update({"val_avg_dice/direct_ema": np.mean(multi_dice)})
        metrics_dict.update({"val_avg_iou/direct_ema": np.mean(multi_iou)})
        for cls in range(1, self.num_classes):
            metrics_dict.update({f"val_avg_dice/direct_ema_{cls}": multi_dice[cls-1]})
        for cls in range(1, self.num_classes):
            metrics_dict.update({f"val_avg_iou/direct_ema_{cls}": multi_iou[cls-1]})
        seg_label_dict.update({"latent_seg_label-direct_ema": seg_label_pair[0],
                               "image_seg_label-direct-ema": seg_label_pair[1]})
        
        # print("ddim steps: ", steps)
        # ema_dice, ema_iou, seg_label_pair = get_dice(data, used_sampler="ddim", save_dir=save_dir, ddim_steps=steps)
        # metrics_dict.update({"val_avg_dice/ddim_ema": np.mean(np.array(ema_dice))})
        # metrics_dict.update({"val_avg_iou/ddim_ema": np.mean(np.array(ema_iou))})
        # seg_label_dict.update({"latent_seg_label-ddim_ema": seg_label_pair[0],
        #                     "image_seg_label-ddim-ema": seg_label_pair[1]})


        # choose one as the segmentation monitor
        metrics_dict.update({"val_avg_dice": list(multi_dice)})
        metrics_dict.update({"val_avg_iou": list(multi_iou)})

        # self.model.train()    # ImageLogger will handle this
        return metrics_dict, seg_label_dict

    @staticmethod
    @torch.no_grad()
    def prepare_latent_to_log(latent):
        # expected input shape: b c h w -> b c 1 h w == n_log_step, n_row, C, H, W
        latent = latent.unsqueeze(2)
        latent_grid = rearrange(latent, 'n b c h w -> b n c h w')
        latent_grid = rearrange(latent_grid, 'b n c h w -> (b n) c h w')
        return make_grid(latent_grid, nrow=latent.shape[0])

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=True, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):  # TODO: ddim_steps

        use_ddim = ddim_steps is not None
        if use_ddim:
            ddim_steps = self.num_timesteps // 5

        log = dict()
        # z: seg-map after autoencoder encode
        # c: CT slice after autoencoder encode
        # _: original seg-map (input of autoencoder)
        # x: original seg-map (output of autoencoder)
        # xrec: autoencoder decode output of z
        # xc: the CT slice image
        z, c, _, cls_id, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                              return_first_stage_outputs=True,
                                              force_c_encode=True,
                                              return_original_cond=True,
                                              bs=N)
        print(f"[logging class ID]: {cls_id.detach().cpu()}")
        c = dict(c_concat=[c], c_crossattn=[cls_id])

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["latent"] = self.prepare_latent_to_log(z)
        log["reconstruction"] = xrec
        # latent_seg = self.latent2seg(z).to(float)
        # latent_label = self.x2label(x).to(float)
        # log["latent_seg_label"] = self.prepare_latent_to_log(
        #     torch.cat((latent_seg, latent_label), dim=1)
        # )
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)  # not using
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):  # used for CT slice
                log["conditioning"] = xc
                log["conditioning_latent"] = self.prepare_latent_to_log(c["c_concat"][0])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            diffusion_row_latent = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row_latent.append(z_noisy)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

            diffusion_row_latent = torch.stack(diffusion_row_latent)  # n_log_step, n_row, C, H, W
            diffusion_grid_latent = rearrange(diffusion_row_latent, 'n b c h w -> b n c h w')
            diffusion_grid_latent = rearrange(diffusion_grid_latent, 'b n c h w -> (b n) c h w')
            diffusion_grid_latent = make_grid(diffusion_grid_latent, nrow=diffusion_row_latent.shape[0])
            log["diffusion_row_latent"] = diffusion_grid_latent

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            log["samples_latent"] = samples
            if plot_denoise_rows:
                denoise_grid = self.get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            """ only VQ-VAE can quantized denoise, will be ignored in KL-VAE and DDIM """
            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            """ useless for segmentation task """
            # if inpaint:
            #     # make a simple center square
            #     b, h, w = z.shape[0], z.shape[2], z.shape[3]
            #     mask = torch.ones(N, h, w).to(self.device)
            #     # zeros will be filled in
            #     mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            #     mask = mask[:, None, ...]
            #     with self.ema_scope("Plotting Inpaint"):
            #         samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
            #                                      ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_inpainting"] = x_samples
            #     log["mask"] = mask
            #
            #     # outpaint
            #     with self.ema_scope("Plotting Outpaint"):
            #         samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
            #                                      ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            #     x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row, prog_row_latent = self.get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row
            log["progressive_row_latent"] = prog_row_latent

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        # the whole unet except `label_embed`
        params_dict = [{"params": self.model.diffusion_model.input_blocks.parameters()}] + \
                      [{"params": self.model.diffusion_model.middle_block.parameters()}] + \
                      [{"params": self.model.diffusion_model.output_blocks.parameters()}] + \
                      [{"params": self.model.diffusion_model.out.parameters()}] + \
                      [{"params": self.model.diffusion_model.time_embed.parameters()}]
        if "label_emb" in self.unet_sd_keys:    # multi-class, including label embeddings
            print(f"{self.__class__.__name__}: Also optimizing multi-class label-embedding params!")
            params_dict.append({"params": self.model.diffusion_model.label_emb.parameters(), "lr": lr * 100})
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params_dict.append(
                {"params": self.cond_stage_model.parameters(), "lr": lr * 1}    # original: 4.5e-6
            )
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params_dict.append(
                {"params": self.logvar}
            )
        opt = torch.optim.AdamW(params_dict, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':     # for multi-class segmentation
            xc = torch.cat([x] + c_concat, dim=1)
            cc = c_crossattn[0]                     # modified
            # cc = torch.cat(c_crossattn, dim=1)    # original
            out = self.diffusion_model(xc, t, y=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs