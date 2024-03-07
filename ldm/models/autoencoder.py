import sys

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from einops import rearrange, repeat
from torchvision.utils import make_grid

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"\033[32mRestored VQ-first-stage-model from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys\033[0m")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 num_classes=2,
                 ):
        super().__init__()
        self.image_key = image_key
        self.num_classes = num_classes
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, ddconfig=ddconfig)

    def init_from_ckpt(self, path, ignore_keys=list(), ddconfig=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        # detect and init multi-class input-output layer:
        # model_in, model_out = ddconfig.in_channels, ddconfig.out_ch
        # sd_in, sd_out = sd["encoder.conv_in.weight"].shape, sd["decoder.conv_out.weight"].shape
        # # print(sd_in, sd["encoder.conv_in.bias"].shape)
        # # print(sd_out, sd["decoder.conv_out.bias"].shape)
        # assert model_in == model_out and sd_in[1] == sd_out[0]
        # if model_in != sd_in[1] and self.num_classes > 2:
        #     sd_in, sd_out = torch.tensor(sd_in), torch.tensor(sd_out)
        #     sd_in[1], sd_out[0] = model_in, model_out
        #     sd["encoder.conv_in.weight"] = torch.rand(tuple(sd_in))
        #     sd["encoder.conv_in.bias"] = torch.rand(sd["encoder.conv_in.bias"].shape)#.fill_(1)
        #     sd["decoder.conv_out.weight"] = torch.rand(tuple(sd_out))
        #     sd["decoder.conv_out.bias"] = torch.rand(model_out)
        #     print("\033[31m[ATT]: rand-initialize autoencoder with multi-channel input-output.\033[0m")

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"\033[32mRestored KL-first-stage-model from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys\033[0m")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        ae_param = list(self.encoder.parameters()) + \
                   list(self.decoder.parameters()) + \
                   list(self.quant_conv.parameters()) + \
                   list(self.post_quant_conv.parameters())
        opt_ae = torch.optim.Adam(ae_param,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            samples = self.decode(torch.randn_like(posterior.sample()))
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
                samples = self.to_rgb(samples)
            log["samples"] = samples
            log["reconstructions"] = xrec
            log["latent"] = self.prepare_latent_to_log(latent=posterior.sample())
        log["inputs"] = x
        return log

    @staticmethod
    def prepare_latent_to_log(latent):
        # expected input shape: b c h w -> b c 1 h w == n_log_step, n_row, C, H, W
        latent = latent.unsqueeze(2)
        latent_grid = rearrange(latent, 'n b c h w -> b n c h w')
        latent_grid = rearrange(latent_grid, 'b n c h w -> (b n) c h w')
        return make_grid(latent_grid, nrow=latent.shape[0])

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class OnehotAutoencoderKL(AutoencoderKL):  # for multi-class segmentation
    def __init__(self, *args, **kwargs):
        super(OnehotAutoencoderKL, self).__init__(*args, **kwargs)

        # mapping between one-hot and label
        # self.onehot2label_conv = torch.nn.Sequential(
        #     torch.nn.Identity()
        # )
        self.label2onehot_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),  # depth-wise convolution
            torch.nn.Conv2d(3, self.num_classes, kernel_size=1, stride=1, padding=0, groups=1),  # pixel-wise convolution
            # V2:
            torch.nn.BatchNorm2d(self.num_classes),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=1, padding=1, groups=self.num_classes)
        )

        self.loss = torch.nn.CrossEntropyLoss()

        def disabled_train(self, mode=True):
            """Overwrite model.train with this function to make sure train/eval mode
            does not change anymore."""
            return self

        # froze original autoencoder
        self.encoder.eval()
        self.encoder.train = disabled_train
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder.eval()
        self.decoder.train = disabled_train
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.quant_conv.eval()
        self.quant_conv.train = disabled_train
        for param in self.quant_conv.parameters():
            param.requires_grad = False

        self.post_quant_conv.eval()
        self.post_quant_conv.train = disabled_train
        for param in self.post_quant_conv.parameters():
            param.requires_grad = False

        self.loss.eval()
        self.loss.train = disabled_train
        for param in self.loss.parameters():
            param.requires_grad = False

    def forward(self, input, sample_posterior=True):
        with torch.no_grad():
            posterior = self.encode(input)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            dec = self.decode(z)
        dec = self.label2onehot_conv(dec)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)  # 3 -> in
        reconstructions, posterior = self(inputs)       # out -> 14

        target = ((inputs[:, 0, ...]+1)/2*(self.num_classes-1)).long()
        loss = self.loss(reconstructions, target)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        target = ((inputs[:, 0, ...]+1)/2*(self.num_classes-1)).long()
        loss = self.loss(reconstructions, target)
        self.log("val/rec_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_param = list(self.label2onehot_conv.parameters())
        opt_ae = torch.optim.Adam(ae_param,
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_onehot = self.get_input(batch, "segmentation_onehot")
        x = x.to(self.device)
        x_onehot = x_onehot.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            samples = self.decode(torch.randn_like(posterior.sample()))
            xrec, samples = self.onehot_to_label(xrec), self.onehot_to_label(samples)  # [0, num_classes-1]
            log["samples"] = samples / (self.num_classes-1) * 2 - 1  # [-1, 1]
            log["reconstructions"] = xrec / (self.num_classes-1) * 2 - 1  # [-1, 1]

            xrec = F.one_hot(xrec[:, 0, ...].long(),
                             num_classes=self.num_classes).permute(0, 3, 1, 2)
            samples = F.one_hot(samples[:, 0, ...].long(),
                                num_classes=self.num_classes).permute(0, 3, 1, 2)
            log["samples_onehot"] = self.to_col(samples) * 2 - 1
            log["reconstructions_onehot"] = self.to_col(xrec) * 2 - 1
            log["latent"] = self.prepare_latent_to_log(latent=posterior.sample())
        log["inputs"] = x
        log["inputs_onehot"] = self.to_col(x_onehot) * 2 - 1
        return log

    def to_col(self, x):
        assert self.image_key == "segmentation"
        return torch.cat(tuple(x[:, i:i + 1, ...] for i in range(x.shape[1])), dim=2)

    @staticmethod
    def onehot_to_label_with_grad(x):
        # x: tensor, b class h w
        # y: tensor, b 3 h w
        x_softmax = x.softmax(dim=1)
        x_argmax = x_softmax.argmax(dim=1, keepdim=True)
        x_softmax_sum = x_softmax.sum(dim=1, keepdim=True) / 3  # will repeat 3 times when return
        grad_c = x_argmax - x_softmax_sum  # contains sum of grad of softmax
        return (x_softmax_sum + grad_c).repeat((1, 3, 1, 1)).float()

    @staticmethod
    def onehot_to_label(x):
        # x: tensor, b class h w
        # y: tensor, b 3 h w
        return x.softmax(dim=1).argmax(dim=1, keepdim=True).repeat((1, 3, 1, 1)).float()


class Nonlinearity(torch.nn.Module):  # reference: autoencoder.encoder.resblock
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ZoomAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 status="",  # first stage or conditioning
                 colorize_nlabels=None,
                 monitor=None,
                 upsample_method="interpolation",
                 downsample_method="interpolation",
                 zoom_out_op="identity"
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # << super resolution blocks >>
        self.up_methods = dict(
            interpolation=lambda x: F.interpolate(x, scale_factor=2., mode="nearest"),
            # TODO
        )
        self.down_methods = dict(
            maxpool=torch.nn.MaxPool2d(kernel_size=2, stride=2),
            interpolation=lambda x: F.interpolate(x, scale_factor=.5, mode="nearest")
            # TODO
        )
        self.zoom_block = lambda *args, **kwargs: torch.nn.Sequential(
            # torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(3),
            # torch.nn.Identity()
        )
        self.zoom_in_upsample = self.up_methods[upsample_method]
        self.zoom_in_HR_block = self.zoom_block()
        self.zoom_in_downsample = self.down_methods[downsample_method]
        self.zoom_in_LR_block = dict(identity=torch.nn.Identity(), __zoom_in__=self.zoom_block())[zoom_out_op]
        self.zoom_out_upsample = self.up_methods[upsample_method]
        self.zoom_out_HR_block = self.zoom_block()
        self.zoom_out_downsample = self.down_methods[downsample_method]
        self.zoom_out_LR_block = dict(identity=torch.nn.Identity(), __zoom_in__=self.zoom_block())[zoom_out_op]

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, status=status)

    def init_from_ckpt(self, path, ignore_keys=list(), status=""):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"\033[32mRestored {status} Model from {path}\033[0m")
        print(f"[{self.__class__.__name__}][miss keys]: {missing_keys}")
        print(f"[{self.__class__.__name__}][unexpected keys]: {unexpected_keys}")

    def encode(self, x):
        x = self.zoom_encoder(x)  # zoom_in in encode process! # zoom in: 256 -> 512 -> 256
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.zoom_decoder(dec)  # zoom_out in decode process!   # zoom out: 256 -> 512 -> 256
        return dec

    def zoom_encoder(self, x):  # b, 3, 256, 256
        x = self.zoom_in_upsample(x)  # 256 -> 512
        x = self.zoom_in_HR_block(x)
        x = self.zoom_in_downsample(x)  # 512 -> 256
        x = self.zoom_in_LR_block(x)
        return x

    def zoom_decoder(self, x):  # b, 3, 256, 256
        x = self.zoom_out_upsample(x)  # 256 -> 512
        x = self.zoom_out_HR_block(x)
        x = self.zoom_out_downsample(x)  # 512 -> 256
        x = self.zoom_out_LR_block(x)
        return x

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        param_base = self.set_param_lr(
            lr,
            self.encoder, self.decoder, self.quant_conv, self.post_quant_conv
        )
        param_zoom = self.set_param_lr(
            lr * 100,
            self.zoom_in_HR_block, self.zoom_in_LR_block, self.zoom_out_HR_block, self.zoom_out_LR_block
        )
        opt_ae = torch.optim.Adam(param_base + param_zoom, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def set_param_lr(self, lr, *args):
        param_list = list()
        for module in args:
            param_list.append(dict(params=module.parameters(), lr=lr))
        return param_list

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
