import argparse, os, sys, glob

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from ldm.data.synapse import SynapseValidation, SynapseValidationVolume
from ldm.data.refuge2 import REFUGE2Validation, REFUGE2Test
from ldm.data.sts3d import STS3DValidation, STS3DTest
from ldm.data.cvc import CVCValidation, CVCTest
from ldm.data.kseg import KSEGValidation, KSEGTest

from scipy.ndimage import zoom

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor


def prepare_for_first_stage(x, gpu=True):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    if gpu:
        x = x.to(memory_format=torch.contiguous_format).float().cuda()
    else:
        x = x.float()
    return x


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # if targs is None:
    #     return None
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() > 0 and targs.sum() == 0:
        return 1
    elif pred.sum() > 0 and targs.sum() > 0:
        # intersection = (pred * targs).sum()
        # union = pred.sum() + targs.sum() - intersection
        # return (2. * intersection) / (union + 10e-6)
        return (2. * (pred * targs).sum()) / (pred.sum() + targs.sum() + 1e-10)
    else:
        return 0


def iou_score(pred, targs):
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)

    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    # return intersection, union
    return intersection / (union + 1e-10)



def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    # print(set(key.split(".")[0] for key in sd.keys()))
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    print("\033[31mmissing keys:\033[0m")
    print(m)
    # if len(u) > 0 and verbose:
    print("\033[31munexpected keys:\033[0m")
    print(u)
    # model.cuda()
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["pred_sum"] > 0 and kwargs["gt_sum"] > 0:
        return 2 * inter / (union + 1e-10)
    elif kwargs["pred_sum"] > 0 and kwargs["gt_sum"] == 0:
        return 1
    else:
        return 0


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["raw", "direct", "ddim", "plms", "dpm_solver"],
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument("--dataset", type=str,  # '-b' for binary, '-m' for multi
                        help="uses the model trained for given dataset", )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--times", type=int, default=1,
                        help="times of testing for stability evaluation", )
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the predictions for the whole test set.", )
    opt = parser.parse_args()

    
    if opt.dataset == "synapse-b":
        run = "the name of your experiment"      # for example: 2024-02-13T17-09-00_binary
        print("Evaluate on synapse dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/last.ckpt"      # name of the trained model
        opt.outdir = "outputs/slice2seg-samples-synapse-b"
        dataset = SynapseValidationVolume(num_classes=2)
    elif opt.dataset == "synapse-m":
        run = "the name of your experiment" 
        print("Evaluate on synapse dataset in multi-organ segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/epoch=107-step=14999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-synapse-m"
        dataset = SynapseValidationVolume(num_classes=9)
    elif opt.dataset == "refuge2-b":
        run = "the name of your experiment" 
        print("Evaluate on refuge2 dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        # opt.ckpt = "models/ldm/synapse_binary/model.ckpt"
        opt.ckpt = f"logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "outputs/slice2seg-samples-refuge2-b"
        dataset = REFUGE2Test()
    elif opt.dataset == "sts-3d": 
        run = "the name of your experiment" 
        print("Evaluate on sts-3d dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        # opt.ckpt = "models/ldm/synapse_binary/model.ckpt"
        opt.ckpt = f"logs/{run}/checkpoints/epoch=199-step=124999.ckpt"
        opt.outdir = "outputs/slice2seg-samples-sts-3d"
        dataset = STS3DTest()
    elif opt.dataset == "cvc":
        run = "the name of your experiment" 
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "outputs/slice2seg-samples-cvc"
        dataset = CVCTest()
    elif opt.dataset == "kseg":
        run = "the name of your experiment" 
        print("Evaluate on kseg dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/model.ckpt"
        opt.outdir = "outputs/slice2seg-samples-cvc"
        dataset = KSEGTest()
    else:
        raise NotImplementedError(f"Not implement for dataset {opt.dataset}")

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    config = OmegaConf.load(f"{opt.config}")
    config["model"]["params"].pop("ckpt_path")
    config["model"]["params"]["cond_stage_config"]["params"].pop("ckpt_path")
    config["model"]["params"]["first_stage_config"]["params"].pop("ckpt_path")

    model, pl_sd = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)


    for idx in range(opt.times):
        if opt.times > 1:   # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)
        print(f"\033[32m seed:{opt.seed}\033[0m")
        
        outpath = os.path.join(opt.outdir, str(opt.seed))
        os.makedirs(outpath, exist_ok=True)

        metrics_dict, _ = model.log_dice(data=data, save_dir=outpath if opt.save_results else None) 

        dice_list = metrics_dict["val_avg_dice"]
        iou_list = metrics_dict["val_avg_iou"]
        print(f"\033[31m[Mean Dice][{opt.dataset}][direct]: {sum(dice_list) / len(dice_list)}\033[0m")
        print(f"\033[31m[Mean  IoU][{opt.dataset}][direct]: {sum(iou_list) / len(iou_list)}\033[0m")

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()
