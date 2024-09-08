<p align="center">
 <h1 align="center">Stable Diffusion Segmentation (SDSeg)</h2>
 <p align="center">The official implementation of <a href="https://arxiv.org/abs/2406.18361">Stable Diffusion Segmentation for Biomedical Images with Single-step Reverse Process</a> at <a href="https://conferences.miccai.org/2024/en/default.asp">MICCAI 2024</a>.</p>
</p>
  <p align="center">
    <a href="https://arxiv.org/abs/2406.18361"><img alt="Static Badge" src="https://img.shields.io/badge/_-arxiv-red?style=for-the-badge&logo=arxiv"></a>
    <a href="https://lin-tianyu.github.io/Stable-Diffusion-Seg/"><img alt="Static Badge" src="https://img.shields.io/badge/_-Project_-green?style=for-the-badge&logo=github"></a>
    <a href="https://arxiv.org/abs/2406.18361"><img alt="Static Badge" src="https://img.shields.io/badge/_-code-black?style=for-the-badge&logo=github"></a>
  </p>
<p align="center">
  <a href="https://lin-tianyu.github.io"><img alt="Static Badge" src="https://img.shields.io/github/stars/lin-tianyu?label=Tianyu%20Lin"></a>
    <a href=""><img alt="Static Badge" src="https://img.shields.io/badge/Zhiguang_Chen-_-white?style=social"></a>
    <a href="https://github.com/zzzyzh"><img alt="Static Badge" src="https://img.shields.io/github/stars/zzzyzh?label=Zhonghao%20Yan"></a>
    <a href="https://github.com/yuweijiang"><img alt="Static Badge" src="https://img.shields.io/github/stars/yuweijiang?label=Weijiang%20Yu"></a>
    <a href=""><img alt="Static Badge" src="https://img.shields.io/badge/Fudan_Zheng-_-white?style=social"></a>
  <br />
</p>

<!-- This is the official implementation of [**Stable Diffusion Segmentation for Biomedical Images with Single-step Reverse Process**](https://arxiv.org/abs/2406.18361) at [<b>MICCAI 2024</b>](https://conferences.miccai.org/2024/en/default.asp). -->

<!-- [![Static Badge](https://img.shields.io/badge/_-Project_Page-white?style=social&logo=github)](https://lin-tianyu.github.io/Stable-Diffusion-Seg/) \| [![Static Badge](https://img.shields.io/badge/arxiv-2406.18361-white?style=flat&logo=arxiv)](https://arxiv.org/abs/2406.18361) \| [![GitHub Repo stars](https://img.shields.io/github/stars/lin-tianyu/Stable-Diffusion-Seg?label=Code)](https://github.com/lin-tianyu/Stable-Diffusion-Seg/) -->

<!-- [![GitHub User's stars](https://img.shields.io/github/stars/lin-tianyu?label=Tianyu%20Lin)](https://lin-tianyu.github.io) | Zhiguang Chen | [![GitHub User's stars](https://img.shields.io/github/stars/zzzyzh?label=Zhonghao%20Yan)](https://github.com/zzzyzh) | [![GitHub User's stars](https://img.shields.io/github/stars/yuweijiang?label=Weijiang%20Yu)](https://github.com/yuweijiang) | Fudan Zheng -->



## 📣 News
- 07/14: We release a [![Static Badge](https://img.shields.io/badge/_-Project_Page-white?style=social&logo=github)](https://lin-tianyu.github.io/Stable-Diffusion-Seg/) for you to understand our work better. Check it out!
- 06/27: The paper of SDSeg has been pre-released on [![Static Badge](https://img.shields.io/badge/arxiv-2406.18361-white?style=flat&logo=arxiv)](https://arxiv.org/abs/2406.18361)
- 06/17: 🎉🥳 **SDSeg** has been accepted by MICCAI2024!  Our paper will be available soon.

## 📌 SDSeg Framework
<img src="assets/framework-v2.jpg" alt="framework" width="80%" height="80%" />

SDSeg is built on Stable Diffusion (V1), with a downsampling-factor 8 autoencoder, a denoising UNet, and trainable vision encoder (with the same architecture of the encoder in the f=8 autoencoder).


## ⚙️ Requirements

A suitable [conda](https://conda.io/) environment named `sdseg` can be created
and activated with:

```bash
conda env create -f environment.yaml
conda activate sdseg
```

Then, install some dependencies by:
```bash
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

    
<details>

<summary>Solve GitHub connection issues when downloading <code class="inlinecode">taming-transformers</code> or <code class="inlinecode">clip</code></summary>


After creating and entering the `sdseg` environment:
1. create an `src` folder and enter:
```bash
mkdir src
cd src
```
2. download the following codebases in `*.zip` files and upload to `src/`:
    - https://github.com/CompVis/taming-transformers, `taming-transformers-master.zip`
    - https://github.com/openai/CLIP, `CLIP-main.zip`
3. unzip and install taming-transformers:
```bash
unzip taming-transformers-master.zip
cd taming-transformers-master
pip install -e .
cd ..
```
4. unzip and install clip:
```bash
unzip CLIP-main.zip
cd CLIP-main
pip install -e .
cd ..
```
5. install latent-diffusion:
```bash
cd ..
pip install -e .
```

Then you're good to go!

</details>


## 🩻 Dataset Settings
> [!NOTE]
> The image data should be place at `./data/`, while the dataloaders are at `./ldm/data/`

We evaluate SDSeg on the following medical image datasets:

| Dataset        | URL                                                                                           | Preprocess                                         |
|----------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------|
| `BTCV`         | [This URL](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752), <br>download the `Abdomen/RawData.zip`. | Use the code in <br>`./data/synapse/nii2format.py`     |
| `STS-3D`       | [This URL](https://toothfairychallenges.github.io/), <br>download the `labelled.zip`.                             | Use the code in <br>`./data/sts3d/sts3d_preprocess.py` |
| `REFUGE2`      | [This URL](https://refuge.grand-challenge.org/)                                                           | Following [this repo](https://github.com/HzFu/MNet_DeepCDR)     |
| `CVC-ClinicDB` | [This URL](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)                                          | None                                               |
| `Kvasir-SEG`   | [This URL](https://datasets.simula.no/kvasir-seg/)                                                        | None                                               |


## 📦 Model Weights

### Pretrained Models
SDSeg uses pre-trained weights from SD to initialize before training.

For pre-trained weights of the autoencoder and conditioning model, run

```bash
bash scripts/download_first_stages_f8.sh
```

For pre-trained wights of the denoising UNet, run

```bash
bash scripts/download_models_lsun_churches.sh
```

### Trained SDSeg Models
> The model weights trained on medical image datasets will be available soon.

## 📄 Scripts
### Training Scripts

Take CVC dataset as an example, run

```bash
nohup python -u main.py --base configs/latent-diffusion/cvc-ldm-kl-8.yaml -t --gpus 0, --name experiment_name > nohup/experiment_name.log 2>&1 &
```

You can check the training log by 

```bash
tail -f nohup/experiment_name.log
```

Also, tensorboard will be on automatically. You can start a tensorboard session with `--logdir=./logs/`

> [!NOTE]
> If you want to use parallel training, the code `trainer_config["accelerator"] = "gpu"` in `main.py` should be changed to `trainer_config["accelerator"] = "ddp"`. However, parallel training is not recommended. 

> [!WARNING]
> A single SDSeg model ckeckpoint is around 5GB. By default, save only the last model and the model with the highest dice score. If you have tons of storage space, feel free to save more models by increasing the `save_top_k` parameter in `main.py`.



### Testing Scripts

After training an SDSeg model, you should **manually modify the run paths** in `scripts/slice2seg.py`, and begin an inference process like

```bash
python -u scripts/slice2seg.py --dataset cvc
```



### Stability Evaluaition

To conduct an stability evaluation process mentioned in the paper, you can start the test by

```bash
python -u scripts/slice2seg.py --dataset cvc --times 10 --save_results
```

This will save 10 times of inference results in `./outputs/` folder. To run the stability evaluation, open `scripts/stability_evaluation.ipynb`, and **modify the path for the segmentation results**. Then, click `Run All` and enjoy.



## ‼️ Important Files and Folders to Focus on
Training related:
- SDSeg model: `./ldm/models/diffusion/ddpm.py` in the class `LatentDiffusion`.
- Experiment Configurations: `./configs/latent-diffusion`

Inference related:
- Inference starting scripts: `./scripts/slice2seg.py`, 
- Inference implementation: `./ldm/models/diffusion/ddpm.py`, under the `log_dice` method of `LatentDiffusion`.

Dataset related:
- Dataset storation: `./data/`
- Dataloader files: `./ldm/data/`

## 📝 Citation
If you find our work useful, please cite:
```bibtex
@article{lin2024stable,
  title={Stable Diffusion Segmentation for Biomedical Images with Single-step Reverse Process},
  author={Lin, Tianyu and Chen, Zhiguang and Yan, Zhonghao and Yu, Weijiang and Zheng, Fudan},
  journal={arXiv preprint arXiv:2406.18361},
  year={2024}
}
```
```bibtex
@inproceedings{lin2024stable,
  title={Stable Diffusion Segmentation for Biomedical Images with Single-step Reverse Process},
  author={Lin, Tianyu and Chen, Zhiguang and Yan, Zhonghao and Yu, Weijiang and Zheng, Fudan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024},
  organization={Springer}
}
```


## 🔜 TODO List

- [ ] Organizing the inference code. (Toooo redundant right now.)
- [ ] Reimplement SDSeg in OOP. (Elegance is the key!)
- [ ] Add README for multi-class segmentation.
- [ ] Release model weights.
- [ ] Reimplement using diffusers.
- [ ] Reduce model checkpoint size (no need to save autoencoder's weights).

<!-- ## ✨ Star History -->

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=lin-tianyu/Stable-Diffusion-Seg&type=Date)](https://star-history.com/#lin-tianyu/Stable-Diffusion-Seg&Date) -->

