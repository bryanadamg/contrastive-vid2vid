# Unpaired Video-to-Video Translation with Contrastive Learning in PyTorch

PyTorch implementations for unpaired video-to-video translation.

The code was written by [Bryan Adam Gunawan](https://github.com/bryanadamg).


**Note**: The current software works well with PyTorch 1.4. Check out the older [branch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1) that supports PyTorch 0.1-0.3.



## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/bryanadamg/contrastive-vid2vid
cd contrastive-vid2vid
```

- Install [PyTorch](http://pytorch.org) and other dependencies

### Train/Test
- Download dataset (e.g. maps):
- Train a CUT model:
```bash
python train.py --dataroot ./datasets/utopilot_sun2rain_downscaled --name utopilot_sun2rain_reduced --CUT_mode CUT --dataset_mode unaligned_triplet --load_size 270 --crop_size 256 --batch_size 2
python train.py --gpu_ids -1 --dataroot ./datasets/utopilot_sun2rain_downscaled --netG swin_unet --crop_size 224 --name test1 --CUT_mode CUT --dataset_mode unaligned_triplet --model swin_unet_cut --display_id -1 --num_threads 0
python train.py --dataroot /root/autodl-fs/utopilot_sun2rain/ --netG swin_unet --crop_size 224 --name first_test --CUT_mode CUT --dataset_mode unaligned_triplet --model swin_unet_cut
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/docker_dataset --name docker --CUT_mode CUT --dataset_mode unaligned_triplet --phase train
python test.py --dataroot ./datasets/utopilot_sun2rain_downscaled --gpu_ids -1 --netG swin_unet --name fourth_test --CUT_mode CUT --dataset_mode unaligned_triplet --model swin_unet_cut --num_threads 0 --phase test --num_test 300 --crop_size 224 --load_size 224 --preprocess resize --epoch 80
python test.py --dataroot ./datasets/utopilot_sun2rain_downscaled --gpu_ids -1 --name utopilot_sun2rain_reduced --CUT_mode CUT --dataset_mode unaligned_triplet --num_threads 0 --phase test --num_test 300
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.
- Local ssh to view visdom server:
```bash
ssh -CNgv -L 8097:127.0.0.1:8097 root@connect.southb.gpuhub.com -p 38759
```

### Evaluate FID

```bash
python -m pytorch_fid [path to real test images] [path to generated images]
```





## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.


## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

Code adapted from:
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)


## Related Projects
**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**<br>
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**<br>
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**

## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
