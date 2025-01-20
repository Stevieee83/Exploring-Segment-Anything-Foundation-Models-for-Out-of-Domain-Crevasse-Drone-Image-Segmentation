# Exploring Segment Anything Foundation Models for Out-of-Domain Crevasse Drone Image Segmentation Development Repository

The Segment Anything Model (SAM) and Segment Anything Model 2 (SAM 2) are foundation image segmentation models. SAM 2 can also be used to segment video data.

<div align="center">

# SAM/SAM 2 Image Segmentation Model Architecture

</div>

![SAM/SAM 2 Image Segmentation Model Architecture](./image/sam-sam-2-architecture.png?raw=true)

## Installation and Use

Please follow the installation instructions on the SAM and SAM 2 GitHub repositories to install the required Python packages and download the SAM or SAM 2 model weights files. A link to the SAM and SAM 2 GitHub repositories are provided at the following links:

[SAM GitHub Repository](https://github.com/facebookresearch/segment-anything)

[SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2)

The SAM and SAM 2 Python code used for our experiments has been included in this GitHub repository. After cloning the repository, and installing the required Python packages with the requirements.txt file from our repository, the following steps must be carried out to run our code.

In the data folder, add the image data and segmentation labels to their respective folder.

In the checkpoints folder, store the model weights file/s for the SAM or SAM 2 model that you would like to use.

In the sam2_configs folder, store the config .yaml for the SAM 2 model you would like to use. This step is not required for SAM as the scripts run with a .yaml config file.

We have provided the file path directories used for our experiments. If you are changing the file path directories in any way please update them in the sam_background_foreground.py, sam2_background_foreground.py, sam_everything.py or sam2_everything.py you are using.

After all of the steps above have been carried out run the Python script you are using in the same way you would run any other Python script by parsing in the required arguments. An example to read in image number 1 for the largest SAM and SAM 2 ViT models has been provided.

### SAM Background/Foreground Inference Mode Test in Single Mask Mode Example

    python sam_background_foreground.py --file_number=1 --multi_mask=False --sam_checkpoint='./checkpoints/sam_vit_h_4b8939.pth' --model_type='vit_h'

### SAM Mask Generator Inference Mode Test Example

    python sam_everything.py --file_number=1 --sam_checkpoint='./checkpoints/sam_vit_h_4b8939.pth' --model_type='vit_h'

### SAM 2 Background/Foreground Inference Mode Test in Single Mask Mode Example

    python sam2_everything.py --file_number=1 --sam2_checkpoint='./checkpoints/sam2_hiera_large.pt' --model_cfg='./sam2_configs/sam2_hiera_l.yaml'

### SAM 2 Mask Generator Inference Mode Test Example

    python sam2_background_foreground.py --file_number=1 --multi_mask=False --sam2_checkpoint='./checkpoints/sam2_hiera_large.pt' --model_cfg='./sam2_configs/sam2_hiera_l.yaml'

To run either the sam_background_foreground.py or sam2_background_foreground.py Python scripts with the models in the multi-mask mode change the multi_mask argument to --multi_mask=True

To cite the work, please use the BibTex citation with the Meta-FAIR reference below.

## Conference Paper BibTex Citation

    @inproceedings{wallace2025exploring,
    title={Exploring Segment Anything Foundation Models for Out of Domain Crevasse Drone Image Segmentation},
    author={Wallace, Steven and Durrant, Aiden and Harcourt, William and Hann, Richard and Leontidis, Georgios},
    booktitle={Proceedings of the 6th Northern Lights Deep Learning Conference (NLDL)},
    year={2025}
    }

## References

    @article{kirillov2023segany,
      title={Segment Anything},
      author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
      journal={arXiv:2304.02643},
      year={2023}
    }

    @article{ravi2024sam2,
      title={SAM 2: Segment Anything in Images and Videos},
      author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan             Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
      journal={arXiv preprint arXiv:2408.00714},
      url={https://arxiv.org/abs/2408.00714},
      year={2024}
    }
