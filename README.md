# HiNet: Deep Image Hiding by Invertible Network
This repo is the official code for

* [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html) 
  * [*Junpeng Jing*](https://tomtomtommi.github.io/), [*Xin Deng*](http://www.commsp.ee.ic.ac.uk/~xindeng/), [*Mai Xu*](http://shi.buaa.edu.cn/MaiXu/zh_CN/index.htm), [*Jianyi Wang*](http://buaamc2.net/html/Members/jianyiwang.html), [*Zhenyu Guan*](http://cst.buaa.edu.cn/info/1071/2542.htm).

Published on [**ICCV 2021**](http://iccv2021.thecvf.com/home).
By [MC2 Lab](http://buaamc2.net/) @ [Beihang University](http://ev.buaa.edu.cn/).

<center>
  <img src=https://github.com/TomTomTommi/HiNet/blob/main/HiNet.png width=60% />
</center>
 
## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.0.1](https://pytorch.org/) .
- See [environment.yml](https://github.com/TomTomTommi/HiNet/blob/main/environment.yml) for other dependencies.


## Get Started
- Run `python train.py` for training.

- Run `python test.py` for testing.

- Set the model path (where the trained model saved) and the image path (where the image saved during testing) to your local path. 

    `line45:  MODEL_PATH = '' ` 

    `line49:  IMAGE_PATH = '' ` 

## Dataset
- In this paper, we use the commonly used dataset DIV2K, COCO, and ImageNet.

- For train or test on your own dataset, change the code in `config.py`:

    `line30:  TRAIN_PATH = '' ` 

    `line31:  VAL_PATH = '' `


## Trained Model
- Here we provide a trained [model](https://drive.google.com/drive/folders/1l3XBFYPMaNFdvCWyOHfB2qIPkpjIxZgE?usp=sharing).

- Fill in the `MODEL_PATH` and the file name `suffix` before testing by the trained model.

- For example, if the model name is `model.pt` and its path is `/home/usrname/Hinet/model/`, 
set `MODEL_PATH = '/home/usrname/Hinet/model/'` and file name `suffix = 'model.pt'`.


## Training Demo (2021/12/25 Updated)
- Here we provide a training demo to show how to train a converged model in the early training stage. During this process, the model may suffer from explosion. Our solution is to stop the training process at a normal node and abate the learning rate. Then, continue to train the model.

- Note that in order to log the training process, we have imported `logging` package, with slightly modified `train_logging.py` and `util.py` files.


- Stage1: 
  Run `python train_logging.py` for training with initial `config.py` (learning rate=10^-4.5).
  
  The logging file is [train__211222-183515.log](https://github.com/TomTomTommi/HiNet/blob/main/logging/train__211222-183515.log).
  (The values of r_loss and g_loss are reversed due to a small bug, which has been debuged in stage2.)
  <br/>
  <br/>
  See the tensorboard:
  <br/>
  <img src=https://github.com/TomTomTommi/HiNet/blob/main/logging/stage1.png width=60% />
  <br/>
  <br/>
  Note that in the 507-th epoch the model exploded. Thus, we stop the stage1 at epoch 500.


- Stage2: 
  Set `suffix = 'model_checkpoint_00500.pt'` and `tain_next = True` and `trained_epoch = 500`.
  
  Change the learning rate from 10^-4.5 to 10^-5.0.
  
  Run `python train_logging.py` for training.
  <br/>
  The logging file is [train__211223-100502.log](https://github.com/TomTomTommi/HiNet/blob/main/logging/train__211223-100502.log).
  <br/>
  <br/>
  See the tensorboard:
  <br/>
  <img src=https://github.com/TomTomTommi/HiNet/blob/main/logging/stage2.png width=60% />
  <br/>
  <br/>
  Note that in the 1692-th epoch the model exploded. Thus, we stop the stage2 at epoch 1690.


- Stage3: 
  Similar operation.
  
  Change the learning rate from 10^-5.0 to 10^-5.2.
  
  The logging file is [train__211224-105010.log](https://github.com/TomTomTommi/HiNet/blob/main/logging/train__211224-105010.log).
  <br/>
  <br/>
  See the tensorboard:
  <br/>
  <img src=https://github.com/TomTomTommi/HiNet/blob/main/logging/stage3.png width=60% />
  <br/>
  <br/>
  We can see that the network has initially converged. Then, you can change the super-parameters lamda according to the PSNR to balance the quality between stego image and recovered image. Note that the PSNR in the tensorboard is RGB-PSNR and in our paper is Y-PSNR.


## Others
- The `batchsize_val` in `config.py` should be at least `2*number of gpus` and it should be divisible by number of gpus.

## Citation
If you find our paper or code useful for your research, please cite:
```
@InProceedings{Jing_2021_ICCV,
    author    = {Jing, Junpeng and Deng, Xin and Xu, Mai and Wang, Jianyi and Guan, Zhenyu},
    title     = {HiNet: Deep Image Hiding by Invertible Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4733-4742}
}

```
