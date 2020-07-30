# Installation

This document contains detailed instructions for installing the necessary dependencies for FCOT. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh).  
(We refer to [Pytracking](https://github.com/visionml/pytracking/blob/master/INSTALL.md) and 
[pysot](https://github.com/STVIR/pysot/blob/master/INSTALL.md).)

### Requirements  
* Conda with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.
* PyTorch 1.2.0 and torchvision 0.4.0 (We use cuda10.0). Install from https://pytorch.org/get-started/previous-versions/#v120.
* matplotlib pandas opencv-python tensorboardX visdom cython pycocotools jpeg4py ninja-build libturbojpeg tqdm numba colorama

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name fcot python=3.7
conda activate fcot
```

#### Install PyTorch  
Install PyTorch-1.2.0 and torchvision-0.4.0 with cuda10.  
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

**Note:**    
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/#v120.  

#### Install matplotlib, pandas, opencv, visdom and tensorboadX  
```bash
conda install matplotlib pandas
pip install opencv-python tensorboardX visdom
```


#### Install the coco toolkit  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

#### Install others
```bash
pip install tqdm numba colorama
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py. 

#### Build DCNv2
FCOT using [DCNv2](Deformable Convolutional Networks V2) in the classification head and regression head.
```bash
cd ltr/external/DCNv2
./make.sh
```


#### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


#### Download the pre-trained networks  
You can download the pre-trained networks including fcot and dimp50 models from the [google drive](https://drive.google.com/drive/folders/1-TKOF4sKzUUb6C6XfM-rDjrBoEFhovEf?usp=sharing)
or [Baidu Drive](https://pan.baidu.com/s/1jABLxW2RaNr_p-tnT-aHuA) (with extraction code: `15kg`). The networks shoud be saved in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
"models".
You can also download the networks using the gdrive_download bash script as follows.

```bash
# Download the default network for FCOT
bash pytracking/utils/gdrive_download 1ZUsh0gE2I1ERRNVfcHiK1z_qfzwUkp1o models/fcot.pth

# Download the default network for DiMP50
bash pytracking/utils/gdrive_download 14zFM14cjJY-D_OFsLDlF1fX5XrSXGBQV models/dimp50.pth
```
