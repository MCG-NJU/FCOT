#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name python=3.7

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Installing tqdm numba colorama for evaluation ******************"
pip install tqdm numba colorama

echo ""
echo ""
echo "****************** Build DCNv2 ******************"
cd ltr/external/DCNv2
./make.sh
cd ../../..

echo ""
echo ""
echo "****************** Downloading pretrained models: FCOT and DiMP50 ******************"
if [[ ! -d "models" ]]; then
    mkdir models
fi

echo ""
echo ""
echo "****************** Downloading pretrained DiMP50 and FCOT models ******************"
# dimp50: https://drive.google.com/file/d/14zFM14cjJY-D_OFsLDlF1fX5XrSXGBQV/view?usp=sharing
# fcot: https://drive.google.com/file/d/1ZUsh0gE2I1ERRNVfcHiK1z_qfzwUkp1o/view?usp=sharing
bash pytracking/utils/gdrive_download 14zFM14cjJY-D_OFsLDlF1fX5XrSXGBQV models/dimp50.pth
bash pytracking/utils/gdrive_download 1ZUsh0gE2I1ERRNVfcHiK1z_qfzwUkp1o models/fcot.pth

echo ""
echo ""
echo "****************** Build region ******************"
cd pytracking/utils/vot_utils/
python setup.py build_ext --inplace
cd ../../..

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done

echo ""
echo ""
echo "****************** Installation complete! ******************"