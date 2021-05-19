#!/bin/bash
#$ -M kborisia@nd.edu
#$ -m abe

#$ -q gpu
#$ -pe smp 8
#$ -l gpu_card=1
#$ -N MachineLearningFinal_KBor_Ncat

pip install --upgrade pip
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib

python train-grayscale.py
