#!/bin/bash
#$ -M kborisia@nd.edu
#$ -m abe
#$ -pe smp 16
#$ -N MachineLearningFinal_KBor_Ncat

pip install --upgrade pip
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib

python train.py
