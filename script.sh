#!/bin/bash

export CUDA_VISIBLE_DEVICES="2" 
python training.py --epochs=10 --batch_size=16 --logdir='test' --lr=0.001 --dataset='cifar10' --site_number=1 --model_name='resnet18emb' --optimizer_type='sgd' --scheduler_mode='cosine' --T_max=500 --pretrained=True --aug_mode='classification' --partition='regular' 'testcomment'