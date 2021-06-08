import os
import argparse
import sys
import pdb

def main():
    if os.path.exists('alexnet-imagenet.gz'):
        os.system('mkdir -p tf/alexnet')
        os.system('tar zxvf alexnet-imagenet.gz -C tf/alexnet/')
    if os.path.exists('resnet-18-cifar10.zip'):
        os.system('mkdir -p tf/resnet-18-cifar10')
        os.system('unzip -o resnet-18-cifar10.zip -d tf/resnet-18-cifar10')
        os.system('mv tf/resnet-18-cifar10/model tf/resnet-18-cifar10/model_train')
    if os.path.exists('resnet-18-cifar100.zip'):
        os.system('mkdir -p tf/resnet-18-cifar100')
        os.system('unzip -o resnet-18-cifar100.zip -d tf/resnet-18-cifar100')
        os.system('mv tf/resnet-18-cifar100/resnet-18-cifar100 tf/resnet-18-cifar100/model_train')
    if os.path.exists('resnet-50-imagenet.gz'):
        os.system('mkdir -p tf/resnet-50-v2/model')
        os.system('tar zxvf resnet-50-imagenet.gz -C tf/resnet-50-v2/model')
    if os.path.exists('vgg-16-imagenet.gz'):
        os.system('mkdir -p tf/vgg_16/model')
        os.system('tar zxvf vgg-16-imagenet.gz -C tf/vgg_16/model')

if __name__ == '__main__':
    main()