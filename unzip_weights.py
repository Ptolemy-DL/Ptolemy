import os
import argparse
import sys
import pdb

def main():
    if os.path.exists('alexnet_imagenet.gz'):
        os.system('mkdir tf/alexnet')
        os.system('tar zxvf alexnet-imagenet.gz -C tf/alexnet/')
    if os.path.exists('resnet-18-cifar10.zip'):
        os.system('mkdir tf/resnet-18-cifar10')
        os.system('unzip resnet-18-cifar10.zip -d tf/resnet-18-cifar10')
    if os.path.exists('resnet-18-cifar100.zip'):
        os.system('mkdir tf/resnet-18-cifar100')
        os.system('unzip resnet-18-cifar100.zip -d tf/resnet-18-cifar100')
    if os.path.exists('resnet-50-imagenet.gz'):
        os.system('mkdir tf/resnet-50-imagenet')
        os.system('tar zxvf resnet-50-imagenet.gz -C tf/resnet-50-imagenet/')
    if os.path.exists('vgg-16-imagenet.gz'):
        os.system('mkdir tf/vgg-16-imagenet')
        os.system('tar zxvf vgg-16-imagenet.gz -C tf/vgg-16-imagenet')

if __name__ == '__main__':
    main()