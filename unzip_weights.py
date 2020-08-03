import os
import argparse
import sys
import pdb

def main():
    if os.path.exists('alexnet_imagenet.tar.gz'):
        os.system('mkdir tf/alexnet')
        os.system('tar zxvf alexnet_imagenet.tar.gz -C tf/alexnet/')
    if os.path.exists('resnet-18-cifar10.zip'):
        os.system('mkdir tf/resnet-18-cifar10')
        os.system('unzip resnet-18-cifar10.zip -d tf/resnet-18-cifar10')
    if os.path.exists('resnet-18-cifar100.zip'):
        os.system('mkdir tf/resnet-18-cifar100')
        os.system('unzip resnet-18-cifar100.zip -d tf/resnet-18-cifar10')
    if os.path.exists('resnet-18-cifar50.zip'):
        os.system('mkdir tf/resnet-50-imagenet')
        os.system('unzip resnet-50-imagenet.zip -d tf/resnet-50-imagenet')
    if os.path.exists('vgg-16-imagenet.zip'):
        os.system('mkdir tf/vgg-16-imagenet')
        os.system('unzip vgg-16-imagenet.zip -d tf/vgg-16-imagenet')

if __name__ == '__main__':
    main()