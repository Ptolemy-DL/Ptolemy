import os
import pdb

def main():
    #pre-process cifar10/100 dataset
    os.system('mkdir cifar10-raw')
    os.system('tar -xvf cifar-10-python.tar.gz -C cifar10-raw/')
    os.system('mv cifar10-raw/cifar-10-batches-py/* cifar10-raw/')
    os.system('rm -r cifar10-raw/cifar-10-batches-py')
    os.system('mkdir cifar100-raw')
    os.system('tar -xvf cifar-100-python.tar.gz -C cifar100-raw/')
    os.system('mv cifar100-raw/cifar-100-python/* cifar100-raw/')
    os.system('rm -r cifar100-raw/cifar-100-python')


if __name__ == '__main__':
    main()