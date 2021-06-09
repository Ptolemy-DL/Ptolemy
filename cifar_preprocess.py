import os
import pdb

def main():
    #pre-process cifar10/100 dataset
    os.system('mkdir cifar10-raw')
    os.system('tar -xvf cifar-10-binary.tar.gz -C cifar10-raw/')
    os.system('mv cifar10-raw/cifar-10-batches-bin/* cifar10-raw/')
    os.system('rm -r cifar10-raw/cifar-10-batches-bin')
    os.system('mkdir cifar100-raw')
    os.system('tar -xvf cifar-100-binary.tar.gz -C cifar100-raw/')
    os.system('mv cifar100-raw/cifar-100-binary/* cifar100-raw/')
    os.system('rm -r cifar100-raw/cifar-100-binary')


if __name__ == '__main__':
    main()