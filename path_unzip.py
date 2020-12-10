import os
import argparse
import sys

def main():
    os.system('mkdir -p store/analysis')
    os.system('mkdir -p store/analysis/type2_trace')
    os.system('mkdir -p store/analysis/type4_trace')
    os.system('mkdir -p store/analysis/unstructured_class_trace')
    os.system('mkdir -p store/analysis/type21112222')
    os.system('mkdir -p store/analysis/type211111111222222222')

    if os.path.exists('alexnet_imagenet_import_compact_BwCU.zip'):
        os.system('unzip alexnet_imagenet_import_compact_BwCU -d store/analysis/type2_trace')
    if os.path.exists('alexnet_imagenet_import_compact_BwAB.zip'):
        os.system('unzip alexnet_imagenet_import_compact_BwAB -d store/analysis/type4_trace')
    if os.path.exists('alexnet_imagenet_import_compact_FwAB.zip'):
        os.system('unzip alexnet_imagenet_import_compact_FwAB -d store/analysis/unstructured_class_trace')
    if os.path.exists('alexnet_imagenet_import_compact_hybrid.zip'):
        os.system('unzip alexnet_imagenet_import_compact_hybrid -d store/analysis/type21112222')
    if os.path.exists('resnet_18_cifar100_compact_BwCU.zip'):
        os.system('unzip resnet_18_cifar100_compact_BwCU store/analysis/type2_trace')
    if os.path.exists('resnet_18_cifar100_compact_BwAB.zip'):
        os.system('unzip resnet_18_cifar100_compact_BwAB store/analysis/type4_trace')
    if os.path.exists('resnet_18_cifar100_compact_FwAB.zip'):
        os.system('unzip resnet_18_cifar100_compact_FwAB store/analysis/unstructured_class_trace')
    if os.path.exists('resnet_18_cifar100_compact_hybrid.zip'):
        os.system('unzip resnet_18_cifar100_compact_hybrid store/analysis/type211111111222222222')
    if os.path.exists('resnet_50_imagenet_compact_BwCU.zip'):
        os.system('unzip resnet_50_imagenet_compact_BwCU store/analysis/type2_trace')
    if os.path.exists('resnet_50_imagenet_compact_BwAB.zip'):
        os.system('unzip resnet_50_imagenet_compact_BwAB store/analysis/type4_trace')



if __name__ == '__main__':
    main()