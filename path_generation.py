import os
import argparse
import sys
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="Alexnet",
        help="different networks, pick between Alexnet, Resnet-18 and Vgg16",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Imagenet",
        help="different datasets, pick between Imagenet, Cifar-10 and Cifar-100"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="EP",
        help="different types of path extraction, default EP, pick between BwCU, BwAB and FwAB",
    )
    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.5,
        help="cumulative threshold theta, default 0.5",
    )
    parser.add_argument(
        "--absolute_threshold",
        type=float,
        default=0,
        help="absolute threshold phi, default 0",
    )
    params, unparsed = parser.parse_known_args()    

    if params.network == 'Alexnet':
        if params.dataset == 'Imagenet':
            cmd = 'python src/nninst/backend/tensorflow/trace/alexnet_imagenet_class_trace.py --type='+ params.type +' --cumulative_threshold='+str(params.cumulative_threshold) +' --absolute_threshold='+str(params.absolute_threshold)
            #pdb.set_trace()
            os.system(cmd) 
        else:
            print('Network Dataset combination is not supported yet')
    elif params.network == 'Resnet-18':
        if params.dataset == 'Cifar-10':
            cmd = 'python src/nninst/backend/tensorflow/trace/resnet_18_cifar10_class_trace_v2.py --type='+ params.type +' --cumulative_threshold='+str(params.cumulative_threshold)+' --absolute_threshold='+str(params.absolute_threshold)
            #pdb.set_trace()
            os.system(cmd)
        elif params.dataset == 'Cifar-100':
            cmd = 'python src/nninst/backend/tensorflow/trace/resnet_18_cifar100_class_trace.py --type='+ params.type +' --cumulative_threshold='+str(params.cumulative_threshold)+' --absolute_threshold='+str(params.absolute_threshold)
            #pdb.set_trace()
            os.system(cmd)
    elif params.network == "Vgg16":
        if params.dataset == 'Imagenet':
            cmd = 'python src/nninst/backend/tensorflow/trace/vgg_16_imagenet_class_trace.py --type='+ params.type +' --cumulative_threshold='+str(params.cumulative_threshold)+' --absolute_threshold='+str(params.absolute_threshold)
            pdb.set_trace()
            os.system(cmd)
        else:
            print('Network Dataset combination is not supported yet')
    else:
        print('Network Dataset combination is not supported yet')

    
if __name__ == '__main__':
    main()