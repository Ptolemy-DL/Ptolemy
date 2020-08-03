import os
import pdb

def main():
	#generate graph for network
    os.system('python -m nninst.backend.tensorflow.model.alexnet')#alexnet
    os.system('python -m nninst.backend.tensorflow.model.resnet_18_cifar100')#resnet_18
    # to generate other network's graph, check /src/backend/tensorflow/model, we provide other models

    #generate per-layer metrics
    os.system('python -m nninst.backend.tensorflow.attack.generate_example_traces')
    os.system('python -m nninst.backend.tensorflow.attack.calc_per_layer_metrics')

if __name__ =='__main__':
    main()	