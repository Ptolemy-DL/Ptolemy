# Ptolemy: Detecting Deep Learning Adversarial Samples at Inference Time

This repo contains the artifact of [Ptolemy: Architectural Support for Robust Deep Learning](https://www.cs.rochester.edu/horizon/pubs/micro20-ptolemy.pdf), which is mechanism to detect adversarial samples at inference time.

The detection mechanism is evaluated on different networks on different datasets under common attacks. The repo also contains code to generate adaptive attacks that are specifically designed to "defeat" our defense mechanisms.

## Working environment
We have tested our code on a system with Red Hat 4.8.5-39; the machine we run this code is Intel(R) Xeon(R) Silver 4110 with 96115 MB memory in total. The machine has two NVIDIA GeForce 2080Ti GPU with CUDA version 9.0.176. 

We are using Ray 0.7.2 (https://github.com/ray-project/ray) for the distributed computing. If the code accidentaly shuts down half-way, it may be the reason of lacking resource. We recommond set the number of CPUs used in `ray.init()` such as `ray.init(num_cpus=8)` in `src/nninst/backend/tensorflow/trace/*_class_trace.py`.  

## Install Ptolemy

```bash
cd <path-to-project>
conda env create -f environment.yml
source activate nninst
python setup.py develop
```

Notes: Ptolemy needs the MKL support to run on TensorFlow.
Ptolemy uses TensorFlow-MKL by default.
You can build TensorFlow with both MKL and CUDA support from the source by yourself.
The CUDA support can vastly speed up the generation of the adversarial examples.

## Download pretrained weights

All the pre-trained weights can be downloaded [here](https://drive.google.com/drive/folders/1g-Lq50TBrxQiuH6dM-ZHwfT9akqT185g?usp=sharing). Once you download the pre-trained weightsm use the script below to unzip them into proper directories for subsequent processing.

run `python unzip_weights.py`

## Pre-process datasets

### ImageNet
We assume that the Imagenet raw data (i.e., `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`) has been downloaded into the current directory. 

run `python imagenet_preprocess.py`

### CIFAR-10 and CIFAR-100

We assume that the CIFAR10/100 raw data (i.e.,`cifar-10-python.tar.gz` and `cifar-100-python.tar.gz`) has been downloaded into the current directory. 

run `python cifar_preprocess.py`

### Set datasets' paths (optional)

If the Imagenet and CIFAR10/100 raw data has been downloaded into the current directory, you can safely skip this step.

1. Set `IMAGENET_RAW_DIR` in `src/nninst/dataset/envs.py` to ImageNet's path. The default path, if you have followed the instructions above, would be `imagenet-raw-data/` in the current directory.
2. Set `CIFAR10_TRAIN`,`CIFAR10_TEST`,`CIFAR100_TRAIN`,`CIFAR100_TEST` in `src/nninst/backend/tensorflow/dataset/config.py` to change to your specific directory. The default directory would be `cifar10-raw/` and `cifar100-raw/` if you have followed the instructions above.

## Generate network graphs

run `python nninst_preprocess.py`

## Generate per-class activation paths

Run `python path_generation.py --network=Alexnet --dataset=Imagenet --type=BwCU --theta==0.5 --alpha=None`

You can choose different networks `--network`, datasets `--dataset` and detection types `--type`, as well as the algorithm-specific parameters `--theta, --alpha`.

Path generation usually takes a long time. We provide the per-class paths used in our paper [here](https://drive.google.com/drive/folders/1OhqLkEDvn4X2CLBSpKtQdMRU4XxaFnB6?usp=sharing). After downloading the paths, run `python path_unzip.py`, which will unzip them to the proper directories in preparation for subsequent processing.

## Generate adversarial examples

```bash
python -m nninst.backend.tensorflow.attack.generate_adversarial_examples
```
This by default will generate all the attacks we used in the paper, including FGSM, DeepFool, JSMA, BIM, CWL2, and adaptive attacks from layer 8 to layer 1 for AlexNet. For more details of how the adaptive attacks are generated, check Section 7.4 of the paper. To modify the specific types of attacks you want to use, please modify line 61-128 in `src/nninst/backend/tensorflow/attack/generate_adversarial_examples.py`

The hyper-parameters used to generate the attacks can be found [here](https://github.com/Ptolemy-sw/Ptolemy/blob/master/src/nninst/backend/tensorflow/attack/hyperparameters.md).

## Generate activation paths for inputs in the dataset

```bash
python adversarial_example_path_generation.py --network=alexnet --dataset=imagenet --type=BwCU --theta==0.5 --alpha=None
```
This will by default generate activation paths for all the adversarial attacks generated above. The parameters used should be the same with class activation paths.

## Calculate and plot AUCs

```bash
MPLBACKEND=Agg python -m nninst.plot.plot_auc
```

## Publication
```bash
@inproceedings{gan2020ptolemy,
  title={Ptolemy: Architecture support for robust deep learning},
  author={Gan, Yiming and Qiu, Yuxian and Leng, Jingwen and Guo, Minyi and Zhu, Yuhao},
  booktitle={2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages={241--255},
  year={2020},
  organization={IEEE}
}
```

## Pytorch Implementation

We currently do not support Pytorch. For users who need to port Ptolemy into Pytorch, we provides simple tutorial here.
1. Change Pytorch model into nninst.Graph. (https://github.com/Ptolemy-DL/Ptolemy/blob/master/src/nninst/graph.py)
2. Reuse the code to extract path; If new operations is needed, please add the definition of operations (https://github.com/Ptolemy-DL/Ptolemy/blob/master/src/nninst/op.py#L26) and defination of extraction function (https://github.com/Ptolemy-DL/Ptolemy/blob/master/src/nninst/trace.py#L1336) manually.
