import os
import pdb

def main():
    os.system('mkdir -p imagenet-raw-data') #create directory for imagenet raw data

    os.system('mkdir -p imagenet-raw-data/train')#create sub directory for training and validation
    os.system('mkdir -p imagenet-raw-data/validation')

    os.system('tar -xf ILSVRC2012_img_val.tar -C imagenet-raw-data/validation/')#unzip downloading file
    os.system('tar -xvf ILSVRC2012_img_train.tar -C imagenet-raw-data/train/')
    os.system('mv imagenet_2012_bounding_boxes.csv imagenet-raw-data/')

    os.system('cd imagenet-raw-data/train')
    os.system('find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done')
    os.system('cd .. & cd ..')

    #imagenet preprocessing
    os.system('python src/slim/datasets/preprocess_imagenet_validation_data.py imagenet-raw-data/validation/ src/slim/datasets/imagenet_2012_validation_synset_labels.txt')
    os.system('python src/slim/datasets/build_imagenet_data.py --train_directory=imagenet-raw-data/train --validation_directory=imagenet-raw-data/validation --imagenet_metadata_file=src/slim/datasets/imagenet_metadata.txt --labels_file=src/slim/datasets/imagenet_lsvrc_2015_synsets.txt --bounding_box_file=imagenet-raw-data/imagenet_2012_bounding_boxes.csv')
    
    #if you are using python2, change line 574 in build_imagenet_data.py. 
    #change shuffled_index = list(range(len(filenames))) into shuffled_index = range(len(filenames))


    
if __name__ == '__main__':
    main()