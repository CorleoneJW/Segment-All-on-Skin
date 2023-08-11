from datasets.dataset import HAM_datasets
from models.meta import Meta
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import timm
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import os
import sys
import shutil
from PIL import Image

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def preprocess_batch(batch):
    support_images = batch['support_images'].squeeze(0)
    support_masks = batch['support_masks'].squeeze(0)
    query_images = batch['query_images'].squeeze(0)
    query_masks = batch['query_masks'].squeeze(0)
    return support_images, support_masks, query_images, query_masks

# the function of copying the images
def copy_file_to_folder(source_file, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest_path = os.path.join(dest_folder, os.path.basename(source_file))
    shutil.copy(source_file, dest_path)

def main(config):

    print('#----------Generating data----------#')
    images_resources_path = "./HAM10000/origin/images/"         # the resource folder of images
    masks_resources_path = "./HAM10000/origin/masks/"           # the resource folder of masks
    ratio = 0.8     # the dataset and testset ratio
    categories = config.categories
    categories_dictionary = {}
    category_id = 1
    # prepare the csv for groundtruth
    origin_groundtruth_csv = "./HAM10000/origin/groundtruth/HAM10000_groundtruth.csv"   # read the csv file
    origin_groundtruth = pd.read_csv(origin_groundtruth_csv)    # read the csv file of groundtruth
    
    # generating the folders for each category in train folder and test folder
    # create folders for each categories
    trainset_images_path = "./HAM10000/train/images/"     # the images path for train dataset
    trainset_masks_path = "./HAM10000/train/masks/"     # the masks path for train dataset
    testset_images_path = "./HAM10000/test/images/"     # the images path for test dataset
    testset_masks_path = "./HAM10000/test/masks/"      # the masks path for test dataset

    for category in categories:
        # prepare the address for folders
        category_images_train_path = os.path.join(trainset_images_path,category)
        category_masks_train_path = os.path.join(trainset_masks_path,category)
        category_images_test_path = os.path.join(testset_images_path,category)
        category_masks_test_path = os.path.join(testset_masks_path,category)
        #delete the previously exsited folders
        shutil.rmtree(category_images_train_path)
        shutil.rmtree(category_masks_train_path)
        shutil.rmtree(category_images_test_path)
        shutil.rmtree(category_masks_test_path)
        # create corresponding folder for each categories
        os.makedirs(category_images_train_path, exist_ok=True)
        os.makedirs(category_masks_train_path, exist_ok=True)
        os.makedirs(category_images_test_path, exist_ok=True)
        os.makedirs(category_masks_test_path, exist_ok=True)

        # generate the data in trainset and testset for each categories
        dest_folder_images = "./HAM10000/train/images/"+category    # the destination train set folder of copying the images
        dest_folder_masks = "./HAM10000/train/masks/"+category    # the destination trian set folder of copying the masks
        dest_folder_images_change = "./HAM10000/test/images/"+category     # the destination folder of test set images
        dest_folder_masks_change = "./HAM10000/test/masks/"+category      # the destination folder of test set masks
        data_categories = origin_groundtruth[origin_groundtruth['dx'] == category]      # extract each categories 
        length_categories = len(data_categories)
        chaneg_folder_point = math.floor(length_categories * ratio)     # get the point to change directory name 
        elements_count = 0
        for image_name in data_categories['image_id']:      # each image_id in each categories
            if elements_count == chaneg_folder_point:
                dest_folder_images = dest_folder_images_change
                dest_folder_masks = dest_folder_masks_change
            images_file = image_name+".jpg"
            masks_file = image_name+"_segmentation.png"
            source_image = images_resources_path+images_file    # the full path of source of image : path + image file name
            source_mask = masks_resources_path+masks_file       # the full path of source of mask : path + mask file name
            copy_file_to_folder(source_image,dest_folder_images)
            # masks should be preprocess to the form of output for network (Width*Height*Category)
            image = Image.open(source_mask)
            image_array = np.array(image)
            image_array[image_array == 0] = category_id
            image_array[image_array == 255] = 0
            image = Image.fromarray(image_array)
            image.save(os.path.join(dest_folder_masks, masks_file))
            elements_count +=1

        categories_dictionary[category] = category_id       # add the category id in the categories_dictionary
        category_id += 1
    

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = HAM_datasets(config, train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.dataloader_bs, num_workers=config.num_workers)
    test_dataset = HAM_datasets(config, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.dataloader_bs, num_workers=config.num_workers)

    print('#----------Prepareing Model----------#')
    meta_model = Meta()
    meta_model = meta_model.to(device)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    meta_optimizer = get_optimizer(config, meta_model)
    meta_scheduler = get_scheduler(config, meta_optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    """
    annotation: the batch_size like 3000 is the number of sets, which equals to the previous whole dataset (3000 sets = 1 dataset) 
    """
    train_print = config.train_print        # the step that print the result every train_print times
    evaluation_point = config.evaluation_point      # the key point to evaluation the model
    print('#----------Training----------#')
    for epoch in range(start_epoch, (config.epoch_num//config.batch_size)+1):          # e.g. 9000 // 3000 = 3 epochs
        total_loss = 0.0
        step = 0        # according to the step, decide to print the result or do the evaluation
        for i,batch in enumerate(train_loader):
            step += 1       # the i-th step
            support_images, support_masks, query_images, query_masks = preprocess_batch(batch)
            support_images, support_masks, query_images, query_masks = support_images.to(device), support_masks.to(device), query_images.to(device), query_masks.to(device)
            meta_loss = meta_model(support_images,support_masks,query_images,query_masks)
            print(meta_loss)
            
            if step % train_print() == 0:
                print('step:',step,'\ttraining result ')

        # if total_loss < min_loss:
        #     min_loss = total_loss
        #     min_epoch = epoch
        #     torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_loss': min_loss,
        #         'min_epoch': min_epoch,
        #         'loss': total_loss,
        #         'model_state_dict': meta_model  .state_dict(),
        #         'optimizer_state_dict': meta_optimizer.state_dict(),
        #         'scheduler_state_dict': meta_scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'best.pth'))
            
        

if __name__ == '__main__':
    config = setting_config
    main(config)
