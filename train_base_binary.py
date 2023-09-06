#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets.dataset import *
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
from models.basenet import *
from utils import *
from configs.config_setting import setting_config
from copy import deepcopy
import sklearn.metrics as metrics
from torch.cuda.amp import autocast, GradScaler
import torch.nn.init as init
import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings("ignore")

config = setting_config


# In[2]:


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

def evaluation_api(predicted_list,groudtruth_list):
    pre = np.array([item for sublist in predicted_list for item in sublist]).reshape(-1)
    gts = np.array([item for sublist in groudtruth_list for item in sublist]).reshape(-1)
    # confusion_matrix = metrics.confusion_matrix(gts,pre)
    # TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
    dice = metrics.f1_score(gts,pre)
    return dice

def evaluation_epoch(predicted_list,groundtruth_list):
    TP = [0]*config.num_classes
    FP = [0]*config.num_classes
    FN = [0]*config.num_classes
    dice = [0.0]*config.num_classes
    
    for i in range(len(predicted_list)):
        preds = np.array(predicted_list[i]).reshape(-1)
        gts = np.array(groundtruth_list[i]).reshape(-1)
        for j in range(len(preds)):
            if preds[j] == gts[j]:
                TP[gts[j]] += 1
            else:
                FP[preds[j]] += 1
                FN[gts[j]] += 1        
    
    for i in range(config.num_classes):
        dice[i] = (2 * TP[i])/(FP[i]+FN[i]+2*TP[i]+1)

    mdice = (2*np.sum(TP))/(np.sum(FP)+np.sum(FN)+2*np.sum(TP)+1)    
    return dice,mdice

def evaluation_basenet(base_net,query_images,query_masks,criterion):
    predicted = base_net(query_images)
    loss = criterion(predicted,query_masks)
    predicted = torch.argmax(predicted,dim=1).long()
    predict_numpy = predicted.detach().cpu().numpy().reshape(-1)
    masks_numpy = query_masks.long().detach().cpu().numpy().reshape(-1)
    accuracy = metrics.accuracy_score(masks_numpy,predict_numpy)
    f1_score = metrics.f1_score(masks_numpy,predict_numpy,average=None)
    return accuracy,f1_score,loss

def initialize_weights_he(model):
    for param in model.parameters():
        init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

def initialize_weights_xavier(model):
    for param in model.parameters():
        init.xavier_uniform_(param)

def initialize_weights_normal(model):
    for param in model.parameters():
        init.normal_(param, mean=0, std=1)        


# In[3]:


print('#----------Creating logger----------#')
sys.path.append(config.work_dir + '/')
log_dir = os.path.join(config.work_dir, 'log')
checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
resume_model = os.path.join(checkpoint_dir, 'latest.pth')
outputs = os.path.join(config.work_dir, 'outputs')
csv_save = os.path.join(config.work_dir, 'csv')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(outputs):
    os.makedirs(outputs)
if not os.path.exists(csv_save):
    os.makedirs(csv_save)

global logger
logger = get_logger('test', log_dir)
global writer
writer = SummaryWriter(config.work_dir + 'summary')

log_config_info(config, logger)


# In[4]:


print('#----------Generating data----------#')
images_resources_path = "./data/HAM10000/origin/images/"         # the resource folder of images
masks_resources_path = "./data/HAM10000/origin/masks/"           # the resource folder of masks
ratio = 0.8     # the dataset and testset ratio
categories = config.categories
categories_dictionary = {}
category_id = 1
# prepare the csv for groundtruth
origin_groundtruth_csv = "./data/HAM10000/origin/groundtruth/HAM10000_groundtruth.csv"   # read the csv file
origin_groundtruth = pd.read_csv(origin_groundtruth_csv)    # read the csv file of groundtruth

# generating the folders for each category in train folder and test folder
# create folders for each categories
trainset_images_path = "./data/HAM10000/train/images/"     # the images path for train dataset
trainset_masks_path = "./data/HAM10000/train/masks/"     # the masks path for train dataset
testset_images_path = "./data/HAM10000/test/images/"     # the images path for test dataset
testset_masks_path = "./data/HAM10000/test/masks/"      # the masks path for test dataset

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
    dest_folder_images = "./data/HAM10000/train/images/"+category    # the destination train set folder of copying the images
    dest_folder_masks = "./data/HAM10000/train/masks/"+category    # the destination trian set folder of copying the masks
    dest_folder_images_change = "./data/HAM10000/test/images/"+category     # the destination folder of test set images
    dest_folder_masks_change = "./data/HAM10000/test/masks/"+category      # the destination folder of test set masks
    data_categories = origin_groundtruth[origin_groundtruth['dx'] == category]      # extract each categories 
    data_categories = data_categories.sample(frac=1,random_state=config.seed)       # random sample the datagenerating
    length_categories = len(data_categories)
    change_folder_point = math.floor(length_categories * ratio)     # get the point to change directory name 
    elements_count = 0
    for image_name in data_categories['image_id']:      # each image_id in each categories
        if elements_count == change_folder_point:
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
        image_array[image_array == 255] = 1
        image = Image.fromarray(image_array)
        image.save(os.path.join(dest_folder_masks, masks_file))
        elements_count +=1
    categories_dictionary[category] = category_id       # add the category id in the categories_dictionary
    category_id += 1


# In[5]:


print('#----------GPU init----------#')
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
set_seed(config.seed)
device = torch.device('cuda')
torch.cuda.empty_cache()


# In[6]:


print('#----------Prepareing Datasets----------#')
# create the dataset and dataloader
batch_size = config.batch_size
train_dataset = HAMALL_datasets(config, train=True)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=config.num_workers)
test_dataset = HAMALL_datasets(config, train=False)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=config.num_workers)
print("trian_dataset length:",len(train_dataset))
print("test_dataset length:",len(test_dataset))


# In[7]:


print('#----------Prepareing Model----------#')
in_channels = config.in_channels
out_channels = config.out_channels
# base_net = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=config.num_classes, activation=None, aux_params=None)
base_net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=config.out_channels, activation=None, aux_params=None)
# initialize_weights_he(base_net)
base_net = base_net.to(device)


# In[8]:


print('#----------Prepareing loss, opt, sch and amp----------#')
criterion = nn.BCEWithLogitsLoss()
meta_optimizer = get_optimizer(config, base_net)
meta_scheduler = get_scheduler(config, meta_optimizer)
criterion = criterion.to(device)


# In[9]:


print('#----------Set other params----------#')
min_loss = 999
start_epoch = 1
min_epoch = 1
threshold = 0.5


# In[10]:


print('#----------Start training----------#')
torch.cuda.empty_cache()
info = "%d-resizeh, %d-resizew, %f-outer_lr"%(config.resize_h,config.resize_w,config.outer_lr)
print(info)
logger.info(info)
best_dice = 0.0
train_csv = os.path.join(csv_save,"train.csv")
test_csv = os.path.join(csv_save,"test.csv")
train_columns = ['Epoch','Loss',"Mdice"]
train_df = pd.DataFrame(columns=train_columns)
test_columns = ['Epoch','Mdice']
test_df = pd.DataFrame(columns=test_columns)
for epoch in range(start_epoch, config.epoch_num+1):
    # train part
    torch.cuda.empty_cache()
    predicted_list = []
    groundtruth_list = []
    loss_list = []    
    base_net.train()
    for image,mask in train_loader:
        # claer the meta_optimizer, setting zero
        meta_optimizer.zero_grad()
        image = image.to(device)
        mask = mask.to(device)
        image = torch.squeeze(image,dim=1)      # torch.Size([bs, 3, 512, 512])
        mask = torch.squeeze(mask,dim=1)     # torch.Size([bs, 1, 512, 512])
        mask = torch.squeeze(mask,dim=1).float()     # torch.Size([bs, 512, 512])
        predicted = base_net(image)     # torch.Size([bs,out_channels=1,512,512])
        predicted = predicted.squeeze(1)    # torch.Size([bs,512,512])
        loss = criterion(predicted,mask)
        loss.backward()
        meta_optimizer.step()
        loss_list.append(loss.cpu().detach().numpy())
        predicted = (predicted > threshold).long()
        temp_predicted = predicted.cpu().detach().numpy()       # threshold alternative
        predicted_list.append(temp_predicted)
        groundtruth_list.append(mask.long().cpu().detach().numpy())
    # train_dice,train_mdice = evaluation_epoch(predicted_list,groundtruth_list)
    train_dice = evaluation_api(predicted_list,groundtruth_list)
    train_mloss = np.mean(loss_list)
    log_train = f'epoch: {epoch}, loss: {train_mloss}, dice: {train_dice}'
    print("#Train# ",log_train)
    temp_result = pd.Series([epoch,train_mloss,train_dice],index=train_columns)
    train_df = train_df.append(temp_result, ignore_index=True)
    train_df.to_csv(train_csv, index=False)
    
    # test part
    torch.cuda.empty_cache()
    predicted_list = []
    groundtruth_list = []
    base_net.eval()
    with torch.no_grad():
        for image,mask in test_loader:
            # claer the meta_optimizer, setting zero
            meta_optimizer.zero_grad()
            image = image.to(device)
            mask = mask.to(device)
            image = torch.squeeze(image,dim=1)      # torch.Size([bs, 3, 512, 512])
            mask = torch.squeeze(mask,dim=1)     # torch.Size([bs, 1, 512, 512])
            mask = torch.squeeze(mask,dim=1).float()     # torch.Size([bs, 512, 512])
            predicted = base_net(image)
            temp_predicted = (predicted > threshold).long().cpu().detach().numpy()        # (20, 128, 128)
            predicted_list.append(temp_predicted)
            groundtruth_list.append(mask.long().cpu().detach().numpy())
        # test_dice,test_mdice = evaluation_epoch(predicted_list,groundtruth_list)
        test_dice = evaluation_api(predicted_list,groundtruth_list)
        log_test = f'epoch: {epoch}, dice: {test_dice}'
        print("#Test# ",log_test)
        temp_result = pd.Series([epoch,test_dice],index=test_columns)
        test_df = test_df.append(temp_result, ignore_index=True)
        test_df.to_csv(test_csv, index=False)
        logger.info(log_test)

    if test_dice > best_dice:
        torch.save(base_net.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
        best_dice = test_dice
    torch.cuda.empty_cache()



# In[11]:


print("Best dice in testset:",best_dice)

