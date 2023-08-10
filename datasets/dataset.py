import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import shutil
sys.path.append('../')
from configs.config_setting import setting_config

class HAM_datasets(Dataset):
    def __init__(self, config, train=True):
        super(HAM_datasets, self)
        self.categories = config.categories  # the list of categories
        self.clsnum = len(self.categories)  # the length of categories
        self.batch_size = config.batch_size  # batch of set, not batch of imgs
        self.n_way = config.n_way  # n-way
        self.k_shot = config.k_shot  # k-shot
        self.k_query = config.k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = (
            self.n_way * self.k_query
        )  # number of samples per set for evaluation
        self.resize_h = config.resize_h  # resize height
        self.resize_w = config.resize_w     # resize height
        self.startidx = config.startidx  # index label not from 0, but from startidx
        self.train = train  # the mode like train and test
        self.batch_list = []

        print(
            "Train mode :%s, batch_size:%d, %d-way, %d-shot, %d-query, %d-resizeh, %d-resizew"
            % (
                self.train,
                self.batch_size,
                self.n_way,
                self.k_shot,
                self.k_query,
                self.resize_h,
                self.resize_w
            )
        )

        if self.train == True:
            self.transform = config.train_transformer
        else:
            self.transform = config.test_transformer

        self.mask_transform = config.mask_transformer

        """
        path of the dataset (images and masks)
        """
        if train == True:
            self.image_path = os.path.join(
                config.train_set, "images")  # path of train set images
            self.mask_path = os.path.join(
                config.train_set, "masks")  # path of train set mask
        else:
            self.image_path = os.path.join(
                config.test_set, "images")  # path of test set images
            self.mask_path = os.path.join(
                config.test_set, "masks")  # path of test set mask
        
        self.class_to_samples = {cls: self.load_samples(cls) for cls in self.categories}
        
        # generating the temp directory for the train or test
        # self.create_tempfolder(self.image_path,self.mask_path,self.categories)
        
        # generate the batchlist for __getitem__
        self.create_batchs()


    def __getitem__(self, idx):
        batch_list = self.batch_list
        return batch_list[idx]

    def __len__(self):
        return self.batch_size

    def load_samples(self, cls):
        images_dir = os.path.join(self.image_path, cls)
        masks_dir = os.path.join(self.mask_path, cls)
        image_files = os.listdir(images_dir)
        samples = [
            {
                'image': os.path.join(images_dir, img),
                'mask': os.path.join(masks_dir, img[:-4]+"_segmentation.png")
            }
            for img in image_files
        ]
        return samples

    def get_task(self):
        sampled_classes = random.sample(self.categories, self.n_way)    # No duplicate naturally
        
        support_set = []
        query_set = []

        for cls in sampled_classes:
            sampled_samples = random.sample(self.class_to_samples[cls], self.k_shot + self.k_query)
            support_samples = sampled_samples[:self.k_shot]
            query_samples = sampled_samples[self.k_shot:]
            
            support_set.extend(support_samples)
            query_set.extend(query_samples)
        
        return support_set, query_set       # [{image:xxx1.jpg, mask:xxx1.png},{image:xxx2.jpg, mask:xxx2.png}]

    def create_batchs(self):
        batch_size = self.batch_size
        for time in range(batch_size):
            support_set, query_set = self.get_task()        # [{image:xxx1.jpg, mask:xxx1.png},{image:xxx2.jpg, mask:xxx2.png}]

            for i,sample in enumerate(support_set):
                if i == 0:
                    support_images = self.transform(Image.open(sample['image']).convert('RGB')).unsqueeze(0)
                    support_masks = self.mask_transform(Image.open(sample['mask']).convert('L')).unsqueeze(0)
                else:
                    support_image = self.transform(Image.open(sample['image']).convert('RGB')).unsqueeze(0)
                    support_mask = self.mask_transform(Image.open(sample['mask']).convert('L')).unsqueeze(0)
                    support_images = torch.cat((support_images,support_image),dim=0)
                    support_masks = torch.cat((support_masks,support_mask),dim=0)

            for i,sample in enumerate(query_set):
                if i == 0:
                    query_images = self.transform(Image.open(sample['image']).convert('RGB')).unsqueeze(0)
                    query_masks = self.mask_transform(Image.open(sample['mask']).convert('L')).unsqueeze(0)
                else:
                    query_image = self.transform(Image.open(sample['image']).convert('RGB')).unsqueeze(0)
                    query_mask = self.mask_transform(Image.open(sample['mask']).convert('L')).unsqueeze(0)
                    query_images = torch.cat((query_images,query_image),dim=0)
                    query_masks = torch.cat((query_masks,query_mask),dim=0)

            generation_result = {
                'support_images': support_images, 'support_masks': support_masks,
                'query_images': query_images, 'query_masks': query_masks
            }                           #{'support_images':[images_tensor1,images_tensor2],'support_masks':[masks_tensor1,masks_tensor2]}
            
            self.batch_list.append(generation_result)

    def create_tempfolder(self, image_path, mask_path, categories):
        # delete the previous folder
        delete_folder_image = os.path.join(image_path, "temp")
        delete_folder_mask = os.path.join(mask_path, "temp")
        shutil.rmtree(delete_folder_image)
        shutil.rmtree(delete_folder_mask)
        # according to the categories, generating new temp folder
        try:
            destination_image = delete_folder_image
            destination_mask = delete_folder_mask
            os.makedirs(destination_image, exist_ok=True)
            os.makedirs(destination_mask, exist_ok=True)
            for category in categories:
                source_image = os.path.join(image_path, category)
                source_mask = os.path.join(mask_path, category)
                # copy the images
                for file in os.listdir(source_image):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        source_path = os.path.join(source_image, file)
                        destination_path = os.path.join(
                            destination_image, file)
                        shutil.copy(source_path, destination_path)
                # copy the masks
                for file in os.listdir(source_mask):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        source_path = os.path.join(source_mask, file)
                        destination_path = os.path.join(destination_mask, file)
                        shutil.copy(source_path, destination_path)
        except Exception as e:
            print(f"Error occurred while copying：{str(e)}")


"""
data sturcture:
variable:
dataset: len(): config(batch_size)
dataloader: len: len(dataset)/param(batch_size)  e.g. 64/8 = 8
train_loader: iteration item in train_loader: dictionary: {"support_images":..., "support_masks":..., "query_images":..., "query_masks":...,}  
"support_images:" [tensor([[[]]])，tensor([[[]]])] length: n_way * k_shots(k_query)
"""