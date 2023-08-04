import os
import sys
import torch
from torch.utils.data import Dataset
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
        self.resize = config.resize  # resize to
        self.startidx = config.startidx  # index label not from 0, but from startidx
        self.train = train  # the mode like train and test

        print(
            "shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, "
            % (
                self.train,
                self.batch_size,
                self.n_way,
                self.k_shot,
                self.k_query,
            ),"pixel size(resize)",self.resize
        )

        if self.train == True:
            self.transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize(self.resize),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

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

        """
        generating the temp directory for the train or test
        """
        # self.create_tempfolder(self.image_path,self.mask_path,self.categories)

    def __getitem__(self, index):
        
        return 

    def __len__(self):
        return self.batch_size

    def load_images(self, class_dir):
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        return images

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

if __name__ == "__main__":
    config = setting_config
    # ham = HAM_datasets(config, train=True)
    sampled = random.sample(["1","2","3"],2)
    print(sampled)
    