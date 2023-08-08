from datasets.dataset import HAM_datasets
from models.meta import Meta
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import timm
from tensorboardX import SummaryWriter

import os
import sys

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

def main(config):

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
