import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy
import sys
from .basenet import SimpleNet
from .basenet import UNet
sys.path.append('../')
from configs.config_setting import setting_config
class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self):
        super(Meta, self).__init__()
        config = setting_config 
        self.base_net = UNet(config.in_channels,config.num_classes)
        self.inner_lr = config.inner_lr
        self.outer_lr = config.outer_lr
        self.inner_steps = config.inner_steps
        self.meta_optimizer = optim.Adam(self.base_net.parameters(), lr=self.outer_lr)
    
    def forward(self, support_images, query_images, support_masks, query_masks):
        updated_params = self.inner_loop(support_images,support_masks)
        loss = self.compute_loss(updated_params, query_images,query_masks)
        return loss

    def inner_loop(self, support_images, support_masks):
        optimizer = optim.SGD(self.base_net.parameters(), lr=self.inner_lr)
        
        for step in range(self.inner_steps):
            optimizer.zero_grad()
            loss = self.compute_loss(self.base_net, support_images, support_masks)
            loss.backward()
            optimizer.step()
        
        updated_params = self.base_net.state_dict().copy()
        return updated_params

    def compute_loss(self, model, images, masks):
        # 根据数据计算损失，这里需要根据具体任务来实现
        """
        images: "support_images:" [tensor([[[]]])，tensor([[[]]])] length: n_way * k_shots(k_query)
        write iteration for data
        """
        criterion = nn.CrossEntropyLoss()
        # calculate the loss of each images according to the masks
        print(images.shape)
        print(masks.shape)
        predict = model(images)
        temp_loss = criterion(predict,masks)

        return temp_loss

    if __name__ == "__main__":
        pass