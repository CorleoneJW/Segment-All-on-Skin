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
from .basenet import UnetPlusPlus
sys.path.append('../')
from configs.config_setting import setting_config
import segmentation_models_pytorch as smp
from copy import deepcopy
import sklearn.metrics as metrics


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self):
        super(Meta, self).__init__()
        config = setting_config
        self.base_net = UNet(config.in_channels,config.num_classes)
        # self.base_net = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights=None, decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=3, classes=1, activation=None, aux_params=None)
        # self.base_net = UnetPlusPlus(config.in_channels,config.num_classes)
        self.inner_lr = config.inner_lr
        self.outer_lr = config.outer_lr
        self.inner_steps = config.inner_steps
        self.meta_optimizer = optim.Adam(self.base_net.parameters(), lr=self.outer_lr)
    
    def forward(self, support_images, support_masks, query_images, query_masks):
        temp_net = self.inner_loop(support_images,support_masks)
        loss = self.compute_loss(temp_net, query_images,query_masks)
        return loss

    def inner_loop(self, support_images, support_masks):
        temp_net = deepcopy(self.base_net)
        inner_optimizer = optim.SGD(temp_net.parameters(), lr=self.inner_lr)
    
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            loss = self.compute_loss(temp_net, support_images, support_masks)
            loss.backward()
            inner_optimizer.step()
        
        return temp_net

    def compute_loss(self, model, images, masks):
        # 根据数据计算损失，这里需要根据具体任务来实现
        """
        images: "support_images:" [tensor([[[]]])，tensor([[[]]])] length: n_way * k_shots(k_query)
        write iteration for data
        """
        criterion = nn.CrossEntropyLoss()
        # calculate the loss of each images according to the masks
        predict = model(images)
        masks = torch.squeeze(masks,dim=1).long()
        temp_loss = criterion(predict,masks)

        return temp_loss

    def evaluation_basenet(self,query_images,query_masks):
        predicted = self.base_net(query_images)
        predicted = torch.argmax(predicted,dim=1).long()
        predict_numpy = predicted.detach().cpu().numpy().reshape(-1)
        masks_numpy = query_masks.long().detach().cpu().numpy().reshape(-1)
        accuracy = metrics.accuracy_score(masks_numpy,predict_numpy)
        f1_score = metrics.f1_score(masks_numpy,predict_numpy,average=None)
        return accuracy,f1_score

    if __name__ == "__main__":
        pass