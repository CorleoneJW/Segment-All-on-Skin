from torchvision import transforms
from torch import nn
from utils import *

from datetime import datetime

class setting_config:
    """
    
    the config of training or testing setting.
    category_dictionary = {         # the dictionary for different categories
    'akiec':,
    'bcc':,
    'bkl':,
    'df':,
    'mel':,
    'nv':,
    'vasc':,
    }

    """
    gpu_id = '3'
    categories = ['mel','bkl',"bcc"]      # the categories of meta learning
    num_classes = len(categories)+1         # the number of categories, add the background
    epoch_num = 100                    # the number of training the meta net
    inner_steps = 2                    # the number of inner steps of iteration
    batch_size = 64                 # the batch size of training or testing
    dataloader_bs = 1               # the batch size of dataloader(corresponding to dataloader_bs * n_way * k_shot every batch)
    n_way = 2                       # n ways, should be smaller than the number of categories
    k_shot = 10                     # k shots, the number of each subset, k for support set
    k_query = 10                     # k for the evaluation, k for query set
    resize_h = 128                    # the height of resize in transformer
    resize_w = 128                  # the width of resize in transformer
    startidx = 0                     # the index that data starts
    train_set = "./data/HAM10000/train" # the root path of train set
    test_set = "./data/HAM10000/test"  # the root path of test set
    val_set = "./data/HAM10000/val"  # the root path of validation set
    in_channels = 3                         # According to the RBG image, the input channels should be 3
    out_channels = 1
    inner_lr = 1e-3
    outer_lr = 1e-4                          # as the lr in training baseline
    criterion = nn.BCEWithLogitsLoss()
    num_workers = 0
    train_print = 2                        # print the train result every (train_print) step
    evaluation_point = 5                  # evaluate the model every (evaluation_point) step
    network = 'baseline'

    train_transformer = transforms.Compose([
        # myNormalize("isic18", train=True),
        transforms.Resize((resize_w,resize_h)),
        transforms.ToTensor(),
        # myRandomHorizontalFlip(p=0.5),
        # myRandomVerticalFlip(p=0.5),
        # myRandomRotation(p=0.5, degree=[0, 360]),
        # myResize(resize_h, resize_w)
    ])
    test_transformer = transforms.Compose([
        # myNormalize("isic18", train=False),
        transforms.Resize((resize_w,resize_h)),
        transforms.ToTensor(),
        # myResize(resize_h, resize_w)
    ])
    mask_transformer = transforms.Compose([     # for train and test dataloader ,the mask transformer are the same
        transforms.Resize((resize_w,resize_h)),
        maskToTensor(),
        # myResize(resize_h, resize_w)
    ])

    catStr = ""
    for i,category in enumerate(categories):
        if i != 0:
            catStr = catStr+"+"
        catStr = catStr + str(category)
    
    work_dir = 'results/' + network + '_' + catStr + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        'c_list': [8,16,24,32,48,64], 
        'bridge': True,
        'gt_ds': True,
    }

    datasets = 'isic17' 
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    pretrained_path = './pre_trained/'
    distributed = False
    local_rank = -1
    seed = 42
    world_size = None
    rank = None
    amp = False

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = outer_lr # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = outer_lr # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = outer_lr # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = outer_lr # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = outer_lr # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = outer_lr # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = outer_lr # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = outer_lr # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = outer_lr # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = outer_lr # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epoch_num // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20