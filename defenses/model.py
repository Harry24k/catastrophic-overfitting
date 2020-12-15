import torch.nn as nn

from .models.preact_resnet import PreActBlock, PreActResNet
from .models.wide_resnet import WideResNet
from .models.normalize import Normalize, Identity

def get_model(name, num_classes, fc_input_dim_scale=1):
    
    # CIFAR10 w/ Normalize Layer
    norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    if name == "WRN28" :
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN28-D0" :
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN34" :
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN34-D0" :
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40" :
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40-2" :
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40-W2-D0" :
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "PRN18" :
        model = PreActResNet(PreActBlock, num_blocks=[2,2,2,2], num_classes=num_classes, fc_input_dim_scale=fc_input_dim_scale)
        
    print(name, "is loaded.")
        
    return nn.Sequential(norm, model)