# model/vit.py
import timm
import torch.nn as nn

def get_vit_model(name="vit_base_patch16_224", num_classes=100, pretrained=True):
    model = timm.create_model(name, pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
