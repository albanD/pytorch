import torch
from torchvision import models

from utils import make_functional, load_weights

def get_resnet18(device):
    N = 10
    model = models.resnet18(pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    params, names = make_functional(model)

    inputs = torch.rand([N, 3, 224, 224], device=device)
    labels = torch.rand(N, device=device).mul(10).long()

    def forward(*new_params):
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    return forward, params

def get_fcn_resnet(device):
    N = 10
    criterion = torch.nn.MSELoss()
    model = models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=False)
    model.to(device)
    params, names = make_functional(model)

    inputs = torch.rand([N, 3, 224, 224], device=device)
    # Given model has 21 classes
    labels = torch.rand([N, 21, 224, 224], device=device)

    def forward(*new_params):
        load_weights(model, names, new_params)
        out = model(inputs)['out']

        loss = criterion(out, labels)
        return loss

    return forward, params

