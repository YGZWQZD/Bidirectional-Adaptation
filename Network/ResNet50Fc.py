from torchvision import models
import torch.nn as nn
import torch
from torch.autograd import Variable

def to_var(x, requires_grad=True,device='cuda:0'):
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad)

class ResNet50Fc(nn.Module):

    def __init__(self, num_classes, output_feature=False):
        super(ResNet50Fc, self).__init__()
        _model_resnet = models.resnet50(pretrained=True)
        model_resnet = _model_resnet
        self.conv1 = model_resnet.conv1
        self.bn_source = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        __in_features = model_resnet.fc.in_features
        self.output_feature=output_feature
        self.bottleneck = nn.Linear(__in_features, 256)
        self.relu_bottle = nn.ReLU()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x,source=False):
        x = self.conv1(x)
        x = self.bn_source(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x= self.relu_bottle(x)
        c = self.fc(x)
        if self.output_feature:
            return x, c
        else:
            return c

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False,device='cuda:0'):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data,device=device)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp,grad,device=device)
        else:

            for name, param in self.named_parameters():
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data,device=device)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp,device=device)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param,device=device)

    def set_param(self, curr_mod, name, param,grad=None,device='cuda:0'):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
    def output_num(self):
        return self.__in_features