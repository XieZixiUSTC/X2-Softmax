import torch
from torch import nn
import math
from typing import Callable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)


class my_CE_0(torch.nn.Module):
    def __init__(self, loss_function: Callable, embedding_size: int, num_classes: int):
        super(my_CE_0, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # margin_loss
        if isinstance(loss_function, Callable):
            self.loss_function = loss_function
        else:
            raise

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        weight = self.weight

        with torch.cuda.amp.autocast(False):
            logits = nn.functional.linear(embeddings, weight)

        logits = self.loss_function(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss


class my_CE_1(torch.nn.Module):
    # for LSoftmax and ASoftmax
    def __init__(self, loss_function: Callable, embedding_size: int, num_classes: int):
        super(my_CE_1, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # margin_loss
        if isinstance(loss_function, Callable):
            self.loss_function = loss_function
        else:
            raise

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Embedding : m , d
        weight = self.weight

        with torch.cuda.amp.autocast(False):
            logits = nn.functional.linear(embeddings, weight)  # [m, d] [c, d]
            norm_embeddings = nn.functional.normalize(embeddings)
            norm_weight_activated = nn.functional.normalize(weight)

        logits = self.loss_function(logits, labels, norm_embeddings, norm_weight_activated)
        loss = self.cross_entropy(logits, labels)
        return loss


class my_CE_2(torch.nn.Module):
    # 权重和特征都做归一化
    def __init__(self, loss_function: Callable, embedding_size: int, num_classes: int):
        super(my_CE_2, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # margin_loss
        if isinstance(loss_function, Callable):
            self.loss_function = loss_function
        else:
            raise

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        weight = self.weight

        with torch.cuda.amp.autocast(False):
            norm_embeddings = nn.functional.normalize(embeddings)
            norm_weight_activated = nn.functional.normalize(weight)
            logits = nn.functional.linear(norm_embeddings, norm_weight_activated)

        logits = self.loss_function(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss


class MagLinear(torch.nn.Module):
    def __init__(self, loss_function: Callable, embedding_size: int, num_classes: int, parameters):
        super(MagLinear, self).__init__()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))  # c, d

        self.s = parameters[0]
        self.l_m = parameters[1]
        self.u_m = parameters[2]
        self.l_a = parameters[3]
        self.u_a = parameters[4]
        self.lamb = parameters[5]

        if isinstance(loss_function, Callable):
            self.loss_function = loss_function
        else:
            raise

    def _margin(self, x):
        margin = (self.u_m - self.l_m) / (self.u_a - self.l_a) * (x - self.l_a) + self.l_m
        return margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        x_p = torch.norm(embeddings, dim=1, keepdim=True).clamp(self.l_a, self.u_a)  # m, 1
        ada_m = self._margin(x_p)
        cos_m, sin_m = torch.cos(ada_m), torch.sin(ada_m)

        weight_norm = nn.functional.normalize(self.weight)  # c, 1
        x_norm = nn.functional.normalize(embeddings)
        costheta = nn.functional.linear(x_norm, weight_norm)
        costheta = costheta.clamp(-1, 1)
        sintheta = torch.sqrt(1.0 - torch.pow(costheta, 2))
        costheta_m = costheta * cos_m - sintheta * sin_m

        mm = torch.sin(math.pi - ada_m) * ada_m
        threshold = torch.cos(math.pi - ada_m)
        costheta_m = torch.where(costheta > threshold, costheta_m, costheta - mm)

        costheta = costheta * self.s
        costheta_m = costheta_m * self.s

        loss = self.loss_function(costheta, costheta_m, labels, x_p)
        return loss


class DynArcLinear(torch.nn.Module):
    # 权重和特征都做归一化
    def __init__(self, loss_function: Callable, embedding_size: int, num_classes: int):
        super(DynArcLinear, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # margin_loss
        if isinstance(loss_function, Callable):
            self.loss_function = loss_function
        else:
            raise

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        weight = self.weight

        with torch.cuda.amp.autocast(False):
            norm_embeddings = nn.functional.normalize(embeddings)
            norm_weight_activated = nn.functional.normalize(weight)
            logits = nn.functional.linear(norm_embeddings, norm_weight_activated)

        logits = self.loss_function(logits, labels, norm_weight_activated)
        loss = self.cross_entropy(logits, labels)
        return loss
