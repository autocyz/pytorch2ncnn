
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch


class small(nn.Module):

    def __init__(self, features, num_classes=2):
        super(small, self).__init__()
        self.features = features
        self.fc1 = nn.Linear(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(14)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [16, 'M', 32, 'M', 32, 32, 'M', 64, 64, 'M', 256],
    'B': [16, 'M', 32, 'M', 32, 'M', 32, 'M', 256],
}


def smallA(**kwargs):
    model = small(make_layers(cfg['A']), **kwargs)
    return model


def smallA_bn(**kwargs):
    model = small(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

def smallB(**kwargs):
    model = small(make_layers(cfg['B']), **kwargs)
    return model


def smallB_bn(**kwargs):
    model = small(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model

# test case
if __name__ == '__main__':
    net = smallA_bn()
    a = torch.randn(4, 3, 224, 224)
    out = net(a)
    print(out.shape)
