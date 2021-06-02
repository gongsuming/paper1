import torch
from torch import nn
from torch.nn import functional as F

class SELayer_number1(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer_number1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            nn.Sigmoid()
        )
        self.temp = nn.Conv2d(channel, channel, (s, s), groups=channel, bias = False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.temp(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_number2(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer_number2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            nn.Sigmoid()
        )
        self.temp = nn.Conv2d(channel, channel, (s, s), groups=channel, bias = False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.temp(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + y.expand_as(x)

class SELayer_number3(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer_number3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
        )
        self.temp = nn.Conv2d(channel, channel, (s, s), groups=channel, bias = False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.temp(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + y.expand_as(x)

class SELayer_number4(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer_number1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            nn.ReLU6(inplace=True)
        )
        self.temp = nn.Conv2d(channel, channel, (s, s), groups=channel, bias = False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.temp(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ST(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ST, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()  
        input1 = x.view(batch // 8, 8, channel, height, width)    
        input1 = x.view(batch, channel, height * width)    
        input1 = input1.unsqueeze(2)
        input2 = x.view(batch // 8, 8, channel, height, width)     
        input2 = input2.view(batch // 8, 8 * channel, height * width)    
        input2 = input2.unsqueeze(1)                  
        input2 = self.conv1(input2)           
        input2 = input2.squeeze(1)              
        input2 = input2.view(batch , channel, height , width)  
        input2 = self.conv2(input2)    
        input2 = input2.view(batch,  1, height * width)    
        input2 = input2.unsqueeze(-1)                         
        input2 = self.softmax(input2)                     
        out = torch.matmul(input1, input2)   
        out = out.view(batch, channel, 1)        

        return out

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.spatial_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class DS(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(DS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            #nn.GroupNorm(1, self.channel // reduction),          #说明：可以使用BN、GN、无
            #nn.LayerNorm([self.channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            #nn.Sigmoid()               #说明：可以使用、无
        )

        self.temp = nn.Conv2d(channel, channel, (s, s), groups=channel, bias = False)

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.temp(x).view(b,c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class try_att(nn.Module):
    def __init__(self, net, channel, s):
        super(try_att, self).__init__()
        self.block = net
        #self.se = DS(channel, s)
        self.se = ST(channel)
        #self.se = SELayer_number1(channel, s)
        #self.se = SELayer_number2(channel, s)
        #self.se = SELayer_number3(channel, s)
        #self.se = SELayer_number4(channel, s)

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)

        return x

def NL3DWrapper(stage, channel, s):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i %2 == 0:
            blocks[i].bn3 = try_att(b.bn3, channel = channel, s = s)

    return nn.Sequential(*blocks)


def make_non_local(net, n_segment):
    import torchvision

    if isinstance(net, torchvision.models.ResNet):
        net.layer1 = NL3DWrapper(net.layer1, channel=256, s = 56)
        net.layer2 = NL3DWrapper(net.layer2, channel=512, s = 28 )
        net.layer3 = NL3DWrapper(net.layer3, channel=1024, s = 14)
        net.layer4 = NL3DWrapper(net.layer4, channel=2048, s = 7)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch 
    import torchvision.models as models
    model = models.resnet50(pretrained=False)

    input = torch.randn(8,3,224,224)
    make_non_local(model, n_segment=8)
    out = model(input)

    for k, v in model.state_dict().items():
        print(k)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)      
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)    