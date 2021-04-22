import torch
from torch import nn
from torch.nn import functional as F

class SELayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3,stride=1, padding=1)
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

        input1 = x.unsqueeze(1)        #[ N, 1, C, H, W ]

        input1 = input1.view(batch, 1, channel, height * width)       #[ N, 1, C, H * W ]

        input2 = x                             #[ N, C, H, W ]

        input2 = self.conv1(input2)             #[ N,  C, H, W ]

        input2 = self.conv2(input2)             #[ N, 1, H, W ]

        input2 = input2.view(batch,  1, height * width)        #[ N,  1, H * W ]
  
        input2 = input2.unsqueeze(-1)                            #[ N, 1, H * W, 1 ]

        input2 = self.softmax(input2)                                #[ N, 1, H * W, 1 ]

        out = torch.matmul(input1, input2)                 #[ N, 1, C, 1 ]

        out = out.view(batch, channel, 1, 1)

        return out

    def forward(self, x):
        b, c, h, w = x.size()
        print("2:", b,c,h,w)
        y = self.spatial_pool(x).view(b, c)
        print("3:",y.shape)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class SELayer222(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer222, self).__init__()

        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.pool3d = nn.AvgPool3d(3, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()       #[ bt, C, H, W ]
        print("1:", x.shape)
        input1 = x.view(batch, channel, height * width)        #[ bt, C, H*W ]
        input1 = input1.unsqueeze(2)       #[ bt, C, 1, H * W ]
        print("3:",input1.shape)


        input2 = x.view(batch//8, 8, channel, height , width)       #[ b, t, C, H , W ]
        input2 = input2.permute(0,2,1,3,4)             #[ b,  C, t, H, W ]
        print("4:",input2.shape)
        input2 = self.pool3d(input2)             #[ b, c, t, H, W ]
        print("5:",input2.shape)
        input2 = input2.permute(0,2,1,3,4).contiguous()       #[ b, t, c, H, W ]
        input2 = input2.view(-1,  channel, height , width)        #[ bt,  c, H , W ]
        input2 = self.conv1(input2)                                   #[ bt, 1, H, W ]
        print("6:",input2.shape)
        input2 = input2.view(batch,  1, height * width)         #[ bt, 1, H*W ]
        input2 = input2.unsqueeze(-1)                            #[ bt, 1, H * W, 1 ]
        print("7:",input2.shape)
        input2 = self.softmax(input2)                                #[ bt, 1, H * W, 1 ]
        print("8:",input2.shape)
        out = torch.matmul(input1, input2)                 #[ bt, C, 1, 1 ]
        print("9:",out.shape)
        #out = out.view(batch, channel, 1, 1)             #[ bt, C, 1, 1 ]
        #print("10:",out.shape)
        return out

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.spatial_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class SELayer3(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer3, self).__init__()

        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        #self.pool3d = nn.AvgPool3d(3, 1, 1)
        self.temp = nn.Conv3d(channel, channel, (3, s, s), padding=(1, 3, 3), groups=channel, bias = False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()       #[ bt, C, H, W ]

        input1 = x.view(batch, channel, height * width)        #[ bt, C, H*W ]
        input1 = input1.unsqueeze(2)       #[ bt, C, 1, H * W ]



        input2 = x.view(batch//8, 8, channel, height , width)       #[ b, t, C, H , W ]
        input2 = input2.permute(0,2,1,3,4)             #[ b,  C, t, H, W ]

        input2 = self.temp(input2)             #[ b, c, t, H, W ]

        input2 = input2.permute(0,2,1,3,4).contiguous()       #[ b, t, c, H, W ]
        input2 = input2.view(-1,  channel, height , width)        #[ bt,  c, H , W ]
        input2 = self.conv1(input2)                                   #[ bt, 1, H, W ]

        input2 = input2.view(batch,  1, height * width)         #[ bt, 1, H*W ]
        input2 = input2.unsqueeze(-1)                            #[ bt, 1, H * W, 1 ]

        input2 = self.softmax(input2)                                #[ bt, 1, H * W, 1 ]

        out = torch.matmul(input1, input2)                 #[ bt, C, 1, 1 ]

        #out = out.view(batch, channel, 1, 1)             #[ bt, C, 1, 1 ]
        #print("10:",out.shape)
        return out

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.spatial_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class SELayer22(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer22, self).__init__()
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
        batch, channel, height, width = x.size()       #[ bt, C, H, W ]

        input1 = x.view(batch // 8, 8, channel, height, width)       #[b, t, c,h,w]

        input1 = x.view(batch, channel, height * width)       #[ bt, C, H * W ]
        input1 = input1.unsqueeze(2)


        input2 = x.view(batch // 8, 8, channel, height, width)       #[b, t, c,h,w]
        input2 = input2.view(batch // 8, 8 * channel, height * width)       #[ b, tC, H * W ]
        input2 = input2.unsqueeze(1)                   #[ b, 1, tC, H * W ]
        input2 = self.conv1(input2)             #[ b, 1, tC, H*W ]

        input2 = input2.squeeze(1)                   #[ b, tC, H * W ]

        input2 = input2.view(batch , channel, height , width)   #[ bt, C, H , W ]

        input2 = self.conv2(input2)             #[ bt, C, H , W ]

        input2 = input2.view(batch,  1, height * width)        #[ N,  1, H * W ]

        input2 = input2.unsqueeze(-1)                            #[ N, 1, H * W, 1 ]

        input2 = self.softmax(input2)                                #[ N, 1, H * W, 1 ]

        out = torch.matmul(input1, input2)                 #[ N, 1, C, 1 ]

        out = out.view(batch, channel, 1)             #[ N, C, 1, 1 ]

        return out

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.spatial_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)





class SELayer1(nn.Module):
    def __init__(self, channel, s, reduction=16):
        super(SELayer1, self).__init__()
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

        '''for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)

            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)   '''

    def forward(self, x):
        b, c, h, w = x.size()

        #temp = nn.Conv2d(c, c, (h,w), groups=c)#.cuda()
        #y=temp(x).view(b,c)
        y = self.temp(x).view(b,c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class try_att(nn.Module):
    def __init__(self, net, channel, s):
        super(try_att, self).__init__()
        self.block = net
        #self.se = SELayer1(channel, s)
        self.se = SELayer22(channel)

    def forward(self, x):
        x = self.block(x)
        print("1:",x.shape)
        x = self.se(x)
        #x2 = self.sa_t(x) * x
        #x = x1 + x2
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
    #print(model)

    input = torch.randn(8,3,224,224)
    make_non_local(model, n_segment=8)
    out = model(input)
    #print(model)

    '''pretrained_dict = torch.load("H://111111//test.pth")
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)   '''

    for k, v in model.state_dict().items():
        print(k)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)      
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)    