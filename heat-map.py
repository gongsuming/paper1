import io
from PIL import Image
import torch
from torchvision import models , transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import os
import torchvision.models as models
from re_i1 import make_non_local
for i in range(122):
#from rgb_resnet import rgb_resnet50

    savepath= 'E://features'


    net = models.resnet50(pretrained=False)


    #from try_att import make_non_local          ours
    #make_non_local(net, n_segment=8)            ours

    num_fc_ftr = net.fc.in_features
    net.fc = torch.nn.Linear(num_fc_ftr, 101)

    pretrained_dict = torch.load("D://my_torch//tsm_result//i1//ucf//ds//baseline_heart.pth", map_location='cpu')
    model_dict = net.state_dict()
# # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
# # 3. load the new state dict
    net.load_state_dict(model_dict)

    finalconv_name = 'layer4'

    net.eval()

# hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
 
# get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256

        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape

        output_cam = []

        for idx in range(0,101):
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    lujing = os.path.join('D://my_torch//tsm_result//i1//ucf//ds//5//', 'frame{:06d}.jpg'.format(i+1))

    transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img1=cv2.imread(lujing)
    img1=cv2.resize(img1,(224,224))
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img1=transform(img1)
    img_variable=img1.unsqueeze(0)

    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])



    img1 = cv2.imread(lujing)
    height, width, _ = img1.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img1 * 0.5
    baocunlujing = os.path.join('E://features//5-2//', 'frame{:06d}.jpg'.format(i+1))
    cv2.imwrite(baocunlujing, result)                          
