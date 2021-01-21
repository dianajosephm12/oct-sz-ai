import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
import sys
from torch.autograd import Variable
from models.conv_nn import ConvNet


class CareArch(nn.Module):
    def __init__(self, in_channels):
        super(CareArch, self).__init__()
        self.layers = nn.ModuleList()
        self.inf_path = None

        #3x3 , symmetric padding, relu activation, 32 features -- 4 of these
        self.layers.append(torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        #max pool 2x2
        self.layers.append(torch.nn.MaxPool2d(2))

        #3x3 , symmetric padding, relu activation, 64 features -- 4 of these
        self.layers.append(torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        #max pool 2x2
        self.layers.append(torch.nn.MaxPool2d(2))
        #3x3 , symmetric padding, relu activation, 128 features -- 4 of these
        self.layers.append(torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        #upsample -- nearest
        self.layers.append(torch.nn.Upsample(scale_factor=2))
        #3x3 , symmetric padding, relu activation, 64 features -- 4 of these
        self.layers.append(torch.nn.Conv2d(192, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        #upsample-- nearest
        self.layers.append(torch.nn.Upsample(scale_factor=2))
        #3x3 , symmetric padding, relu activation, 32 features -- 4 of these
        self.layers.append(torch.nn.Conv2d(96, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='reflect'))
        self.layers.append(torch.nn.ReLU(inplace=True))
        #Need a 1x1 filter for final conv
        self.layers.append(torch.nn.Conv2d(32, 1, 1))
        self.encoding = None


    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i == 7:
                before_skip_0 = x
            if i == 16:
                before_skip_1 = x
            if i == 26:
                x = torch.cat((x,before_skip_1),1)
            if i ==35:
                x = torch.cat((x,before_skip_0),1)
        return x

    def inf(self, x, corner, img_id):
        for i in range(len(self.layers)):
            if i == 8:
                np.save('/media/twrichar/Elements/carestream/gram_final/gram_matrices/features/denoise'+ str(self.inf_path) + '_1_'+str(corner)+ '_' + str(img_id) + '.npy', x.cpu().numpy())
            elif i ==17:
                np.save('/media/twrichar/Elements/carestream/gram_final/gram_matrices/features/denoise'+ str(self.inf_path) + '_2_'+str(corner)+ '_' + str(img_id) + '.npy', x.cpu().numpy())
            elif i ==26:
                np.save('/media/twrichar/Elements/carestream/gram_final/gram_matrices/features/denoise'+ str(self.inf_path) + '_3_'+str(corner)+ '_' + str(img_id) + '.npy', x.cpu().numpy())
            elif i ==35:
                np.save('/media/twrichar/Elements/carestream/gram_final/gram_matrices/features/denoise'+ str(self.inf_path) + '_4_'+str(corner)+ '_' + str(img_id) + '.npy', x.cpu().numpy())
            elif i ==44:
                np.save('/media/twrichar/Elements/carestream/gram_final/gram_matrices/features/denoise'+ str(self.inf_path) + '_5_'+str(corner)+ '_' + str(img_id) + '.npy', x.cpu().numpy())
            x = self.layers[i](x)
            if i == 7:
                before_skip_0 = x
            if i == 16:
                before_skip_1 = x
            if i == 26:
                x = torch.cat((x,before_skip_1),1)
            if i ==35:
                x = torch.cat((x,before_skip_0),1)

        return x, 0

    def inf_2(self, x):
        for i in range(len(self.layers)):
            if i == 8:
                np.save('/media/twrichar/Elements/RDSC_1/gram_matrices/single/features/denoise'+ str(self.inf_path) + '_1.npy', x.cpu().numpy())
            elif i ==17:
                np.save('/media/twrichar/Elements/RDSC_1/gram_matrices/single/features/denoise'+ str(self.inf_path) + '_2.npy', x.cpu().numpy())
            elif i ==26:
                np.save('/media/twrichar/Elements/RDSC_1/gram_matrices/single/features/denoise'+ str(self.inf_path) + '_3.npy', x.cpu().numpy())
            elif i ==35:
                np.save('/media/twrichar/Elements/RDSC_1/gram_matrices/single/features/denoise'+ str(self.inf_path) + '_4.npy', x.cpu().numpy())
            elif i ==44:
                np.save('/media/twrichar/Elements/RDSC_1/gram_matrices/single/features/denoise'+ str(self.inf_path) + '_5.npy', x.cpu().numpy())
            x = self.layers[i](x)
            if i == 7:
                before_skip_0 = x
            if i == 16:
                before_skip_1 = x
            if i == 26:
                x = torch.cat((x,before_skip_1),1)
            if i ==35:
                x = torch.cat((x,before_skip_0),1)

        return x, 0

    def inf_3(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i == 16:
                return x

        return x