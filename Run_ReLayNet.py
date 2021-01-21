#!/usr/bin/env python
# coding: utf-8

# ## Train ReLayNet
# RunFile of OCT segmentation

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from networks.relay_net import ReLayNet
from networks.data_utils import get_imdb_data

#torch.set_default_tensor_type('torch.FloatTensor')

plt.switch_backend('MacOsx')

# run as $ ipython python/filename.py https://stackoverflow.com/a/32539282
get_ipython().run_line_magic('matplotlib', 'osx')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:

# GET DATA
# train_data, test_data = get_imdb_data()
# print("Train size: %i" % len(train_data))
# print("Test size: %i" % len(test_data))


# In[3]:
# TRAINING MODEL
#
# from networks.relay_net import ReLayNet
# from networks.solver import Solver
#
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)
#
# param ={
#         'num_channels':1,
#         'num_filters':64,
#         'kernel_h':3,
#         'kernel_w':7,
#         'kernel_c': 1,
#         'stride_conv':1,
#         'pool':2,
#         'stride_pool':2,
#         'num_class':9
#     }
#
# exp_dir_name = 'Exp01'
#
# relaynet_model = ReLayNet(param)
# solver = Solver(optim_args={"lr": 1e-2})
# solver.train(relaynet_model, train_loader, val_loader, log_nth=1, num_epochs=20, exp_dir_name=exp_dir_name)


# ## Save the Model
# 
# When you are satisfied with your training, you can save the model.

# In[4]:


# relaynet_model.save("models/relaynet_model.model")  #SAVING MODEL


# # Deploy Model on Test Data

# In[5]:


SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];
    #{"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];
    
def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


# In[8]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
import os

fp = 'Control30Up_resized'
img_list = os.listdir(fp)
print(len(img_list))
i = 0
while i < 37:   # up to i < 41 for ControlUnder30, i < 29 for fep, i < 37 for all else
    if img_list[i].endswith('.png'):
        img_name = img_list[i]
        print(i)
        print(img_name)
        dj_img_nsa = mpimg.imread('{fp}/{img_name}'.format(fp=fp, img_name=img_name))
        dj_img_nsa = dj_img_nsa[np.newaxis, np.newaxis, :, :]  # FOR GRAYSCALE
        dj_img_tens = torch.from_numpy(dj_img_nsa)
        relaynet_model = torch.load('models/Exp01/relaynet_epoch20.model', map_location='cpu')
        out = relaynet_model(Variable(torch.Tensor(dj_img_tens), volatile=True))
        root_name = img_name[6:-4]
        os.rename('d1.npy', 'd1_{root_name}.npy'.format(root_name=root_name))
        os.rename('d2.npy', 'd2_{root_name}.npy'.format(root_name=root_name))
        os.rename('d3.npy', 'd3_{root_name}.npy'.format(root_name=root_name))
        # os.rename('bn.npy', 'bn_{root_name}.npy'.format(root_name=root_name))
    i = i + 1

# img_name = img_list[7]
#
# print(img_name)
# # img_name = 'raw_valid_rgb.png'
# # dj_img_nsa = mpimg.imread('normalized_test_imgs/{img_name}'.format(img_name=img_name))
# dj_img_nsa = mpimg.imread('{fp}/{img_name}'.format(fp=fp, img_name=img_name))
# print(dj_img_nsa.shape)
# # 4D TENSOR: [NUMDATA, CHANNELS, ROWS, COLS]
# dj_img_nsa = dj_img_nsa[np.newaxis, np.newaxis, :, :]  # FOR GRAYSCALE
# # dj_img_nsa = dj_img_nsa[np.newaxis, np.newaxis, :, :, 0]  # FOR RGB
# dj_img_tens = torch.from_numpy(dj_img_nsa)
# print(dj_img_tens.shape)
#
#
# # ADDED MAP_LOCATION
# relaynet_model = torch.load('models/Exp01/relaynet_epoch20.model', map_location='cpu')
# # out = relaynet_model(Variable(torch.Tensor(test_data.X[0:1]).cuda(),volatile=True))
# out = relaynet_model(Variable(torch.Tensor(dj_img_tens), volatile=True))
# # out = F.softmax(out, dim=1)
# # max_val, idx = torch.max(out, 1)
# # idx = idx.data.cpu().numpy()
# # print(idx.shape)
# # idx = label_img_to_rgb(idx)
# # # mpimg.imsave('result_{img_name}'.format(img_name=img_name), idx)
#
# root_name = img_name[6:-4]
# os.rename('e1.npy', 'e1_{root_name}.npy'.format(root_name=root_name))
# os.rename('e2.npy', 'e2_{root_name}.npy'.format(root_name=root_name))
# os.rename('e3.npy', 'e3_{root_name}.npy'.format(root_name=root_name))
# os.rename('bn.npy', 'bn_{root_name}.npy'.format(root_name=root_name))
#
# # plt.imshow(idx)
# # plt.show(block=True)        # CHANGE BASED ON https://stackoverflow.com/a/48032629
# #
# # # img_test = test_data.X[0:1]
# # img_test = dj_img_tens
# # img_test = np.squeeze(img_test)
# # plt.imshow(img_test)
# # plt.show(block=True)        # CHANGED BASED ON https://stackoverflow.com/a/48032629
# # plt.clf()
# # plt.cla()
# # plt.close()
