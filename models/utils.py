import torch
from collections import OrderedDict

import numpy as np
import cv2
import matplotlib.pyplot as plt

class dataPrefetcher():

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.next_input, self.next_target, _, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        targets = self.next_target
        self.preload()
        return inputs, targets
    def __iter__(self):
        return self

    
class dataPrefetcherUnsupervised():

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        self.preload()
        return inputs
    def __iter__(self):
        return self    
    

def loadWeights(net, pth_file, device):
    state_dict = torch.load(pth_file, map_location=device)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


def getRealHeatmaps(paramsTarget, delta=1.0):
    coords = paramsTarget.cpu().numpy()[:, :, 0:2]
    heatmapsReal = np.ones([coords.shape[0], coords.shape[1], 80, 80])*255
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            coord = coords[i, j, :]
            coord = np.clip(coord, 0, 79)
            heatmapsReal[i, j, int(coord[1]), int(coord[0])] = 0
            pic = heatmapsReal[i, j, :, :]
            pic = np.array(pic, dtype=np.uint8)
            pic = cv2.distanceTransform(pic, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            pic = np.exp(-pic*pic/(2*delta*delta))
            pic[pic < np.exp(-3*delta/(2*delta*delta))] = 0
            heatmapsReal[i, j, :, :] = pic
            
    return torch.Tensor(heatmapsReal).type_as(paramsTarget)


def checkRealHeatmaps(heatmapsReal):
    heatmapsReal = heatmapsReal.cpu().numpy()
    for i in range(heatmapsReal.shape[0]):
        for j in range(heatmapsReal.shape[1]):
            plt.figure('name')
            plt.imshow(heatmapsReal[i, j, :, :])
            plt.axis('off')
            plt.title('name')
            plt.imsave('/home/jin/Desktop/temp'+str(j)+'.png', heatmapsReal[i, j, :, :])
        break
            