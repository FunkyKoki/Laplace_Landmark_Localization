import torch.utils.data as data
from .datasetsInfo import datasetRoot
from .datasetsTools import getRawAnnosList, getItem, getUnsupervisedItem

import glob


class Face300W80(data.Dataset):

    def __init__(self, split='train', augment=True, datasetSourcePath=datasetRoot+"300W80/"):
        annoFile = datasetSourcePath + "300W80_" + split +".txt"
        self.datasetSourcePath = datasetSourcePath
        self.dataset = "300W80"
        self.annosList = getRawAnnosList(annoFile)
        self.augment = augment
    
    def __len__(self):
        return len(self.annosList)

    def __getitem__(self, idx):
        return getItem(self.datasetSourcePath, self.dataset, self.annosList[idx], self.augment)
    

class WiderFaceUnsupervised80(data.Dataset):
    
    def __init__(self, useAmount=10000, augment=True, datasetSourcePath=datasetRoot+"WiderFaceUnsupervised80/"):
        self.augment = augment
        self.useAmount = useAmount
        self.fileList = glob.glob(datasetSourcePath+'*')
    
    def __len__(self):
        return self.useAmount
    
    def __getitem__(self, idx):
        return getUnsupervisedItem(self.fileList[idx], self.augment)
        