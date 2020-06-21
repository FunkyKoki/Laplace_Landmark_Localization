# 该文件用于测试代码的正确性 check check check

from models import getRealHeatmaps, checkRealHeatmaps
from datasets import Face300W80
import torch
trainSet = Face300W80(split="train", augment=True)
dataLoader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
for data in dataLoader:
    img, paramsTarget, _, _, _ = data
    print(paramsTarget)
    print(paramsTarget.size())
    checkRealHeatmaps(getRealHeatmaps(paramsTarget))
    break
    