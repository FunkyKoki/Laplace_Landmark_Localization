import tqdm
import copy
import torch
import numpy as np
import cv2
from models import Generator, loadWeights, GeneratorGhostConv
from datasets import Face300W80, datasetSize
import time
from thop import profile, clever_format

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def testFace300W80(split, test_epoch, logName, rat=1, tempera=2.0):
    testset = Face300W80(split=split, augment=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

#     inputs = torch.randn(1, 3, 80, 80)
#     flops, params = profile(Generator(ratio=1, temperature=1.), inputs=(inputs, ))
#     flops, params = profile(GeneratorGhostConv(ratio=1, temperature=1.), inputs=(inputs, ))
#     flops, params = clever_format([flops, params], "%.3f")
#     print(flops, params)
    
#     model = GeneratorGhostConv(ratio=rat, temperature=tempera)
    model = Generator(ratio=rat, temperature=tempera)
    model = loadWeights(model, '/media/WDC/savedModels/LaplaceLandmarkLocalization/'+logName+'_model_'+str(test_epoch)+'.pth', 'cuda:0')
    model.eval().cuda('cuda:0')

    errorRates = []
    times = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            img, paramsTarget, _, _, _ = data
            img = img.cuda('cuda:0').float()
            paramsTarget = paramsTarget.squeeze().numpy()

            startTime = time.time()
            paramsPredict, _ = model(img)
            times.append(time.time()-startTime)
            paramsPredict = paramsPredict.cpu().squeeze().numpy()
            
            pts = paramsPredict[:, 0:2]
            tpts = paramsTarget[:, 0:2]
            
#             normalizeFactor = np.sqrt(((tpts[39, 0]+tpts[36, 0])/2 - (tpts[42, 0]+tpts[45, 0])/2)**2 + ((tpts[39, 1]+tpts[36, 1])/2 - (tpts[42, 1]+tpts[45, 1])/2)**2)
            normalizeFactor = np.sqrt((tpts[36, 0] - tpts[45, 0])**2 + (tpts[36, 1] - tpts[45, 1])**2)

            errorRate = np.sum(np.sqrt(np.sum(pow(pts-tpts, 2), axis=1)))/68/normalizeFactor
            # if errorRate > 0.03:
                # showImgLandmarksInGetItem(imgRaw, pts, name=imgName, oneByOne=False, KPTNUM=19)
                # showImgGtAndPredLandmarks(imgRaw, tpts, pts, name=imgName, oneByOne=False, KPTNUM=19)

            errorRates.append(errorRate)

    errorRate = sum(errorRates) / datasetSize["300W80"][split] * 100
    print(split + ' error rate: ' + str(errorRate))
    print("Avg forward time is: " + str(sum(times)/datasetSize["300W80"][split]) + "ms")
    print("FPS: " + str(1/sum(times)*datasetSize["300W80"][split]))
    return errorRate


def testFace300W80CPU(split, test_epoch, logName):
    testset = Face300W80(split=split, augment=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

#     inputs = torch.randn(1, 3, 80, 80)
#     flops, params = profile(Generator(ratio=1, temperature=1.), inputs=(inputs, ))
#     flops, params = profile(GeneratorGhostConv(ratio=1, temperature=1.), inputs=(inputs, ))
#     flops, params = clever_format([flops, params], "%.3f")
#     print(flops, params)
    
#     model = Generator(ratio=1, temperature=2.)
    model = GeneratorGhostConv(ratio=1, temperature=2.)
    model = loadWeights(model, '/media/WDC/savedModels/LaplaceLandmarkLocalization/'+logName+'_model_'+str(test_epoch)+'.pth', 'cpu')
    model.eval()

    errorRates = []
    times = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            img, paramsTarget, _, _, _ = data
            img = img.float()
            paramsTarget = paramsTarget.squeeze().numpy()

            startTime = time.time()
            paramsPredict, _ = model(img)
            times.append(time.time()-startTime)
            paramsPredict = paramsPredict.squeeze().numpy()
            
            pts = paramsPredict[:, 0:2]
            tpts = paramsTarget[:, 0:2]
            
            normalizeFactor = np.sqrt((tpts[36, 0] - tpts[45, 0])**2 + (tpts[36, 1] - tpts[45, 1])**2)

            errorRate = np.sum(np.sqrt(np.sum(pow(pts-tpts, 2), axis=1)))/68/normalizeFactor
            # if errorRate > 0.03:
                # showImgLandmarksInGetItem(imgRaw, pts, name=imgName, oneByOne=False, KPTNUM=19)
                # showImgGtAndPredLandmarks(imgRaw, tpts, pts, name=imgName, oneByOne=False, KPTNUM=19)

            errorRates.append(errorRate)

    errorRate = sum(errorRates) / datasetSize["300W80"][split] * 100
    print(split + ' error rate: ' + str(errorRate))
    print("Avg forward time is: " + str(sum(times)/datasetSize["300W80"][split]) + "ms")
    print("FPS: " + str(1/sum(times)*datasetSize["300W80"][split]))
    return errorRate


if __name__ == "__main__":
#     logName = '20200606_face300W80GAN_Train_3'
#     logName = '20200525_face300W80_Train_2'
    logName = '20200523_face300W80_Train_5'
    testFace300W80("full", 'best', logName)