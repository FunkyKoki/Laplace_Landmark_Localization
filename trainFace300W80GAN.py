import os
import time
import math
import tqdm
import torch
import logging
from torch.autograd import Variable

from models import Generator, loadWeights, dataPrefetcher, GeneratorGhostConv, NLayerDiscriminator, dataPrefetcherUnsupervised, getRealHeatmaps
from datasets import Face300W80, datasetSize, WiderFaceUnsupervised80

from testFace300W80 import testFace300W80

torch.backends.cudnn.benchmark = True

# 生成必要的文件夹
if not os.path.exists('/media/WDC/savedModels/LaplaceLandmarkLocalization'):
    os.mkdir('/media/WDC/savedModels/LaplaceLandmarkLocalization')
if not os.path.exists('./logs'):
    os.mkdir('./logs')

logName = '20200609_face300W80GAN_Train_4'
# 生成记录训练过程的loggor
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('./logs/' + logName + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(logName)
logger.info('备注：现在开始进行GhostModule的GAN训练，此次实验，使用50K张无标签人脸图像，无标签数据“不进行”数据增强，判别器网络超参数设置为ndf=64, n_layers=4, use_noise=True, noise_sigma=0.2。判别器网络的学习率为0.001，同样在第500个epoch乘以0.1。生成器的损失函数中，判别器端带来的损失权重lossLambda=0.001，生成真实热图的参数delta=1.0（1.0为最佳）。 1.这一次实验对于KL散度loss同样进行了分母增加了极小量的修正。')

useAmountD = 50000
augmentD=False
ndfD = 64
n_layersD = 4
use_noiseD = True
noise_sigmaD = 0.2
lossLambda = 0.001
deltaGetHeatmap = 1.0

rat = 1
tempera = 2.0


logger.info('Network G: GeneratorGhostConv(ratio='+str(rat)+', keyPoints=68, dropOutRate=0.10, temperature='+str(tempera)+', imageSize=80)')
logger.info('Network D: NLayerDiscriminator(input_nc=71, ndf='+str(ndfD)+', n_layers='+str(n_layersD)+', use_noise='+str(use_noiseD)+', noise_sigma='+str(noise_sigmaD)+')')
logger.info('GPU: \'cuda:0\'')
modelG = GeneratorGhostConv(ratio=rat, keyPoints=68, dropOutRate=0.10, temperature=tempera, imageSize=80)
# model = loadWeights(model, '/media/WDC/savedModels/LaplaceLandmarkLocalization/20200512_face300W80_Train_4_model_best.pth', 'cuda:0')
modelG = modelG.cuda('cuda:0')
modelG.train()
modelD = NLayerDiscriminator(71, ndf=ndfD, n_layers=n_layersD, use_noise=use_noiseD, noise_sigma=noise_sigmaD)
modelD = modelD.cuda('cuda:0')
modelD.train()


logger.info('Generator：Adam -- lr=0.001, weight_decay=1e-5')
logger.info('Discriminator：Adam -- lr=0.001, weight_decay=1e-5')
optimizerG = torch.optim.Adam(modelG.parameters(), lr=0.001, weight_decay=1e-5)
optimizerD = torch.optim.Adam(modelD.parameters(), lr=0.001, weight_decay=1e-5)


logger.info('Training epoch: 750')
logger.info('Dataset: Face300W80(split="train", augment=False)')
logger.info('Dataloader: batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True')
# logger.info('Unsupervised Dataset: Face300W80(split="train", augment=False)')
trainSet = Face300W80(split="train", augment=True)
unsupervisedSet = WiderFaceUnsupervised80(useAmount=useAmountD, augment=augmentD)


batchSize = 16
batchsizeUnsupervised = 16
maxEpoch = 750
numWorkers = 8
itersPerBatch = datasetSize["300W80"]["train"]//batchSize

errorRate=100.0

for epoch in range(0, maxEpoch):
    print("Epoch: " + str(epoch))
    st = time.time()
    lossRecordPerEpoch = []
    lossDRecordPerEpoch = []
    
    if epoch in [500]:
        optimizerG.param_groups[0]['lr'] *= 0.1
        optimizerD.param_groups[0]['lr'] *= 0.1

    dataLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)
    dataLoaderUnsupervised = torch.utils.data.DataLoader(unsupervisedSet, batch_size=batchsizeUnsupervised, shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)
    prefetcher = dataPrefetcher(iter(dataLoader))
    prefetcherUnsupervised = dataPrefetcherUnsupervised(iter(dataLoaderUnsupervised))

    for _ in tqdm.tqdm(range(itersPerBatch)):

        img, paramsTarget = prefetcher.next()
        unlabelImg = prefetcherUnsupervised.next()
        
        assert img.size()[0] == paramsTarget.size()[0] == batchSize
        assert unlabelImg.size()[0] == batchsizeUnsupervised

        img = Variable(img.cuda('cuda:0'))
        paramsTarget = Variable(paramsTarget.cuda('cuda:0'))
        unlabelImg = Variable(unlabelImg.cuda('cuda:0'))

        heatmapsReal = getRealHeatmaps(paramsTarget, delta=deltaGetHeatmap)
        
        optimizerG.zero_grad()
        out, _ = modelG(img)
        lossKL = modelG.calculateLoss(out, paramsTarget)
        _, heatmapsFake = modelG(unlabelImg)
        fakeDiscrimInput = torch.cat((heatmapsFake, unlabelImg), dim=1)
        lossG = modelD.calculateLoss(modelD(fakeDiscrimInput), True)
        lossG = lossKL + lossLambda*lossG
        lossG.backward()
        optimizerG.step()
        
        optimizerD.zero_grad()
        realDiscrimInput = torch.cat((heatmapsReal, img), dim=1)
        fakeDiscrimInput = torch.cat((heatmapsFake.detach(), unlabelImg), dim=1)
        lossReal = modelD.calculateLoss(modelD(realDiscrimInput), True)
        lossFake = modelD.calculateLoss(modelD(fakeDiscrimInput), False)
        lossD = (lossReal + lossFake)/2
        lossD.backward()
        optimizerD.step()

        lossRecordPerEpoch.append(lossKL.item())
        lossDRecordPerEpoch.append(lossD.item())

    epochTime = time.time() - st
    logger.info('epoch: ' + str(epoch) + ' lrG: ' + str(optimizerG.param_groups[0]['lr']) + ' lrD: ' + str(optimizerD.param_groups[0]['lr']) + ' lossKL: ' + str(sum(lossRecordPerEpoch)/len(lossRecordPerEpoch)) + ' lossD: ' + str(sum(lossDRecordPerEpoch)/len(lossDRecordPerEpoch)) + ' time: ' + str(epochTime))

    torch.save(modelG.state_dict(), '/media/WDC/savedModels/LaplaceLandmarkLocalization/' + logName + '_model_tempG.pth')
    errorRateTemp = testFace300W80('full', 'tempG', logName, rat=rat, tempera=tempera)
    if errorRateTemp < errorRate:
        errorRate = errorRateTemp
        torch.save(modelG.state_dict(), '/media/WDC/savedModels/LaplaceLandmarkLocalization/' + logName + '_model_bestG.pth')
        torch.save(modelD.state_dict(), '/media/WDC/savedModels/LaplaceLandmarkLocalization/' + logName + '_model_bestD.pth')
        errorRateCommon = testFace300W80('common', 'bestG', logName, rat=rat, tempera=tempera)
        errorRateChallenge = testFace300W80('challenge', 'bestG', logName, rat=rat, tempera=tempera)
        logger.info('Performance becomes better in epoch: ' + str(epoch))
        logger.info('Now the error rate is: ' + str(errorRate) + '%  common--' + str(errorRateCommon) + '%  challenge--' + str(errorRateChallenge) + '%')

logger.info(logName)
logger.info("Best Error rate: full--" + str(errorRate) + '%  common--' + str(errorRateCommon) + '%  challenge--' + str(errorRateChallenge) + '%')
