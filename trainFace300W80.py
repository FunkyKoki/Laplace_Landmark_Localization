import os
import time
import math
import tqdm
import torch
import logging
from torch.autograd import Variable

from models import Generator, loadWeights, dataPrefetcher, GeneratorGhostConv
from datasets import Face300W80, datasetSize

from testFace300W80 import testFace300W80

# 生成必要的文件夹
if not os.path.exists('/media/WDC/savedModels/LaplaceLandmarkLocalization'):
    os.mkdir('/media/WDC/savedModels/LaplaceLandmarkLocalization')
if not os.path.exists('./logs'):
    os.mkdir('./logs')

logName = '20200525_face300W80_Train_4'
# 生成记录训练过程的loggor
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('./logs/' + logName + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(logName)
logger.info('备注：由之前的实验可知，ghostRatio设置为2是最佳的。由于GhostModule中的depthwise卷积本身具有一定的正则化作用，因此需要考虑调整dropout的dropoutRate，本次设为0.00。 1.这一次实验对于loss损失同样进行了分母增加了极小量的修正；2.其余超参数均按照之前实验得到的最佳参数进行设定。')
rat = 1
tempera = 2.0

torch.backends.cudnn.benchmark = True

logger.info('Network: GeneratorGhostConv(ratio='+str(rat)+', keyPoints=68, dropOutRate=0.00, temperature='+str(tempera)+', imageSize=80)')
logger.info('GPU: \'cuda:0\'')
model = GeneratorGhostConv(ratio=rat, keyPoints=68, dropOutRate=0.00, temperature=tempera, imageSize=80)
# model = loadWeights(model, '/media/WDC/savedModels/LaplaceLandmarkLocalization/20200512_face300W80_Train_4_model_best.pth', 'cuda:0')
model = model.cuda('cuda:0')
model.train()

logger.info('Adam：lr=0.001, weight_decay=1e-5')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

logger.info('Training epoch: 750')
# logger.info('Lr schedule: cosine, warm epoch: 15, epoch size: 998 iterations, max lr: 1e-3, min lr: 1e-7')
logger.info('Dataset: Face300W80(split="train", augment=False)')
logger.info('Dataloader: batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True')
trainSet = Face300W80(split="train", augment=True)
maxEpoch = 750
batchSize = 16
numWorkers = 8
itersPerBatch = datasetSize["300W80"]["train"]//batchSize

# warmEpoch = 15
# epochSize = itersPerBatch
# lrMax = 1e-3
# lrMin = 1e-7
# iteration = 0

errorRate=100.0

for epoch in range(0, maxEpoch):
    print("Epoch: " + str(epoch))
    st = time.time()
    lossRecordPerEpoch = []
    
    if epoch in [500]:
        optimizer.param_groups[0]['lr'] *= 0.1

    dataLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)
    prefetcher = dataPrefetcher(iter(dataLoader))

    for _ in tqdm.tqdm(range(itersPerBatch)):

        img, paramsTarget = prefetcher.next()
        assert img.size()[0] == paramsTarget.size()[0] == batchSize

        img = Variable(img.cuda('cuda:0'))
        paramsTarget = Variable(paramsTarget.cuda('cuda:0'))

#         if iteration <= epochSize*warmEpoch:
#             optimizer.param_groups[0]['lr'] = lrMin + (lrMax - lrMin)*iteration/(epochSize*warmEpoch)
#         else:
#             t1 = iteration - warmEpoch * epochSize
#             t2 = (maxEpoch - warmEpoch) * epochSize
#             t = t1 * math.pi / t2
#             optimizer.param_groups[0]['lr'] = lrMin + (lrMax - lrMin) * 0.5 * (1 + math.cos(t))

        optimizer.zero_grad()
        out = model(img)
        loss = model.calculateLoss(out, paramsTarget)
        loss.backward()
        optimizer.step()

        lossRecordPerEpoch.append(loss.item())

#         iteration += 1

    epochTime = time.time() - st
    logger.info('epoch: ' + str(epoch) + ' lr: ' + str(optimizer.param_groups[0]['lr']) + ' loss: ' + str(sum(lossRecordPerEpoch)/len(lossRecordPerEpoch)) + ' time: ' + str(epochTime))

    torch.save(model.state_dict(), '/media/WDC/savedModels/LaplaceLandmarkLocalization/' + logName + '_model_temp.pth')
    errorRateTemp = testFace300W80('full', 'temp', logName, rat=rat, tempera=tempera)
    if errorRateTemp < errorRate:
        errorRate = errorRateTemp
        torch.save(model.state_dict(), '/media/WDC/savedModels/LaplaceLandmarkLocalization/' + logName + '_model_best.pth')
        errorRateCommon = testFace300W80('common', 'best', logName, rat=rat, tempera=tempera)
        errorRateChallenge = testFace300W80('challenge', 'best', logName, rat=rat, tempera=tempera)
        logger.info('Performance becomes better in epoch: ' + str(epoch))
        logger.info('Now the error rate is: ' + str(errorRate) + '%  common--' + str(errorRateCommon) + '%  challenge--' + str(errorRateChallenge) + '%')

logger.info(logName)
logger.info("Best Error rate: full--" + str(errorRate) + '%  common--' + str(errorRateCommon) + '%  challenge--' + str(errorRateChallenge) + '%')
