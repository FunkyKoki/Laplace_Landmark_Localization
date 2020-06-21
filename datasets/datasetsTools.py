import cv2
import time
import random
import numpy as np
import copy
from matplotlib import pyplot as plt
from .datasetsInfo import imageSize, datasetSize, keypointsNum, flipRelation


theta = 20.         # image rotation param
beta = 0.2          # shear transform param
sigma = 1.          # Gaussian blur param
eta = 0.4           # color jetting param
gamma = 0.3         # occlusion param


def getRawAnnosList(annoFile):
    annos = []
    with open(annoFile) as f:
        for line in f:
            annos.append(line.rstrip().split())
    print("The annotation file's length is: " + str(len(annos)))
    return annos


def showImgLandmarksInGetItem(img, pts, keypointNum, name='img', oneByOne=True, save=False):
    # 由于是在getItem函数中使用该函数，因此输入的img是float32类型的，pts亦是
    # 这里的img和pts均是数据增强以及裁剪之后的
    print("./"+name.split('/')[-1])
    
    imgRaw = copy.deepcopy(img)
    imgRaw = np.clip(cv2.cvtColor(imgRaw, cv2.COLOR_RGB2BGR) / 255.0, 0., 1.)
    imgRaw = np.clip(np.array(imgRaw * 255., dtype=np.uint8), 0, 255)
    for i in range(keypointNum):
        cv2.circle(
            imgRaw,
            (int(pts[i][0]), int(pts[i][1])),
            1,
            (0, 0, 255),
            -1
        )
        if oneByOne and not save:
            cv2.imshow(name, imgRaw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    if not oneByOne and not save:
        cv2.imshow(name, imgRaw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("./"+name.split('/')[-1], imgRaw)
        
        
def colorJetting(img, eta_in):
    eta_min = 255 * np.array([random.uniform(0, eta_in), random.uniform(0, eta_in), random.uniform(0, eta_in)])
    eta_max = 255 * np.array([random.uniform(1-eta_in, 1), random.uniform(1-eta_in, 1), random.uniform(1-eta_in, 1)])
    for ch in range(3):
        img[:, :, ch][np.where(img[:, :, ch] < eta_min[ch])] = 0.
        img[:, :, ch][np.where(img[:, :, ch] > eta_max[ch])] = 255.
        img[:, :, ch][np.where((img[:, :, ch] >= eta_min[ch]) & (img[:, :, ch] <= eta_max[ch]))] = \
            255.*(img[:, :, ch][np.where((img[:, :, ch] >= eta_min[ch]) & (img[:, :, ch] <= eta_max[ch]))] -
                  eta_min[ch])/(eta_max[ch]-eta_min[ch])
    return img


def imgOcclusion(img, gamma_in):
    x_loc = random.randint(0, imageSize - 1)
    y_loc = random.randint(0, imageSize - 1)
    w = random.randint(0, int(imageSize * gamma_in))
    h = random.randint(0, int(imageSize * gamma_in))
    up = int(y_loc-h/2) if int(y_loc-h/2) >= 0 else 0
    down = int(y_loc+h/2) if int(y_loc+h/2) <= int(imageSize-1) else int(imageSize-1)
    left = int(x_loc-w/2) if int(x_loc-w/2) >= 0 else 0
    right = int(x_loc+w/2) if int(x_loc+w/2) <= int(imageSize-1) else int(imageSize-1)
    img[up:down, left:right, :] = np.random.randint(0, int(imageSize-1), size=(down-up, right-left, 3))
    img = np.array(img, dtype=np.float32)
    return img


def randomParam(augment, width, height):
    angle, cX, cY, flip, sigma_, eta_, gamma_  = 0., 0., 0., 0, 0, 0, 0
    if augment:
        random.seed(time.time())
        theta_ = random.randint(0, 1)  # do rotation or not
#         beta_ = random.randint(0, 1)   # do shear transformation or not
#         sigma_ = random.randint(0, 1)  # do blur or not
#         eta_ = random.randint(0, 1)    # do color jetting or not
#         gamma_ = random.randint(0, 1)  # do image occlusion or not
        
        angle = random.uniform(-theta, theta) if theta_ else 0.
#         cX = random.uniform(-beta, beta) if beta_ else 0.
#         cY = random.uniform(-beta, beta) if beta_ else 0.
#         flip = random.randint(0, 1)

    shearMat = np.array([[1., cX, 0.], [cY, 1., 0.]], dtype=np.float32)

    return angle, shearMat, flip, sigma_, eta_, gamma_


def getItem(rootdir, dataset, annoline, augment=False):

    img = cv2.imread(rootdir+annoline[-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    imgWidth, imgHeight = img.shape[1], img.shape[0]
    
    # 获取原始图片的关键点坐标并转化为（kpt_num, 2）的shape
    pts = np.array(list(map(float, annoline[:2 * keypointsNum[dataset]])), dtype=np.float32)
    pts = (np.vstack((pts[:2*keypointsNum[dataset]:2], pts[1:2*keypointsNum[dataset]:2]))).transpose().reshape(-1, 2)
    ptsRaw = copy.deepcopy(pts)

    angle, shearMat, flip, sigma_, eta_, gamma_ = randomParam(augment, imgWidth, imgHeight)
    
    # 旋转
    rotMat = cv2.getRotationMatrix2D((imgWidth / 2.0 - 0.5, imgHeight / 2.0 - 0.5), angle, 1)
    img = cv2.warpAffine(img, rotMat, (imgWidth, imgHeight))
    center = np.array([imgWidth / 2.0, imgHeight / 2.0], dtype=np.float32)
    rad = angle*np.pi/180
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts = np.matmul(pts-center, rotMat) + center

    # shear transform
    img = cv2.warpAffine(img, shearMat, (imgWidth, imgHeight))
    pts = np.matmul(pts, (shearMat[:, :2]).transpose())

    # flip
    img = np.array(np.fliplr(img), dtype=np.float32) if flip else img
    pts[:, 0] = imageSize - pts[:, 0] if flip else pts[:, 0]
    pts = pts[np.array(flipRelation[dataset])[:, 1], :] if flip else pts
    pts = np.array(pts, dtype=np.float32)
    
    # texture data augment
    img = cv2.GaussianBlur(img, (5, 5), random.uniform(0, sigma)) if sigma_ else img
    img = colorJetting(img, eta) if eta_ else img
    img = imgOcclusion(img, gamma) if gamma_ else img

    imgRaw = copy.deepcopy(img)
    # if want to visualise the img and pts, here is the place
#     showImgLandmarksInGetItem(imgRaw, pts, keypointNum=keypointsNum[dataset], name=annoline[-1], oneByOne=False, save=True)

    mean, std = cv2.meanStdDev(img/255.)
    for ch in range(3):
        if std[ch, 0] < 1e-6:
            std[ch, 0] = 1.
    mean = np.array([[[mean[0, 0], mean[1, 0], mean[2, 0]]]])
    std = np.array([[[std[0, 0], std[1, 0], std[2, 0]]]])
    img = (img / 255. - mean) / std
    img = img.transpose((2, 0, 1))
    
    paramsTarget = np.concatenate((pts, np.ones(pts.shape)), 1)

    return img, paramsTarget, imgRaw, ptsRaw, (annoline[-1]).split('/')[-1]


def getUnsupervisedItem(filePath, augment=False):

    img = cv2.imread(filePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    imgWidth, imgHeight = img.shape[1], img.shape[0]

    angle, shearMat, flip, sigma_, eta_, gamma_ = randomParam(augment, imgWidth, imgHeight)
    
    # 旋转
    rotMat = cv2.getRotationMatrix2D((imgWidth / 2.0 - 0.5, imgHeight / 2.0 - 0.5), angle, 1)
    img = cv2.warpAffine(img, rotMat, (imgWidth, imgHeight))

    # shear transform
    img = cv2.warpAffine(img, shearMat, (imgWidth, imgHeight))

    # flip
    img = np.array(np.fliplr(img), dtype=np.float32) if flip else img
    
    # texture data augment
    img = cv2.GaussianBlur(img, (5, 5), random.uniform(0, sigma)) if sigma_ else img
    img = colorJetting(img, eta) if eta_ else img
    img = imgOcclusion(img, gamma) if gamma_ else img
    
    imgRaw = copy.deepcopy(img)
    imgRaw = np.clip(cv2.cvtColor(imgRaw, cv2.COLOR_RGB2BGR) / 255.0, 0., 1.)
    imgRaw = np.clip(np.array(imgRaw * 255., dtype=np.uint8), 0, 255)
    cv2.imwrite('/home/jin/Desktop/temp.png', imgRaw)

    mean, std = cv2.meanStdDev(img/255.)
    for ch in range(3):
        if std[ch, 0] < 1e-6:
            std[ch, 0] = 1.
    mean = np.array([[[mean[0, 0], mean[1, 0], mean[2, 0]]]])
    std = np.array([[[std[0, 0], std[1, 0], std[2, 0]]]])
    img = (img / 255. - mean) / std
    img = img.transpose((2, 0, 1))

    return img