import cv2
import numpy as np

import os
import tqdm


def createDataset(dataset='300W', split='challenge', size=80, landmark=68):
    if not os.path.exists('/media/WDC/datasets/'+dataset+str(size)):
        os.mkdir('/media/WDC/datasets/'+dataset+str(size))
    if not os.path.exists('/media/WDC/datasets/'+dataset+str(size)+'/'+split):
        os.mkdir('/media/WDC/datasets/'+dataset+str(size)+'/'+split)
    with open('./'+dataset+'_'+split+'_annos.txt') as f, open('/media/WDC/datasets/'+dataset+str(size)+'/'+dataset+str(size)+'_'+split+'.txt', 'w') as newAnnoFile:
        annoLines = f.readlines()
        annoLines = [line.rstrip().split() for line in annoLines]
        for idx, line in tqdm.tqdm(enumerate(annoLines)):
            img = cv2.imread(line[-1])
            pts = [float(num) for num in line[:landmark*2]]
            imgRawHeight, imgRawWidth = img.shape[0], img.shape[1]

            # 该段代码根据关键点生成tight bounding box，在tight bbox基础上，宽高均外扩16%的长度
            # 在外扩的基础上，再根据外扩的bbox的长边，将外扩bbox的短边继续外扩，使得最终的bbox为正方形
            # 最后保证bbox边长为偶数，便于后面的处理
            # bbox含有4个元素，第一个是边界框左上角点x坐标，第二个是左上角点y坐标，第三个是右下角点x坐标，第四个是右下角点y坐标

            pts5 = np.zeros(10)
#             pts5[0] = np.mean(pts[72:84:2])  # left pupil
#             pts5[1] = np.mean(pts[73:84:2])
#             pts5[2] = np.mean(pts[84:96:2])  # right pupil
#             pts5[3] = np.mean(pts[85:96:2])
            pts5[0] = pts[72]  # left ocular
            pts5[1] = pts[73]
            pts5[2] = pts[90]  # right ocular
            pts5[3] = pts[91]
            pts5[4] = pts[60]  # nose mid
            pts5[5] = pts[61]
            pts5[6] = pts[96]  # left mouth corner
            pts5[7] = pts[97]
            pts5[8] = pts[108]  # right mouth corner
            pts5[9] = pts[109]

            bbox = [min(list(map(int, pts5[::2]))), min(list(map(int, pts5[1::2]))), max(list(map(int, pts5[::2]))), max(list(map(int, pts5[1::2])))]
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            bbox = [int(bbox[0]-0.05*width), int(bbox[1]-0.05*height), int(bbox[2]+0.05*width), int(bbox[3]+0.05*height)]
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if width > height:
                longSize, shortSize = max(width, height), min(width, height)
                diff = longSize - shortSize
                correctDistance = diff//2
                bbox[1] = bbox[1] - correctDistance
                bbox[3] = bbox[3] + correctDistance + diff%2
            else:
                longSize, shortSize = max(width, height), min(width, height)
                diff = longSize - shortSize
                correctDistance = diff//2
                bbox[0] = bbox[0] - correctDistance
                bbox[2] = bbox[2] + correctDistance + diff%2
            assert bbox[2] - bbox[0] == bbox[3] - bbox[1]
            if (bbox[2] - bbox[0] + 1)%2 == 1:
                bbox[2] += 1
                bbox[3] += 1
            bbox = np.array(bbox)

            # 将bbox外扩200%，得到一个大范围内的人脸图像，resize为size*size后保存该图像作为数据集，便于之后在该图像上进行数据增强并抠出halfSize*halfSize大小的区域
            bboxSize = bbox[2]-bbox[0]+1
            assert bboxSize%2 == 0
            bboxSize2x = 3*bboxSize
            imgNew = np.zeros((bboxSize2x, bboxSize2x, 3))
            bboxNew, bboxLimited = np.zeros(bbox.shape), np.zeros(bbox.shape)
            bboxNew[0], bboxNew[1], bboxNew[2], bboxNew[3] = bbox[0]-bboxSize, bbox[1]-bboxSize, bbox[2]+bboxSize, bbox[3]+bboxSize
            bboxLimited[0] = bboxNew[0] if bboxNew[0] >= 0 else 0
            bboxLimited[1] = bboxNew[1] if bboxNew[1] >= 0 else 0
            bboxLimited[2] = bboxNew[2] if bboxNew[2] < imgRawWidth else imgRawWidth - 1
            bboxLimited[3] = bboxNew[3] if bboxNew[3] < imgRawHeight else imgRawHeight - 1
            bboxLimited = np.array(bboxLimited, dtype=np.int32)

            # 从原始图像中提取处需要的部分，置入新的图像空间中
            imgNew[int(bboxLimited[1]-bboxNew[1]):int(bboxLimited[3]-bboxLimited[1]+bboxLimited[1]-bboxNew[1]+1), int(bboxLimited[0]-bboxNew[0]):int(bboxLimited[2]-bboxLimited[0]+bboxLimited[0]-bboxNew[0]+1), :] = \
                img[int(bboxLimited[1]):int(bboxLimited[3]+1), int(bboxLimited[0]):int(bboxLimited[2]+1), :]
            imgNew = np.array(imgNew, dtype=np.uint8)
            ptsNew = np.reshape(np.transpose(np.vstack((np.array(pts[::2]) - bboxNew[0], np.array(pts[1::2]) - bboxNew[1]))), (-1,))

            # resize图像为size*size的大小，并相应得对关键点坐标进行变换
            imgNew = cv2.resize(imgNew, (size, size), interpolation=cv2.INTER_LINEAR)
            ptsNew = ptsNew*size/bboxSize2x

            # 生成的时候可进行可视化，查验数据的正确性
            # for i in range(len(ptsNew)//2):
            #     cv2.circle(imgNew, (int(ptsNew[2*i]), int(ptsNew[2*i+1])), 1, (0,0,255), -1)
            # cv2.imshow('img', imgNew)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # ptsNew = np.hstack((ptsNew[::2], ptsNew[1::2]))  # x1, x2, x3, ..., y1, y2, y3, ...
            cv2.imwrite('/media/WDC/datasets/'+dataset+str(size)+'/'+split+'/'+dataset+str(idx).zfill(4)+'.png', imgNew)
            for i in range(len(ptsNew)):  # x1, y1, x2, y2, ..., xn, yn
                newAnnoFile.write(str(ptsNew[i]) + ' ')
                if 0 <= ptsNew[i] <= size:
                    pass
                else:
                    print(idx, line[-1], ptsNew[i])
            newAnnoFile.write(split+'/'+dataset+str(idx).zfill(4)+'.png\n')

if __name__ == '__main__':
    createDataset()
