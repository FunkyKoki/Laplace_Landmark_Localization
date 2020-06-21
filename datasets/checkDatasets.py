import cv2
import pandas as pd

# dataframe = pd.read_csv("/media/WDC/datasets/data/300w/face_landmarks_300w_valid.csv")

# # with open('./300W_full_annos.txt', 'w') as f:
# for j in range(689):
#     line = dataframe.iloc[j]
#     pic = cv2.imread("/media/WDC/datasets/300W/"+line[0])
#     for i in range(68):
#         cv2.circle(pic, (int(float(line[2*i+4])), int(float(line[2*i+5]))), 5, (0, 0, 255))
#         cv2.imshow("ok", pic)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         #     f.write(str(line[2*i+4]) + " " + str(line[2*i+5]) + " ")
#         # f.write("/media/WDC/datasets/300W/"+line[0]+"\n")

# with open("/media/WDC/datasets/300W80/300W80_challenge.txt")  as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip().split()
#         pic = cv2.imread("/media/WDC/datasets/300W80/"+line[-1])
#         for i in range(68):
#             cv2.circle(pic, (int(float(line[2*i])), int(float(line[2*i+1]))), 1, (0, 0, 255))
#         cv2.imshow("ok", pic)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


