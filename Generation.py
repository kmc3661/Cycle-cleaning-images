import glob
import random
import numpy as np
from PIL import Image
import cv2
import os

path_dir = 'Data/Drone'
file_list = os.listdir(path_dir)

# #add rain+ Gaussian Blur + Darkening
for i in range(len(file_list)):
    img_name = path_dir + '/' + file_list[i]

    image = cv2.imread(img_name)
    width, height,channel = image.shape
    if (width != 1280) and (height != 1024):
        image2 = cv2.resize(image, (1280, 1024))
    M = np.ones(image2.shape, dtype = "uint8") * random.randint(10,60)
    subtracted = cv2.subtract(image2, M)

    blurImageNDArray = cv2.GaussianBlur(subtracted, (0, 0), 4)


    img_bg = Image.fromarray(blurImageNDArray)
    width, height = img_bg.size

    images = glob.glob('Raindrop/*.png')
    for j in range(5):
        for img in images:
            img = Image.open(img)
            x = random.randint(30, width-30)
            y = random.randint(20, height-20)
            width1, height1 = img.size
            img_bg.paste(img, (x, y),img)
    img_bg.save('Conv_Data/rain/rain_'+str(i+620)+'.jpg')

#Convert to small size
path_dir2='Conv_Data/temp'
file_list2 = os.listdir(path_dir2)
for k in range(len(file_list2)):
    img_name2=path_dir2+'/'+file_list2[k]
    small_img=cv2.imread(img_name2)
    small_img2=cv2.resize(small_img,(200,200))
    small_img3 = Image.fromarray(small_img2)
    small_img3.save('Conv_Data/small_temp/origin' + str(k)+'.jpg')

#add snow
# for i in range(len(file_list)):
#     img_name = path_dir + '/' + file_list[i]
#     img_bg = Image.open(img_name)
#     width, height = img_bg.size
#     images = glob.glob('Snow_patch/*.png')
#     for j in range(30):
#         for img in images:
#             img = Image.open(img)
#             x = random.randint(30, width-30)
#             y = random.randint(20, height-20)
#             width1, height1 = img.size
#             img_bg.paste(img, (x, y),img)
#     img_bg.save('Conv_Data/snow/snow_'+file_list[i])

