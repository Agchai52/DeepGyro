import numpy as np
import math
import cv2
import os
from skimage.measure import compare_ssim as ssim


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


deblu_root = './output'  #
sharp_root = './dataset/AidedDeblur/test/'  # _

f_test = open("./dataset/AidedDeblur/test_instance_names.txt", "r")
test_data = f_test.readlines()
sharp_list = [line.rstrip() for line in test_data]
f_test.close()
deblu_list = os.listdir(deblu_root)
sharp_list = sorted(sharp_list, key=str.lower)
PSNR_all = []
SSIM_all = []

for item in sharp_list:
    if True:

        name_sharp = item[-6:]
        name_deblu = name_sharp + '_blur_err.png'

        path_deblu = os.path.join(deblu_root, name_deblu)
        path_sharp = item + '_ref.png'

        img_deblu = cv2.imread(path_deblu, cv2.IMREAD_COLOR).astype(np.float)
        img_sharp = cv2.imread(path_sharp, cv2.IMREAD_COLOR).astype(np.float)

        #print(img_deblu.shape)
        #img_deblu = cv2.resize(img_deblu, (1280, 720))

        #print(img_deblu.shape)
        # cv2.imshow('sharp', img_sharp/255)
        # cv2.imshow('deblur', img_deblu/255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        psnr_n = psnr(img_deblu, img_sharp)
        ssim_n = ssim(img_deblu / 255, img_sharp / 255, gaussian_weights=True, multichannel=True,
                      use_sample_covariance=False)

        if item[-3:] == '001':
            print("Test Image {}, PSNR = {}, SSIM = {}".format(name_sharp, psnr_n, ssim_n))
        PSNR_all.append(psnr_n)
        SSIM_all.append(ssim_n)

PSNR = np.mean(PSNR_all)
SSIM = np.mean(SSIM_all)
# print(PSNR_all)
# print(SSIM_all)
print(PSNR)
print(SSIM)