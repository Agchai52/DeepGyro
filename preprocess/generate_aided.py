import numpy as np

# Load camera and IMU calibration information
import calibration as calib
import os
import IO    # For reading and writing data
from utils import *
from PIL import Image
from visualize import plotBlurVectors
from computeR import IMUComputeH

def save_blurField(img, outpath, folder, imgName):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(outpath + '/' + folder + '/' + imgName)

if __name__ == '__main__':

    inpath, outpath = IO.parseInputs()
    print("Input folder: %s" %inpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath + '/blurred')
        os.makedirs(outpath + '/blurx')
        os.makedirs(outpath + '/blury')
        #os.makedirs(outpath + '/visualization/')

    ''' Generate blur field for each image '''

    f_test = open("./dataset/AidedDeblur/test_instance_names.txt", "r")
    imgsName = f_test.readlines()
    imgsName = [line.rstrip() for line in imgsName]
    f_test.close()
    imgsName = sorted(imgsName)

    for i, imgName in enumerate(imgsName):
        print("Processing image: {}".format(imgName))
        img = Image.open(imgName + '_blur_err.png').convert('RGB')
        img = np.array(img)
        R_computer = IMUComputeH(imgName)

        R = R_computer.compute_rotations()
        R = np.array(R)  # rotation matrix (index, 3, 3)

        K = R_computer.intrinsicMat  # intrinsic matrix
        time_stamp = R_computer.time_stamp  # number of poses
        height = R_computer.image_H
        width = R_computer.image_W
        tr = R_computer.read_out
        te = R_computer.exposure

        Bx, By = computeBlurfield_aided(R, K, time_stamp, te, tr, height, width)
        save_blurField(img, outpath, 'blurred/', imgName[-6:]+'_blur_err.png')
        save_blurField(Bx, outpath, 'blurx/', imgName[-6:]+'_blur_err.png')
        save_blurField(By, outpath, 'blury/', imgName[-6:]+'_blur_err.png')
        
        #plotBlurVectors(Bx, By, img, outpath, idx=i) # Optional

# python generate_aided.py -i ./myrawdata -o ./input