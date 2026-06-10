##### imports #####
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import os
from tqdm import tqdm 
from scipy import signal
from PIL import Image
import configparser
from io import StringIO
from pathlib import Path


##### globals #####
path = r'//Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
boarder_crop = 50

os.makedirs(path + outputs, exist_ok=True)

##### define functions #####
class PrepareImages():
    def __init__(self, img):
        self.img = img
        self.im = Image.open(self.img)

    def extract_metadata(self):
        metadata_text = self.im.tag_v2.get(34682)
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read_file(StringIO(metadata_text))

        metadata = {'HFW': config['IBeam']['HFW'], 'ResolutionX': config['Image']['ResolutionX'],
            'ResolutionY': config['Image']['ResolutionY'], 'DwellTime': config['Scan']['Dwelltime'],
            'Voltage': config['IBeam']['HV'], 'Current': config['IBeam']['BeamCurrent']}

        return metadata
    
    def crop_DC(self):
        im_arr = np.array(self.im)
        im_arr  = im_arr[boarder_crop:-boarder_crop, boarder_crop:-boarder_crop]
        
        return im_arr

    def fft_imgs(self, image):
        im = np.fft.fft2(image)
        im = np.fft.fftshift(im)
        im = np.log1p(np.abs(im))

        return im
    
    def denoise(self, image):
        background = cv2.GaussianBlur(image, (0, 0), sigmaX=40, sigmaY=40)
        im = image - background
        im = im[150: -150, 150:-150]

        return im

    def run(self):
        metadata = self.extract_metadata()
        im_arr = self.crop_DC()
        im = self.fft_imgs(im_arr)
        im = self.denoise(im)

        return metadata, im


for file in os.listdir(path + inputs):
    if '.DS_Store' in file:
        continue
    for image in tqdm(os.listdir(path + inputs + '/' + file)):
        if not image.endswith('.tif'):
            continue
        prep = PrepareImages(path + inputs + '/' + file + '/' + image)
        metadata, im = prep.run()

        name = image.removesuffix('.tif')
        plt.imsave(path + outputs + '/' + name + '.png', im, cmap = 'gray')
        plt.close()


