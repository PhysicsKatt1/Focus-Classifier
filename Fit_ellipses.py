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
import pandas as pd
import seaborn as sns
from skimage.measure import label, regionprops
import matplotlib.cm as cm


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

        im_name = self.img.removesuffix('.tif')
        im_name = im_name.split('data_')

        metadata = {'Image': im_name[1], 'HFW': config['IBeam']['HFW'], 'ResolutionX': config['Image']['ResolutionX'],
            'ResolutionY': config['Image']['ResolutionY'], 'DwellTime': config['Scan']['Dwelltime'],
            'Voltage': config['IBeam']['HV'], 'Current': config['IBeam']['BeamCurrent']}

        return metadata
    
    def crop_DC(self):
        im_arr = cv2.normalize(np.array(self.im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im_arr  = im_arr[boarder_crop:-boarder_crop, boarder_crop:-boarder_crop]
        background = cv2.GaussianBlur(im_arr, (0, 0), sigmaX=10, sigmaY=10)
        im_arr = im_arr - background
        
        return im_arr

    def fft_imgs(self, image):
        im = np.fft.fft2(image)
        im = np.fft.fftshift(im)
        im = np.log1p(np.abs(im))

        return im
    
    def denoise(self, image):
        im = image
        h, w = image.shape
        hc, wc = h//2, w//2 
        im = im[hc - 130: hc + 130, wc - 130: wc + 130]
   
        return im

    def run(self):
        metadata = self.extract_metadata()
        im_arr = self.crop_DC()
        im = self.fft_imgs(im_arr)
        im = self.denoise(im)

        return metadata, im

class FitEllipses():
    def __init__(self, im, name):
        self.im = im
        self.name = name
        
        return 

    def thresholds(self):
        flat = np.abs(np.ravel(self.im))**2    
        idx = np.argsort(flat)[::-1]  
        cumsum = np.cumsum(flat[idx])

        intensity90 = cumsum[-1] * 0.9
        count90 = np.searchsorted(cumsum, intensity90) + 1
        idx90 = idx[:count90]

        intensity50 = cumsum[-1] * 0.05
        count50 = np.searchsorted(cumsum, intensity50) + 1
        idx50 = idx[:count50]

        return idx90, idx50, flat
         
    def highlight50(self, idx50):
        result = cv2.cvtColor(self.im, cv2.COLOR_GRAY2BGR)
        rows, cols = np.unravel_index(idx50, self.im.shape[:2])
        result[rows, cols] = [0, 0, 255]
        cv2.imwrite(path + outputs + '/' + file + '/' + self.name + '_50.png', result)

        return 
    
    def run(self):
        idx90, idx50, flat = self.thresholds()
        self.highlight50(idx50)
        
        return idx90, idx50
    
##### call functions #####
meta = []
names = []
for file in os.listdir(path + inputs):
    os.makedirs(path + outputs + '/' + file, exist_ok=True)
    
    if '.DS_Store' in file:
        continue

    count90 = []
    count50 = []
    names2 = []
    for image in tqdm(os.listdir(path + inputs + '/' + file)):
        if not image.endswith('.tif'):
            continue

        name = image.removesuffix('.tif')
        names.append(name)
    
        prep = PrepareImages(path + inputs + '/' + file + '/' + image)
        metadata, im = prep.run()

        ellipse = FitEllipses(im, name)
        idx90, idx50 = ellipse.run()
        
        metadata = pd.DataFrame([metadata])
        meta.append(metadata)

        count90.append(len(idx90))
        count50.append(len(idx50))
        names2.append(name)
        
        plt.imsave(path + outputs + '/' + file + '/' + name + '.png', im, cmap = 'gray')
        plt.close()
    
    count_df = pd.DataFrame({'Image': names2, 'Pixels 90': count90, 'Pixels 50': count50})
    count_df.to_csv(path + outputs + '/' + file + '/' + name + '.csv')


all_metadata = pd.concat(meta, ignore_index=True)
all_metadata.to_csv(path + outputs + '/all_metadata.csv')


 

