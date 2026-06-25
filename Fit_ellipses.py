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
from skimage.measure import label, regionprops
from skimage.draw import ellipse_perimeter
import seaborn as sns
from scipy.signal import find_peaks, peak_widths, peak_prominences


##### globals #####
path = r'//Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
boarder_crop = 30

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
        background = cv2.GaussianBlur(im_arr, (0, 0), sigmaX = 50, sigmaY = 50) 
        im_arr = im_arr - background
        
        return im_arr

    def fft_imgs(self, image, metadata):
        im = np.fft.fft2(image)
        im = np.fft.fftshift(im)
        im = np.log1p(np.abs(im))

        return im
    
    def denoise(self, image):
        im = image
        h, w = image.shape
        hc, wc = h//2, w//2 
        im = im[hc - 150: hc + 150, wc - 150: wc + 150]
   
        return im

    def run(self):
        metadata = self.extract_metadata()
        im_arr = self.crop_DC()
        im = self.fft_imgs(im_arr, metadata)
        im = self.denoise(im)

        return metadata, im
    
class FitEllipses():
    def __init__(self, im, name, metadata):
        self.im = im
        self.name = name
        self.metadata = metadata
        
        return 

    def thresholds(self):
        flat = np.abs(np.ravel(self.im))**2    
        idx = np.argsort(flat)[::-1]  
        cumsum = np.cumsum(flat[idx])

        intensity90 = cumsum[-1] * 0.9
        count90 = np.searchsorted(cumsum, intensity90) + 1
        idx90 = idx[:count90]

        intensity50 = cumsum[-1] * 0.1     
        count50 = np.searchsorted(cumsum, intensity50) + 1
        idx50 = idx[:count50]

        return idx90, idx50, flat
         
    def highlight50(self, idx50):
        result = cv2.cvtColor(self.im, cv2.COLOR_GRAY2BGR)
        rows, cols = np.unravel_index(idx50, self.im.shape[:2])
        result[rows, cols] = [255, 0, 255]
        cv2.imwrite(path + outputs + '/' + file + '/' + self.name + '_50.png', result)

        return
    
    def ellipse50(self, idx50):
        img = cv2.normalize(self.im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = np.zeros(img.shape[:2], dtype=bool)
        rows, cols = np.unravel_index(idx50, img.shape[:2])
        mask[rows, cols] = True
        labeled = label(mask)
        regions = regionprops(labeled)
        r = max(regions, key=lambda r: r.area)

        y0, x0 = r.centroid
        a = r.axis_major_length / 2
        b = r.axis_minor_length / 2
        orientation = r.orientation  

        rr, cc = ellipse_perimeter(int(y0), int(x0), int(a), int(b), orientation=orientation)
        rr = np.clip(rr, 0, img.shape[0] - 1)
        cc = np.clip(cc, 0, img.shape[1] - 1)
        img[rr, cc] = (255, 255, 0)
        cv2.imwrite(path + outputs + '/' + file + '/' + self.name + '_ellipse50.png', img)

        lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        ellipse_dict = {'Major Axis': r.axis_major_length, 'Minor Axis': r.axis_minor_length, 
                        'Orientation (rad)': r.orientation, 'Laplacian Variance': lap_var}

        return ellipse_dict
    
    def plot_frequency(self):
        cy, cx = self.im.shape[0] // 2, self.im.shape[1] // 2
        y, x = np.indices(self.im.shape)
        radius = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        amp = np.bincount(radius.ravel(), self.im.ravel())
        counts = np.bincount(radius.ravel())
        amp = amp /counts
        freq = np.arange(radius.max() + 1)
        peaks, _ = find_peaks(amp, prominence = 1.5, distance = 5) 

        amp = amp/max(amp)
        widths, width_heights, left_ips, right_ips = peak_widths(amp, peaks, rel_height=0.5)

        return amp, freq, peaks, widths
    
    def plot_masked_frequency(self, idx50):
        cy, cx = self.im.shape[0] // 2, self.im.shape[1] // 2
        y, x = np.indices(self.im.shape)
        radius = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

        mask = np.zeros(self.im.shape, dtype=bool)
        rows, cols = np.unravel_index(idx50, self.im.shape)
        mask[rows, cols] = True

        amp = np.bincount(radius[mask], weights=self.im[mask])
        counts = np.bincount(radius[mask])
        amp_mask = amp / counts
        freq_mask = np.arange(len(amp))

        peaks_mask, _ = find_peaks(amp, prominence = 1.5, distance = 5)

        amp_mask = amp_mask/max(amp_mask)

        return amp_mask, freq_mask, peaks_mask

    def run(self):
        idx90, idx50, flat = self.thresholds()
        self.highlight50(idx50)
        ellipse_dict = self.ellipse50(idx50)
        amp, freqs, peaks, widths = self.plot_frequency()
        amp_mask, freq_mask, peaks_mask = self.plot_masked_frequency(idx50)
        
        return idx90, idx50, ellipse_dict, amp, freqs, peaks, amp_mask, freq_mask, peaks_mask, widths
    
##### call functions #####
meta = []
names = []
ellipses = []
for file in os.listdir(path + inputs):
    os.makedirs(path + outputs + '/' + file, exist_ok=True)
    
    if '.DS_Store' in file:
        continue

    count90 = []
    count50 = []
    names2 = []
    profiles = []
    profiles_mask = []
    fig, ax = plt.subplots(figsize = (6, 6))

    for image in tqdm(os.listdir(path + inputs + '/' + file)):
        if not image.endswith('.tif'):
            continue

        name = image.removesuffix('.tif')
        names.append(name)
    
        prep = PrepareImages(path + inputs + '/' + file + '/' + image)
        metadata, im = prep.run()

        ellipse = FitEllipses(im, name, metadata)
        idx90, idx50, ellipse_dict, amp, freqs, peaks, amp_mask, freq_mask, peaks_mask, widths = ellipse.run()
        
        metadata = pd.DataFrame([metadata])
        
        metadata['Abs Offset'] = pd.to_numeric(metadata['Image'].str.split('__').str[1]).abs()
        
        metadata['Offset'] = metadata['Image'].str.split('__').str[1]
        offset =  int(metadata['Offset'].iloc[0].removesuffix('.0'))
        metadata['Offset'] = offset
        
        metadata['Peak Width'] = widths
        ellipse_df = pd.DataFrame([ellipse_dict])
        ellipse_df['File'] = file

        all_dat = metadata.join(ellipse_df, how = 'inner')
        meta.append(all_dat)

        count90.append(len(idx90))
        count50.append(len(idx50))
        names2.append(name)
        
        profiles.append({'Image': name, 'Offset': offset, 'Frequency': freqs, 'Amplitude': amp,
                         'File': file, 'Current': metadata['Current'].iloc[0]})
        profiles_mask.append({'Image': name, 'Offset': offset, 'Frequency': freq_mask, 
                              'Amplitude': amp_mask, 'File': file, 'Current': metadata['Current'].iloc[0]})
        
        # plt.imsave(path + outputs + '/' + file + '/' + name + '.png', im, cmap = 'gray')
        # plt.close()
  
    count_df = pd.DataFrame({'Image': names2, 'Pixels 90': count90, 'Pixels 50': count50})
    count_df.to_csv(path + outputs + '/' + file + '/' + file + '.csv')

    all_profiles = pd.DataFrame(profiles)
    all_profiles = all_profiles.explode(['Frequency', 'Amplitude'], ignore_index = True)
    all_profiles['Frequency'] = pd.to_numeric(all_profiles['Frequency'])
    all_profiles['Amplitude'] = pd.to_numeric(all_profiles['Amplitude'])
    p1 = sns.relplot(kind = 'line', data = all_profiles, x = 'Frequency', y = 'Amplitude', col = 'Current', hue = 'Offset',
                palette = 'tab20')
    p1.set_titles('')
    p1.set_xlabels('')
    plt.xlabel('Frequency')
    plt.suptitle('Beam Profile from Whole FFT Intensity', x = 0.5, ha = 'center')
    plt.tight_layout()
    plt.savefig(path + outputs + '/' + file + '_profile50.png')

    # all_profiles_mask = pd.DataFrame(profiles_mask)
    # all_profiles_mask = all_profiles_mask.explode(['Frequency', 'Amplitude'], ignore_index = True)
    # all_profiles_mask['Frequency'] = pd.to_numeric(all_profiles_mask['Frequency'])
    # all_profiles_mask['Amplitude'] = pd.to_numeric(all_profiles_mask['Amplitude'])
    # p2 = sns.relplot(kind = 'line', data = all_profiles_mask, x = 'Frequency', y = 'Amplitude', col = 'Current', hue = 'Offset',
    #              palette = 'tab20')
    # p2.set_titles('')
    # plt.suptitle('Beam Profile from Cropped FFT Intensity', ha = 'center')
    # plt.tight_layout()
    # plt.savefig(path + outputs + '/' + file + '_profile50_mask.png')

all_metadata = pd.concat(meta, ignore_index=True)
all_metadata.to_csv(path + outputs + '/all_metadata.csv')


data = pd.read_csv(path + outputs + '/all_metadata.csv')
offset0 = data[data['Offset'] == 0]

mean_major_axis = offset0.groupby('Current')['Major Axis'].mean()
data['Major Axis Focus Mean'] = data['Current'].map(mean_major_axis)
data['Major Axis Delta'] = data['Major Axis Focus Mean'] - data['Major Axis']

# data = data[data['Offset'] != 0]

fig, ax = plt.subplots(figsize=(6, 6))
g = sns.relplot(data = data, x = 'Offset', y = 'Peak Width', col = 'Current', hue = 'File', palette = 'tab20')
g.set_titles('')
g.set_xlabels('')
plt.xlabel('Absolute Offeset', x = 0.5, ha = 'center')
plt.suptitle('Core Intensity Width by Absolute Offset')
plt.legend(bbox_to_anchor = (0.9, 1.1))
plt.savefig(path + outputs + '/Peak_Intensity_Width_by_Offset.png')

 
