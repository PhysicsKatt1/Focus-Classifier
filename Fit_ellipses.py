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
from scipy.stats import linregress


##### globals #####
path = r'//Users/trentstarkey/Desktop'
inputs = '/Offsets_all'
outputs = '/Offsets_FFTs'
test = '/Offsets_Test'
boarder_crop = 3
sigX = 28
sigY = 28
av_mean = 1e-05
sample = 'All'
reg = '/all_metadata_' + sample + '.csv'
error_bins = [0, 10, 20, 30, 40, 50]
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
        background = cv2.GaussianBlur(im_arr, (0, 0), sigmaX = sigX, sigmaY = sigY)
        im_arr = (im_arr - background)
        im_arr = im_arr / np.max(im_arr)
        # im_arr = im_arr - (np.mean(im_arr) - av_mean)
        
        return im_arr

    def fft_imgs(self, image):
        im = np.fft.fft2(image)
        im = np.fft.fftshift(im)
        im = np.log1p(np.abs(im))
        im = im - (np.mean(im) - av_mean)

        return im
    
    def denoise(self, image):
        h, w = image.shape
        hc, wc = h//2, w//2 
        im = image[hc - 128: hc + 128, wc - 128: wc + 128] 
   
        return im

    def run(self):
        metadata = self.extract_metadata()
        im_arr = self.crop_DC()
        im = self.fft_imgs(im_arr)
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
        
        peaks, _ = find_peaks(amp, prominence = 0.05, distance = 5)
        amp = amp/max(amp)

        widths, width_heights, left_ips, right_ips = peak_widths(amp, peaks, rel_height=0.5)

        return amp, freq, peaks, widths
    

    def run(self):
        idx90, idx50, flat = self.thresholds()
        self.highlight50(idx50)
        ellipse_dict = self.ellipse50(idx50)
        amp, freqs, peaks, widths = self.plot_frequency()
        
        return idx90, idx50, ellipse_dict, amp, freqs, peaks, widths

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
    fig, ax = plt.subplots(figsize = (6, 6))

    in_focus_fft = {}
    image_list = [i for i in os.listdir(path + inputs + '/' + file)
              if i.endswith('.tif')]

    for image in image_list:
        prep = PrepareImages(path + inputs + '/' + file + '/' + image)
        metadata, im = prep.run()

        if '0.0__0.0__0.0' in image:
            metadata['Current'] = float(metadata['Current'])
            metadata['Current'] = np.round(metadata['Current'] / 1e-11, 0) * 1e-11
            in_focus_fft[metadata['Current']] = im.copy()

    for image in tqdm(image_list):
        name = image.removesuffix('.tif')
        names.append(name)
    
        prep = PrepareImages(path + inputs + '/' + file + '/' + image)
        metadata, im = prep.run()

        metadata['Current'] = float(metadata['Current'])
        metadata['Current'] = np.round(metadata['Current'] / 1e-11, 0) * 1e-11
        im = im - 0.5*in_focus_fft[metadata['Current']]

        ellipse = FitEllipses(im, name, metadata)
        idx90, idx50, ellipse_dict, amp, freqs, peaks, widths = ellipse.run()
  
        metadata = pd.DataFrame([metadata])
        
        metadata['Abs Offset'] = pd.to_numeric(metadata['Image'].str.split('__').str[1]).abs()
        
        metadata['Offset'] = metadata['Image'].str.split('__').str[1]
        offset =  int(metadata['Offset'].iloc[0].removesuffix('.0'))
        metadata['Offset'] = offset
        
        metadata['Peak Width'] = widths[0] if len(widths) else np.nan
        ellipse_df = pd.DataFrame([ellipse_dict])
        ellipse_df['File'] = file

        all_dat = metadata.join(ellipse_df, how = 'inner')
        meta.append(all_dat)

        count90.append(len(idx90))
        count50.append(len(idx50))
        names2.append(name)
        
        profiles.append({'Image': name, 'Offset': offset, 'Frequency': freqs, 'Amplitude': amp,
                         'File': file, 'Current': metadata['Current'].iloc[0]})
        
        plt.imsave(path + outputs + '/' + file + '/' + name + '.png', im, cmap = 'gray')
        plt.close()
  
    count_df = pd.DataFrame({'Image': names2, 'Pixels 90': count90, 'Pixels 50': count50})
    count_df.to_csv(path + outputs + '/' + file + '/' + file + '.csv')


    all_profiles = pd.DataFrame(profiles)
    all_profiles = all_profiles.explode(['Frequency', 'Amplitude'], ignore_index = True)
    all_profiles['Frequency'] = pd.to_numeric(all_profiles['Frequency'])
    all_profiles['Amplitude'] = pd.to_numeric(all_profiles['Amplitude'])
    all_profiles['File'] = pd.to_numeric(all_profiles['File'])
    
    offsets_unique = sorted(all_profiles['Offset'].unique())
    cmap = plt.get_cmap('cool')
    colors = cmap(np.linspace(0, 1, len(offsets_unique)))
    palette = dict(zip(offsets_unique, colors))
    
    fig, ax = plt.subplots(figsize=(8, 10))
    p1 = sns.relplot(kind = 'line', data = all_profiles, x = 'Frequency', y = 'Amplitude', col = 'Current', hue = 'Offset',
                palette = palette)
    p1.set_xlabels('')
    p1.figure.subplots_adjust(top=0.84)
    p1.figure.text(0.5, 0.015, 'Frequency', ha = 'center', va = 'center')
    plt.suptitle('Beam Profiles from Whole FFT Intensity \n Sample: ' + sample, x = 0.5, y = 0.99, ha = 'center')
    plt.savefig(path + outputs + '/' + file + '_profileFWHM_' + sample + '.png', bbox_inches = 'tight')
    plt.close()

##### predict defocus offset from peak width #####
all_metadata = pd.concat(meta, ignore_index=True)
all_metadata.to_csv(path + outputs + '/all_metadata_' + sample + '.csv')

data = pd.read_csv(path + outputs + '/all_metadata_' + sample + '.csv')
offset0 = data[data['Offset'] == 0]

mean_major_axis = offset0.groupby('Current')['Major Axis'].mean()
data['Major Axis Focus Mean'] = data['Current'].map(mean_major_axis)
data['Major Axis Delta'] = data['Major Axis Focus Mean'] - data['Major Axis']

# data = data[data['Offset'] != 0]

fig, ax = plt.subplots(figsize=(8, 10))
g = sns.relplot(data = data, x = 'Offset', y = 'Peak Width', col = 'Current', hue = 'File', palette = 'cool')
g.set_xlabels('')
for ax in g.axes.flat:
    ax.set_xticks([-50, -30, -10, 10, 30, 50])
g.figure.subplots_adjust(top=0.84)
g.figure.text(0.5, 0.015, 'Offset', ha = 'center', va = 'center')
plt.suptitle('Prominent Peak Width by Offset \n Sample: ' + sample)
plt.savefig(path + outputs + '/Peak_Intensity_Width_by_Offset_' + sample + '.png', bbox_inches = 'tight')

##### predict offset from peak width #####
data = pd.read_csv(path + outputs + reg)
data = data[data['Offset'] >= 0]

zero_map = (data.loc[data['Offset'] == 0].set_index(['File', 'Current'])['Peak Width'])
data['Zero Offset Peak Width'] = data.set_index(['File', 'Current']).index.map(zero_map)
data['Zero Offset Peak Width Mean']  = data.groupby('Current')['Zero Offset Peak Width'].transform('mean')
data['File'] = pd.to_numeric(data['File'])

grouped_data = data.groupby(['File', 'Current'])

regression_data = []
for (current, file), group in grouped_data:
    regression = linregress(group['Offset'], group['Peak Width'])

    regression_data.append({'Current': file, 'File': current, 'Slope': regression.slope, 
                            'Intercept': regression.intercept, 'R2': regression.rvalue**2, 'P-value': regression.pvalue, 
                            'StdErr': regression.stderr})

regression_data = pd.DataFrame(regression_data)
regression_data['Slope Mean']  = regression_data.groupby('Current')['Slope'].transform('mean')
regression_data['Intercept Mean']  = regression_data.groupby('Current')['Intercept'].transform('mean')

regression_data['Current'] = regression_data['Current'].astype(float)
data['Current'] = data['Current'].astype(float)

regression_data = regression_data.merge(data, on = ['File', 'Current'], how = 'left')
regression_data['Current'] = np.round(regression_data['Current'] / 1e-11, 0) * 1e-11

regression_data['Pred Offset'] = ((regression_data['Peak Width'] - 
                                        regression_data['Zero Offset Peak Width Mean']) / regression_data['Slope Mean']) 
regression_data['Prediction Error'] = np.abs(regression_data['Pred Offset'] - regression_data['Offset'])
regression_data['Error Bin'] = pd.cut(regression_data['Prediction Error'], bins = error_bins, include_lowest = True,
                                right = False)
regression_data.to_csv(path + outputs + '/regression_data_' + sample + '.csv')

regression_data = regression_data[regression_data['Offset'] > 0]

mod_df0 = (regression_data.drop_duplicates(subset='Current')[['Current', 'Slope Mean',
                                                             'Intercept Mean']].reset_index(drop=True))
data['Current'] = np.round(data['Current'] / 1e-11, 0) * 1e-11
mod_df1 = (data.drop_duplicates(subset='Current')[['Current', 'Zero Offset Peak Width Mean']].reset_index(drop=True))
mod_df = pd.merge(mod_df0, mod_df1, on = 'Current')

mod_df.to_csv(path + outputs + '/model_vals_' + sample + '.csv')

fig, ax = plt.subplots(figsize=(8, 10))
p = sns.relplot(data = regression_data, x = 'Offset', y = 'Pred Offset', hue = 'Error Bin', col = 'Current', palette = 'cool')
p.set_xlabels('')
p.figure.subplots_adjust(top=0.84)
p.figure.text(0.5, 0.015, 'Absolute Offset', ha = 'center', va = 'center')
plt.suptitle('Predicted vs Absolute Offset \n Sample: ' + sample)
plt.savefig(path + outputs + '/regression_results_' + sample + '.png', bbox_inches = 'tight')

##### test model #####
mod_dat = pd.read_csv(path + outputs + '/model_vals_' + sample + '.csv')
mod_dat['Current'] = pd.to_numeric(mod_dat['Current'])

meta_test = []
print('Validating model')
for file in os.listdir(path + test):
    if '.DS_Store' in file:
        continue

    if '.csv' in file:
        continue
    
    for image in tqdm(os.listdir(path + test + '/' + file)):
        if not image.endswith('.tif'):
            continue

        name = image.removesuffix('.tif')
    
        prep = PrepareImages(path + test + '/' + file + '/' + image)
        metadata, im = prep.run()

        ellipse = FitEllipses(im, name, metadata)
        idx90, idx50, ellipse_dict, amp, freqs, peaks, widths = ellipse.run()

        metadata = pd.DataFrame([metadata])
        metadata['Abs Offset'] = pd.to_numeric(metadata['Image'].str.split('__').str[1]).abs()
        metadata['Offset'] = metadata['Image'].str.split('__').str[1]
        offset =  int(metadata['Offset'].iloc[0].removesuffix('.0'))
        metadata['Offset'] = offset
        metadata['File'] = file
        metadata['File'] = pd.to_numeric(metadata['File'])
        metadata['Peak Width'] = widths

        meta_test.append(metadata)

all_metadata_test = pd.concat(meta_test, ignore_index=True)
all_metadata_test['Current'] = pd.to_numeric(all_metadata_test['Current'])
all_metadata_test ['Current']= np.round(all_metadata_test ['Current'] / 1e-11, 0) * 1e-11
all_metadata_test = all_metadata_test.merge(mod_dat, on = 'Current', how = 'left')

zero_map_test = (all_metadata_test.loc[all_metadata_test['Offset'] == 0].set_index(['File', 'Current'])['Peak Width'])
all_metadata_test['Zero Offset Peak Width'] = all_metadata_test.set_index(['File', 'Current']).index.map(zero_map_test)

all_metadata_test['Pred Offset'] = (all_metadata_test['Peak Width'] - 
                                     all_metadata_test['Zero Offset Peak Width Mean']) / all_metadata_test['Slope Mean']
all_metadata_test['Prediction Error'] = np.abs(all_metadata_test['Pred Offset'] - all_metadata_test['Offset'])
all_metadata_test['Error Bin'] = pd.cut(all_metadata_test['Prediction Error'], bins = error_bins, include_lowest = True,
                                    right = False)
all_metadata_test.to_csv(path + outputs + '/all_metadata_test_' + sample + '.csv')

all_metadata_test = all_metadata_test[all_metadata_test['Offset'] > 0]

fig, ax = plt.subplots(figsize=(8, 10))
p = sns.relplot(data = all_metadata_test, x = 'Offset', y = 'Pred Offset', hue = 'Error Bin', col = 'Current', palette = 'cool')
p.set_xlabels('')
p.figure.subplots_adjust(top=0.84)
p.figure.text(0.5, 0.015, 'Absolute Offset', ha = 'center', va = 'center')
plt.suptitle('Predicted Offset vs Absolute Offset from Test Data \n Model: ' + sample)
plt.savefig(path + outputs + '/test_results_test_' + sample + '.png', bbox_inches = 'tight')
