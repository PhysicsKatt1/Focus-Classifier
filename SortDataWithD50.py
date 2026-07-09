##### import #####
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")


# ##### globals #####
# path = r'/media/kat/d19098b8-7064-4621-8345-a76510e141a7/ThruFocusData/5kV.11nA'
path = r'/home/kat/Data'
sorted_path = r'/home/kat/Data/SortedJpegs'
trainAndValFolder = '/AllBeams'
in_focus_path = sorted_path + '/In_focus'
not_in_focus_path = sorted_path + '/Not_in_focus'
regression_path = r'/Users/trentstarkey/Desktop/RegressionData_30kV_0.09nA'

##### functions #####
def parse_tiff(image):
    meta_list = []
    volt_list = []
    tags = image.tag.items()
    for item in tags:
        meta_list.append(item[1][0])

    img_meta = meta_list[int(len(meta_list)-2)]
    xml = ET.fromstring(img_meta)

    amp = float(xml[3][4].text)
    volt = float(xml[3][2].text)
    hfw = float(xml[3][5][0].text)
   
    return hfw, volt, amp

def expected_d50(volt, amp):
    lens_table = pd.read_csv(r'/home/kat/Data/LensTable.csv')
    volt = volt//1000
    amp = int(amp*10/10e-12)
    
    subset = lens_table[lens_table['Volt (kV)'] == volt]
    lens_table_amp_match = min(subset['Amp (pA)'], key=lambda x: abs(x - amp))
    subset2 = subset[subset['Amp (pA)']== lens_table_amp_match]
    d50_expected = int(subset2['d50'])
            
    return d50_expected

def experimental_d50(image, hfw):
    img = np.array(image)
    HFW = hfw*1e9 #convert HFW to nm
    width = img.shape[1]
    height = img.shape[0]
    nm_per_pixel= HFW/width
    d50 = []

    dx = np.gradient(img, axis = 0)
    dy = np.gradient(img, axis = 1)
    mag = np.hypot(dx, dy)
    angle = (180./np.pi)*np.arctan2(dy, dx)
    angle[angle < 0 ] += 180

    thresh = 25
    x = []
    y = []
    widths = []
    try:
        for i in range(0, mag.shape[0]):
            for j in range(0, mag.shape[1]):
                if mag[i, j] >= thresh:
                    if angle[i, j] > 45:
                        x.append(i)
                        y.append(j)

        edge_coordinates = pd.DataFrame({'x':x, 'y':y})

        for n in range(0, len(edge_coordinates)):
            widths.append(np.abs(edge_coordinates['y'].iloc[n] - edge_coordinates['y'].iloc[n+1]))   
    except:    
        pass

    unique_widths, counts = np.unique(widths, return_counts=True)
    cut_off = np.where(unique_widths > 8)[0][0]
    unique_widths = unique_widths[:cut_off]
    counts = counts[:cut_off]

    def gauss(x, A, B):
        return (A)*np.exp(-(x-B)**2)

    try:
        param, _ = curve_fit(gauss, unique_widths, counts)
        gauss_fit = gauss(unique_widths, param[0], param[1])

        diff = np.max(gauss_fit) - np.min(gauss_fit)
        diff35 = 0.35*diff
        diff65 = 0.65*diff
        diff75 = 0.75*diff

        def inv_gauss(y, A, B):
            return B + (-np.log((y)/A))**(.5)
        
        x35 = inv_gauss(diff35, param[0], param[1])
        x65 = inv_gauss(diff65, param[0], param[1])
        x75 = inv_gauss(diff75, param[0], param[1])
        d50_pixels = x75 - 1
        d50 = 2.5*d50_pixels*nm_per_pixel
    
    except ValueError:
        pass

    # plt.scatter(unique_widths, counts, label = 'Data')
    # plt.plot(unique_widths, gauss_fit, color = 'darkturquoise', label = 'Gaussian')
    # plt.ylabel('Frequency')
    # plt.xlabel('Edge Width (pixels)')
    # plt.title('Edge Widths per Image')
    # plt.legend()
    # plt.savefig(r'/home/kat/Data/edge_widths.jpeg')
      
    return d50

def expanded_image(image, d50_expected, d50_experimental):
    im_array = np.array(image)
    info = np.array([d50_expected,  d50_experimental])
    place_holder = np.zeros(im_array.shape[1] - 2)
    complete_info = np.concat([info, place_holder])
    im_out = np.vstack((im_array, complete_info)).astype(np.uint8)
    
    return im_out

##### call functions #####
d50_list_expected = []
d50_list = []
beam_list = []
im_list = []
labels = []

for subfolders in os.listdir(path + trainAndValFolder):
    for images, n in zip(os.listdir(path + trainAndValFolder + '/' + subfolders),
                             tqdm(range(len(os.listdir(path + trainAndValFolder + '/' + subfolders))))):
        im = Image.open(path + trainAndValFolder + '/' + subfolders + '/' + images)
        images = images[:-4]

        #--- uncomment to save training and val data for regression model ---#
        hfw, volt, amp = parse_tiff(im)
        name = images.removeprefix('Focus_data_30000.0V_0.09nA__')
        _, x, y, z = name.split('__')

        if float(y) == 0.0 and float(z) == 0.0:
            im.save(regression_path + '/' + n + '.jpeg', format = 'JPEG')
            labels.append({'Image': n, 'Voltage': volt, 'Current': amp, })

            
        #--- uncomment to save test data for classifier ---#
        # im_list.append(images)
        
        #--- uncomment to include a d50 value in each image for the classifier ---#
        # hfw, volt, amp = parse_tiff(im)
        # d50_expected = expected_d50(volt, amp)
        # d50_list_expected.append(d50_expected)
        # d50 = experimental_d50(im, hfw)
        # d50_list.append(d50)
        
        # im_out = expanded_image(im, d50_expected, d50)
        # im_out = Image.fromarray(im_out)
        
        #--- uncomment to save training and val data for classifier ---#
        # if '30000.0V_0.09nA__0.0__0.0__0.0_' in images:
        # if '30000.0V_0.75nA__0.0__0.0__0.0_' in images:
        #     im.save(in_focus_path + '/' + images + '.jpeg')
            # im_out.save(in_focus_path + '/' + images + '.jpeg')
        # else:
        #     im.save(not_in_focus_path + '/'  + images + '.jpeg')
            # im_out.save(not_in_focus_path + '/'  + images + '.jpeg')
        
# dataframe = pd.DataFrame({'d50_expected':d50_list_expected, 'd50':d50_list, 'im': im_list})
# dataframe.to_csv(r'/home/kat/Data/d50_df.csv')








