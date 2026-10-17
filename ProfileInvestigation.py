##### imports #####
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import os
from tqdm import tqdm 
from PIL import Image
import configparser
from io import StringIO
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import pymc as pm
import pytensor.tensor as pt
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from scipy.special import kv
from scipy.signal.windows import tukey
from scipy.ndimage import median_filter

##### globals #####
path = r'/Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
test = '/Offsets_Test'
border_crop = 3
alpha = 0.0

##### define classes #####
def mean_profile(group, freq_col, val_col, n=5000):
    grid = np.linspace(0, min(f[-1] for f in group[freq_col]), n)
    return grid, np.mean([np.interp(grid, f, v) for f, v in zip(group[freq_col], group[val_col])], axis=0)

class PrepareImages():
    def __init__(self, img_path, file_id, sample_name):
        self.img_path = img_path
        self.file_id = file_id
        self.sample_name = sample_name
        self.im = Image.open(self.img_path)

    def extract_metadata(self):
        metadata_text = self.im.tag_v2.get(34682)
        xml_root = ET.fromstring(self.im.tag_v2.get(34683))
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read_file(StringIO(metadata_text))

        img_filename = os.path.basename(self.img_path)
        im_name = img_filename.removesuffix('.tif').split('data_')

        metadata = {
            'Image': im_name[1] if len(im_name) > 1 else im_name[0],
            'HFW': config['IBeam']['HFW'],
            'ResolutionX': config['Image']['ResolutionX'],
            'ResolutionY': config['Image']['ResolutionY'],
            'DwellTime': config['Scan']['Dwelltime'],
            'Voltage': config['IBeam']['HV'],
            'Current': float(config['IBeam']['BeamCurrent']),
            'L2': xml_root.find('Optics/FibL2Voltage').text}
        return metadata

    def crop_DC(self):
        im_arr = np.array(self.im, dtype=np.float32) 
        im_arr = im_arr[border_crop:-border_crop, border_crop:-border_crop]
        im_arr = im_arr - np.mean(im_arr)

        return im_arr

    def fft_imgs(self, image):
        ny, nx = image.shape
        window_y = tukey(ny, alpha=alpha)
        window_x = tukey(nx, alpha=alpha)
        window_2d = np.outer(window_y, window_x)

        windowed_image = image * window_2d
        im_fft = np.fft.fft2(windowed_image)
        # im_fft = np.fft.fft2(image)
        # im_fft = np.fft.fftshift(im_fft)  # uncomment only to plot fft
        # im = np.log1p(np.abs(im_fft))     # uncomment only to plot fft
        im = np.abs(im_fft) / im_fft.size

        return im, im_fft

    def run(self):
        dict_meta = self.extract_metadata()
        metadata = pd.DataFrame([dict_meta])

        metadata['File'] = self.file_id
        metadata['Sample'] = self.sample_name
        metadata['Labeled Offset'] = metadata['Image'].str.split('__').str[1]
        offset = int(metadata['Labeled Offset'].iloc[0].removesuffix('.0'))
        metadata['Labeled Offset'] = offset

        im_arr = self.crop_DC()
        im, im_fft = self.fft_imgs(im_arr)

        return metadata, im, im_fft

class BeamStats():
    def __init__(self, im, metadata):
        self.im = im
        self.metadata = metadata
        self.df = metadata.copy()

    def beam_profiles(self):
        ny, nx = self.im.shape

        hfw = float(self.metadata['HFW']) * 1e6
        dx = hfw / int(self.metadata['ResolutionX'])
        nyquist = 1 / (2 * dx)

        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)
        fx, fy = np.meshgrid(fx, fy)

        radius = np.sqrt(fx**2 + fy**2)
        keep = radius < nyquist

        df = 1 / (max(nx, ny) * dx)
        radial_bin = np.floor(radius / df).astype(int)
        counts = np.bincount(radial_bin[keep], minlength=1)
        valid = counts >= 10

        freq = np.arange(len(counts))[valid] * df

        power = (self.im)**2
        sums_power = np.bincount(radial_bin[keep], weights=power[keep])
        energy = sums_power[valid] / counts[valid]
        amp = np.sqrt(energy)

        freq_raw = radial_bin[keep] * df
        amp_raw = self.im[keep]
        energy_raw = amp_raw**2

        return amp, freq, energy, counts[valid], amp_raw, energy_raw, freq_raw

    def run(self):
        amp, freq,  energy, bin_counts, amp_raw, energy_raw, freq_raw = self.beam_profiles()
        return amp, freq, energy, bin_counts, amp_raw, energy_raw, freq_raw
    
class SampNorm():
    def __init__(self, im, metadata):
        self.im = im
        self.metadata = metadata
        self.df = metadata.copy()

    def SampleNormalized(self):
        self.df['Sample Normalized Amplitude'] = None
        self.df['Sample Normalized Energy'] = None
        
        for (sample, file, current), group in self.df.groupby(['Sample', 'File', 'Current']):
            focused_row = group[group['Labeled Offset'] == 0].iloc[0]
            focused_freq = np.asarray(focused_row['Frequency'], dtype=float).ravel()
            focused_amp = np.asarray(focused_row['Amplitude'], dtype=float).ravel()
            focused_e = np.asarray(focused_row['Energy'], dtype=float).ravel()

            for idx, row in group.iterrows():
                freq = np.asarray(row['Frequency'], dtype=float).ravel()
                amp = np.asarray(row['Amplitude'], dtype=float).ravel()
                e = np.asarray(row['Energy'], dtype=float).ravel()

                amp_ratio = amp / np.interp(freq, focused_freq, focused_amp, left=np.nan, right=np.nan) 
                e_ratio = e / np.interp(freq, focused_freq, focused_e, left=np.nan, right=np.nan)

                self.df.at[idx, 'Sample Normalized Amplitude'] = amp_ratio.tolist()
                self.df.at[idx, 'Sample Normalized Energy'] = e_ratio.tolist()

        return self.df

    def run(self):
        self.df = self.SampleNormalized()
        return self.df

class BeamPlotter:
    def __init__(self, metadata_df, output_path):
        self.df = metadata_df.copy()
        self.output_path = output_path

    @staticmethod
    def profile_plot(freq, amp, sample, current, ylabel='',  xlabel='Spatial Frequency (cycles/µm)', title='',
                     palette='cool'):
        freq, amp = np.asarray(freq), np.asarray(amp)
        hue = np.broadcast_to(sample, freq.shape)
        pp = sns.relplot(kind = 'line', x = freq, y = amp, palette = palette, hue = hue, errorbar = None,
                         col = np.broadcast_to(np.asarray(current,dtype = float), freq.shape))
        pp.set_xlabels('')
        pp.set_ylabels(ylabel)
        pp.set_titles('Current = {col_name}')
        pp.figure.subplots_adjust(top=0.84)
        pp.figure.text(0.5, 0.015, xlabel, ha='center', va='center')
        plt.suptitle(title, x=0.5, y=0.99, ha='center')
        return pp

    def plot_means(self, freq_col, val_col, ylabel, title, filename, drop_focus=False, log=True):
        rows = []
        for (current, offset), g in self.df.groupby(['Current', 'Labeled Offset']):
            f, v = mean_profile(g, freq_col, val_col)
            rows.append(pd.DataFrame({'Current': current, 'Labeled Offset': offset, freq_col: f, val_col: v}))
        d = pd.concat(rows)
        if drop_focus:
            d = d[d['Labeled Offset'] != 0]

        pp = self.profile_plot(freq=d[freq_col], amp=d[val_col],
                           sample=d['Labeled Offset'].astype(str), current=d['Current'],
                           ylabel=ylabel, title=title)
        if log:
            pp.set(yscale='log')

        pp.savefig(self.output_path + filename, bbox_inches='tight')
        plt.close(pp.figure)

    def plot_sample_norm_amp_profiles(self, offset_order, n_samples, random_state=None):
        rng = np.random.default_rng(random_state)
        sample_file_pairs = self.df[['Sample', 'File']].drop_duplicates().to_numpy()
        n_select = min(n_samples, len(sample_file_pairs))
        selected_pairs = sample_file_pairs[rng.choice(len(sample_file_pairs), size=n_select, replace=False)]

        for sample_name, file_id in selected_pairs:
            file_data = self.df[(self.df['Sample'] == sample_name) & (self.df['File'] == file_id)].copy()
            
            plot_freq, plot_amp, plot_offset, plot_current, plot_trace_id = [], [], [], [], []

            for idx, row in file_data.iterrows():
                freq = np.asarray(row['Frequency'], dtype=float)
                amp = np.asarray(row['Sample Normalized Amplitude'], dtype=float)

                n = min(len(freq), len(amp))

                plot_freq.extend(freq[:n])
                plot_amp.extend(amp[:n])
                plot_offset.extend([row['Labeled Offset']] * n)
                plot_current.extend([row['Current']] * n)
                plot_trace_id.extend([str(row['Image'])] * n)

            plot_data = pd.DataFrame({'Frequency': plot_freq, 'Sample Normalized Amplitude': plot_amp,
                'Labeled Offset': plot_offset, 'Current': plot_current, 'Trace_ID': plot_trace_id})

            plot_data['Labeled Offset'] = pd.Categorical(plot_data['Labeled Offset'], 
                categories=offset_order, ordered=True)

            pp = sns.relplot(data=plot_data, kind='line', x='Frequency', y='Sample Normalized Amplitude', 
                             hue='Labeled Offset', units='Trace_ID', estimator=None, 
                             col='Current', palette='cool', alpha=0.85, linewidth=1.2)

            pp.set_xlabels('Spatial Frequency (cycles/µm)')
            pp.set_ylabels('Mean Magnitude per Frequency Bin')
            pp.set_titles('Current = {col_name}')
            pp.figure.subplots_adjust(top=0.84)
            pp.figure.suptitle('Sample Normalized Amplitude ' + sample_name + ' File: ' + file_id)
            pp.set(yscale='log')
            pp.savefig(self.output_path + '/SampNormAmp_' + sample_name + '_' + file_id + '.jpeg', 
                       bbox_inches='tight')
            plt.close(pp.figure)

    def plot_sample_norm_energy_profiles(self, offset_order, n_samples, random_state=None):
        rng = np.random.default_rng(random_state)
        sample_file_pairs = self.df[['Sample', 'File']].drop_duplicates().to_numpy()
        n_select = min(n_samples, len(sample_file_pairs))
        selected_pairs = sample_file_pairs[rng.choice(len(sample_file_pairs), size=n_select, replace=False)]

        for sample_name, file_id in selected_pairs:
            file_data = self.df[(self.df['Sample'] == sample_name) & (self.df['File'] == file_id)].copy()
            
            plot_freq, plot_energy, plot_offset, plot_current, plot_trace_id = [], [], [], [], []

            for idx, row in file_data.iterrows():
                freq = np.asarray(row['Energy Frequency'], dtype=float)
                energy = np.asarray(row['Sample Normalized Energy'], dtype=float)

                n = min(len(freq), len(energy))

                plot_freq.extend(freq[:n])
                plot_energy.extend(energy[:n])
                plot_offset.extend([row['Labeled Offset']] * n)
                plot_current.extend([row['Current']] * n)
                plot_trace_id.extend([str(row['Image'])] * n)

            plot_data = pd.DataFrame({'Energy Frequency': plot_freq, 'Sample Normalized Energy': plot_energy,
                'Labeled Offset': plot_offset, 'Current': plot_current, 'Trace_ID': plot_trace_id})

            plot_data['Labeled Offset'] = pd.Categorical(plot_data['Labeled Offset'], 
                categories=offset_order, ordered=True)

            pp = sns.relplot(data=plot_data, kind='line', x='Energy Frequency', y='Sample Normalized Energy', 
                             hue='Labeled Offset', units='Trace_ID', estimator=None, 
                             col='Current', palette='cool', alpha=0.85, linewidth=1.2)

            pp.set_xlabels('Spatial Frequency (cycles/µm)')
            pp.set_ylabels('Mean Magnitude Squared per Frequency Bin')
            pp.set_titles('Current = {col_name}')
            pp.figure.subplots_adjust(top=0.84)
            pp.figure.suptitle('Sample Normalized Power ' + sample_name + ' File: ' + file_id)
            pp.set(yscale='log')
            pp.savefig(self.output_path + '/SampNormEnergy_' + sample_name + '_' + file_id + '.jpeg', 
                       bbox_inches='tight')
            plt.close(pp.figure)

    def run(self):
        self.plot_means('Frequency', 'Amplitude', 'Magnitude',
                        'Mean Spatial Frequency Distribution', '/Mean_Frequency_Profiles.jpeg')
        self.plot_means('Energy Frequency', 'Energy', 'Power',
                        'Mean Power Distribution', '/Mean_Spatial_Energy.jpeg')
        self.plot_means('Frequency', 'Sample Normalized Amplitude', r'$\tilde{K}(f)$',
                        'Mean Sample Normalized Frequency Magnitude', '/Mean_samp_norm_amp.jpeg', drop_focus=True)
        self.plot_means('Energy Frequency', 'Sample Normalized Energy', r'$\tilde{K}(f)$',
                        'Mean Sample Normalized Power', '/Mean_samp_norm_energy.jpeg', drop_focus=True)

        # offset_order = sorted(self.df['Labeled Offset'].dropna().unique())
        # self.plot_sample_norm_amp_profiles(offset_order, n_samples=3)
        # self.plot_sample_norm_energy_profiles(offset_order, n_samples=3)

##### main execution block #####
meta = []
for folder in os.listdir(path + inputs):
    if '.DS_Store' in folder:
        continue
     
    for file in os.listdir(path + inputs + '/' + folder):
        if '.DS_Store' in file:
            continue
        
        sample = folder.removeprefix('Offsets_')
        os.makedirs(path + outputs + '/' + sample + '/' + file, exist_ok=True)

        for image in tqdm(os.listdir(path + inputs + '/' + folder + '/' + file), desc=file):
            if '.tif' in image:
                image_path = path + inputs + '/' + folder + '/' + file + '/' + image
                image_name = image.removesuffix('.tif')

                # prepare images and extract metadata
                metadata, im, im_fft = PrepareImages(image_path, file, sample).run()

                # calculate beam profile and spatial energy stats
                amp, freq, energy, bin_counts, amp_raw, energy_raw, freq_raw = BeamStats(im, metadata.iloc[0]).run()

                metadata['Amplitude'] = [amp.tolist()]
                metadata['Frequency'] = [freq.tolist()]
                metadata['Energy'] = [energy.tolist()]
                metadata['Energy Frequency'] = [freq.tolist()]
                metadata['Energy Bin Counts'] = [bin_counts.tolist()]
                # metadata['Amplitdude Raw'] = [amp_raw.tolist()]
                # metadata['Energy Raw'] = [energy_raw.tolist()]
                # metadata['Frequency Raw'] = [freq_raw.tolist()]
                meta.append(metadata)

                '''
                # save FFT per image
                cv2.imwrite(path + outputs + '/' + sample + '/' + file + '/' + image_name + '.jpeg', 
                            cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                '''

##### aggregate metadata #####
all_metadata = pd.concat(meta, ignore_index=True)

##### sasmple normalizing #####
print('Sample normalizing.')
all_metadata = SampNorm(None, all_metadata).run()

##### save metadata #####
all_metadata.to_csv(path + outputs + '/all_meta_data.csv', index=False)

##### plot #####
print('Plotting.')
plotter = BeamPlotter(metadata_df = all_metadata, output_path = path + outputs)
plotter.run()

print('Bye!')

