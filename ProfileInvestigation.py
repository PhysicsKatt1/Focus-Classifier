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
from scipy.optimize import curve_fit

##### globals #####
path = r'/Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
test = '/Offsets_Test'
border_crop = 3

##### define classes #####
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
            'Current': config['IBeam']['BeamCurrent'],
            'L2': xml_root.find('Optics/FibL2Voltage').text}
        return metadata

    def crop_DC(self):
        im_arr = np.array(self.im, dtype=np.float32) 
        im_arr = im_arr[border_crop:-border_crop, border_crop:-border_crop]
        im_arr = im_arr - np.mean(im_arr)
        # im_arr = im_arr / np.std(im_arr)

        return im_arr

    def fft_imgs(self, image):
        im_fft = np.fft.fft2(image)
        # im_fft = np.fft.fftshift(im_fft)  # uncomment only to plot fft
        # im = np.log1p(np.abs(im_fft))     # uncomment only to plot fft
        im = np.abs(im_fft)

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
    def __init__(self, im, metadata, im_fft):
        self.im = im
        self.metadata = metadata
        self.im_fft = im_fft

    def beam_profiles(self):
        ny, nx = self.im.shape

        hfw = float(self.metadata['HFW']) * 1e6
        dx = hfw / int(self.metadata['ResolutionX'])

        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)
        fx, fy = np.meshgrid(fx, fy)

        radius = np.sqrt(fx**2 + fy**2)
        df = 1 / (max(nx, ny) * dx)
        radial_bin = np.floor(radius / df).astype(int)
        counts = np.bincount(radial_bin.ravel())
        sums = np.bincount(radial_bin.ravel(), weights=self.im.ravel())

        valid = counts > 0
        amp = sums[valid] / counts[valid]    
        freq = np.arange(len(counts))[valid] * df
        
        return amp, freq

    def beam_energy(self):
        ny, nx = self.im.shape

        hfw = float(self.metadata['HFW']) * 1e6
        resolution_x = int(self.metadata['ResolutionX'])
        dx = hfw / resolution_x

        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)
        fx, fy = np.meshgrid(fx, fy)

        radius = np.sqrt(fx**2 + fy**2)
        df = 1 / (max(nx, ny) * dx)
        radial_bin = np.floor(radius / df).astype(int)

        power = np.abs(self.im) ** 2
        counts = np.bincount(radial_bin.ravel())
        sums = np.bincount(radial_bin.ravel(), weights=power.ravel())

        valid = counts > 0
        energy = sums[valid] / counts[valid] 
        freq_energy = np.arange(len(counts))[valid] * df
        bin_counts = counts[valid]

        return energy, freq_energy, bin_counts

    def run(self):
        amp, freq = self.beam_profiles()
        energy, freq_energy, bin_counts = self.beam_energy()
        return amp, freq, energy, freq_energy, bin_counts

class SampleNormalizedAmplitude:
    def __init__(self, metadata_df):
        self.df = metadata_df.copy()

    def run(self):
        self.df['Sample Normalized Amplitude'] = None

        for (sample, file, current), group in self.df.groupby(['Sample', 'File', 'Current']):
            focused_row = group[group['Labeled Offset'] == 0].iloc[0]
            focused_amp = np.asarray(focused_row['Amplitude'], dtype=float).ravel()
            focused_freq = np.asarray(focused_row['Frequency'], dtype=float).ravel()

            for idx, row in group.iterrows():
                amp = np.asarray(row['Amplitude'], dtype=float).ravel()
                freq = np.asarray(row['Frequency'], dtype=float).ravel()
                amp_matched = np.interp(focused_freq, freq, amp)

                ratio = amp_matched / focused_amp
                self.df.at[idx, 'Sample Normalized Amplitude'] = ratio.tolist()

        return self.df

class SampleNormalizedEnergy:
    def __init__(self, metadata_df):
        self.df = metadata_df.copy()

    def run(self):
        self.df['Sample Normalized Energy'] = None
        self.df['Percent Energy Lost'] = None

        for (sample, file, current), group in self.df.groupby(['Sample', 'File', 'Current']):
            focused_row = group[group['Labeled Offset'] == 0].iloc[0]
            focused_e = np.asarray(focused_row['Energy'], dtype=float).ravel()
            focused_freq = np.asarray(focused_row['Energy Frequency'], dtype=float).ravel()

            for idx, row in group.iterrows():
                e = np.asarray(row['Energy'], dtype=float).ravel()
                freq = np.asarray(row['Energy Frequency'], dtype=float).ravel()
                energy_matched = np.interp(focused_freq, freq, e)

                ratio = energy_matched / focused_e
                self.df.at[idx, 'Sample Normalized Energy'] = ratio.tolist()

                percent_lost = 100 * (1 - ratio)
                self.df.at[idx, 'Percent Energy Lost'] = percent_lost.tolist()

        return self.df


    
class BeamPlotter:
    def __init__(self, metadata_df, output_path, pipeline_results_df=None, pipeline_instance=None):
        self.df = metadata_df.copy()
        self.output_path = output_path
        self.results_df = pipeline_results_df.copy() if pipeline_results_df is not None else None
        self.pipeline = pipeline_instance

    @staticmethod
    def normalize(df, cols):
        df = df.copy()
        for col in cols:
            for current, idx in df.groupby('Current').groups.items():
                group_max = max(np.max(a) for a in df.loc[idx, col])
                df.loc[idx, col] = df.loc[idx, col].apply(
                    lambda a: (np.asarray(a, dtype=float) / group_max).tolist())
        return df

    @staticmethod
    def mean_array(s):
        n = min(len(a) for a in s)
        return [np.mean([a[:n] for a in s], axis=0).tolist()] * len(s)

    @staticmethod
    def profile_plot(freq, amp, sample, current, ylabel='Normalized Signal', xlabel='Spatial Frequency (cycles/µm)', title='Mean Radial Profile'):
        freq, amp = np.asarray(freq), np.asarray(amp)
        hue = np.broadcast_to(sample, freq.shape)
        dashes = {s: (1, 1) if s == 'All Data Mean' else '' for s in np.unique(hue)}
        pp = sns.relplot(kind = 'line', x = freq, y = amp, palette = 'cool', hue = hue, 
                         style = hue, dashes = dashes, 
                         col = np.broadcast_to(np.asarray(current,dtype = float), freq.shape))
        pp.set_xlabels('')
        pp.set_ylabels(ylabel)
        pp.set_titles('Current = {col_name}')
        pp.figure.subplots_adjust(top=0.84)
        pp.figure.text(0.5, 0.015, xlabel, ha='center', va='center')
        plt.suptitle(title, x=0.5, y=0.99, ha='center')
        return pp

    def plot_sample_profiles(self, first_rows, average_rows, total_rows):
        plot_data = (pd.concat([first_rows, average_rows, total_rows])[['Sample', 'Current', 'Labeled Offset', 
                    'Frequency', 'Amplitude']].explode(['Frequency', 'Amplitude']).astype({'Frequency': float, 'Amplitude': float}))

        for offset in sorted(plot_data['Labeled Offset'].unique()):
            d = plot_data[plot_data['Labeled Offset'] == offset]
            pp = self.profile_plot(freq=d['Frequency'], amp=d['Amplitude'], 
                                   sample=d['Sample'], current=d['Current'], 
                                   ylabel='Normalized Amplitude', xlabel='Spatial Frequency (cycles/µm)',
                                   title=f'Beam Profile at {offset}V Defocus Offset')
            pp.set(yscale='log')
            pp.savefig(self.output_path + '/Profiles_' + str(offset) + 'V.jpeg', bbox_inches='tight')
            plt.close(pp.figure)

    def plot_mean_frequency_profiles(self, total_rows, offset_order):
        d = total_rows.copy()
        d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
        d = d.sort_values(['Current', 'Labeled Offset'])
        d = (d[['Current', 'Labeled Offset', 'Frequency', 'Amplitude']].explode(['Frequency', 'Amplitude'])
             .astype({'Frequency': float, 'Amplitude': float}))

        pp = self.profile_plot(freq=d['Frequency'], amp=d['Amplitude'], 
                                sample=d['Labeled Offset'].astype(str), current=d['Current'], 
                                ylabel='Normalized Amplitude', xlabel='Spatial Frequency (cycles/µm)',
                                title='Mean Spatial Frequency Profiles')
        pp.set(yscale='log')
        pp.savefig(self.output_path + '/Mean_Frequency_Profiles.jpeg', bbox_inches='tight')
        plt.close(pp.figure)

    def plot_mean_spatial_energy(self, total_rows, offset_order):
        d = total_rows.copy()
        d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
        d = d.sort_values(['Current', 'Labeled Offset'])

        plot_freq, plot_energy, plot_offset, plot_current = [], [], [], []
        for _, row in d.iterrows():
            freq = np.asarray(row['Energy Frequency'], dtype=float)
            energy = np.asarray(row['Energy'], dtype=float)

            plot_freq.extend(freq)
            plot_energy.extend(energy)
            plot_offset.extend([str(row['Labeled Offset'])] * len(freq))
            plot_current.extend([row['Current']] * len(freq))

        d_plot = pd.DataFrame({'Current': plot_current, 'Labeled Offset': plot_offset, 'Energy Frequency': plot_freq, 'Energy': plot_energy})
        d_plot['Labeled Offset'] = pd.Categorical(d_plot['Labeled Offset'], categories=[str(x) for x in offset_order], ordered=True)

        pp = self.profile_plot(freq=d_plot['Energy Frequency'], 
                               amp=d_plot['Energy'], 
                               sample=d_plot['Labeled Offset'], current=d_plot['Current'], 
                               ylabel='Normalized Energy',
                               xlabel='Spatial Frequency (cycles/µm)', 
                               title='Mean Spatial Spectral Energy Spectrum')
        pp.set(yscale='log')
        pp.savefig(self.output_path + '/Mean_Spatial_Energy.jpeg', bbox_inches='tight')
        plt.close(pp.figure)

    def plot_mean_sample_norm_amp(self, total_rows, offset_order):
        d = total_rows.copy()
        d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
        d = d.sort_values(['Current', 'Labeled Offset'])

        plot_freq, plot_amp, plot_offset, plot_current = [], [], [], []

        for _, row in d.iterrows():
            freq = np.asarray(row['Frequency'], dtype=float)
            amp = np.asarray(row['Sample Normalized Amplitude'], dtype=float)

            plot_freq.extend(freq)
            plot_amp.extend(amp)
            plot_offset.extend([str(row['Labeled Offset'])] * len(freq))
            plot_current.extend([row['Current']] * len(freq))

        d_plot = pd.DataFrame({'Current': plot_current, 'Labeled Offset': plot_offset, 
                               'Frequency': plot_freq, 'Amplitude': plot_amp})
        d_plot['Labeled Offset'] = pd.Categorical(d_plot['Labeled Offset'], 
                                    categories=[str(x) for x in offset_order], ordered=True)

        pp = self.profile_plot(freq=d_plot['Frequency'], amp=d_plot['Amplitude'], 
                                sample=d_plot['Labeled Offset'], current=d_plot['Current'], 
                                ylabel='Normalized Amplitude', xlabel='Spatial Frequency (cycles/µm)', 
                                title='Mean Sample Normalized Amplitude')
        pp.set(yscale='log')
        pp.savefig(self.output_path + '/Mean_samp_norm_amp.jpeg', bbox_inches='tight')
        plt.close(pp.figure)

    def plot_mean_sample_norm_energy(self, total_rows, offset_order):
            d = total_rows.copy()
            d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
            d = d.sort_values(['Current', 'Labeled Offset'])
    
            plot_freq, plot_energy, plot_offset, plot_current = [], [], [], []
    
            for _, row in d.iterrows():
                freq = np.asarray(row['Energy Frequency'], dtype=float)
                energy = np.asarray(row['Sample Normalized Energy'], dtype=float)
    
                plot_freq.extend(freq)
                plot_energy.extend(energy)
                plot_offset.extend([str(row['Labeled Offset'])] * len(freq))
                plot_current.extend([row['Current']] * len(freq))
    
            d_plot = pd.DataFrame({'Current': plot_current, 'Labeled Offset': plot_offset, 
                                   'Energy Frequency': plot_freq, 'Energy': plot_energy})
            d_plot['Labeled Offset'] = pd.Categorical(d_plot['Labeled Offset'], 
                                        categories=[str(x) for x in offset_order], ordered=True)
    
            pp = self.profile_plot(freq=d_plot['Energy Frequency'], amp=d_plot['Energy'], 
                                    sample=d_plot['Labeled Offset'], current=d_plot['Current'], 
                                    ylabel='Normalized Energy', xlabel='Spatial Frequency (cycles/µm)', 
                                    title='Mean Sample Normalized Energy')
            pp.set(yscale='log')
            pp.savefig(self.output_path + '/Mean_samp_norm_energy.jpeg', bbox_inches='tight')
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
                if not isinstance(row['Sample Normalized Amplitude'], list) or len(row['Sample Normalized Amplitude']) == 0:
                    continue

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
            pp.set_ylabels('Normalized Amplitude')
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
                if not isinstance(row['Sample Normalized Energy'], list) or len(row['Sample Normalized Energy']) == 0:
                    continue

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
            pp.set_ylabels('Normalized Energy')
            pp.set_titles('Current = {col_name}')
            pp.figure.subplots_adjust(top=0.84)
            pp.figure.suptitle('Sample Normalized Energy ' + sample_name + 'File: ' + file_id)
            pp.set(yscale='log')
            pp.savefig(self.output_path + '/SampNormEnergy_' + sample_name + '_' + file_id + '.jpeg', 
                        bbox_inches='tight')
            plt.close(pp.figure)

    def plot_mean_sample_norm_energydiff(self, total_rows, offset_order):
        d = total_rows.copy()
        d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
        d = d.sort_values(['Current', 'Labeled Offset'])

        plot_freq, plot_energy, plot_offset, plot_current = [], [], [], []

        for _, row in d.iterrows():
            freq = np.asarray(row['Energy Frequency'], dtype=float)
            energy = np.asarray(row['Percent Energy Lost'], dtype=float)

            plot_freq.extend(freq)
            plot_energy.extend(energy)
            plot_offset.extend([str(row['Labeled Offset'])] * len(freq))
            plot_current.extend([row['Current']] * len(freq))

        d_plot = pd.DataFrame({'Current': plot_current, 'Labeled Offset': plot_offset, 
                                'Energy Frequency': plot_freq, 'Percent Energy Lost': plot_energy})
        d_plot['Labeled Offset'] = pd.Categorical(d_plot['Labeled Offset'], 
                                    categories=[str(x) for x in offset_order], ordered=True)

        pp = self.profile_plot(freq=d_plot['Energy Frequency'], amp=d_plot['Percent Energy Lost'], 
                                sample=d_plot['Labeled Offset'], current=d_plot['Current'], 
                                ylabel=f'% Energy Lost', xlabel='Spatial Frequency (cycles/µm)', 
                                title='Mean Percent Energy Lost per Frequency')
        pp.set(yscale='log')
        pp.savefig(self.output_path + '/Mean_samp_norm_energydiff.jpeg', bbox_inches='tight')
        plt.close(pp.figure)

    def run(self):
        groups = self.df.groupby(['Current', 'Labeled Offset'])
        self.df['Mean Frequency'] = groups['Frequency'].transform(self.mean_array)
        self.df['Mean Amplitude'] = groups['Amplitude'].transform(self.mean_array)
        self.df['Mean Energy'] = groups['Energy'].transform(self.mean_array)
        self.df['Mean Sample Normalized Amplitude'] = groups['Sample Normalized Amplitude'].transform(self.mean_array)
        self.df['Mean Sample Normalized Energy'] = groups['Sample Normalized Energy'].transform(self.mean_array)
        self.df['Mean Energy Frequency'] = groups['Energy Frequency'].transform(self.mean_array)
        self.df['Mean Percent Energy Lost'] = groups['Percent Energy Lost'].transform(self.mean_array)

        offset_order = sorted(self.df['Labeled Offset'].dropna().unique())

        first_rows = self.df.drop_duplicates(subset=['Sample', 'Current', 'Labeled Offset']).copy()

        subset_groups = first_rows.groupby(['Current', 'Labeled Offset'])
        first_rows['Subset Mean Frequency'] = subset_groups['Frequency'].transform(self.mean_array)
        first_rows['Subset Mean Amplitude'] = subset_groups['Amplitude'].transform(self.mean_array)
        first_rows['Subset Mean Energy'] = subset_groups['Energy'].transform(self.mean_array)
        first_rows['Subset Mean Sample Normalized Amplitude'] = subset_groups['Sample Normalized Amplitude'].transform(self.mean_array)
        first_rows['Subset Mean Sample Normalized Energy'] = subset_groups['Sample Normalized Energy'].transform(self.mean_array)
        first_rows['Subset Mean Percent Energy Lost'] = subset_groups['Percent Energy Lost'].transform(self.mean_array)

        average_rows = first_rows.drop_duplicates(subset=['Current', 'Labeled Offset']).copy()
        average_rows['Sample'] = 'Average'
        average_rows['Frequency'] = average_rows['Subset Mean Frequency']
        average_rows['Amplitude'] = average_rows['Subset Mean Amplitude']
        average_rows['Energy'] = average_rows['Subset Mean Energy']
        average_rows['Sample Normalized Amplitude'] = average_rows['Subset Mean Sample Normalized Amplitude']
        average_rows['Sample Normalized Energy'] = average_rows['Subset Mean Sample Normalized Energy']
        average_rows['Percent Energy Lost'] = average_rows['Subset Mean Percent Energy Lost']

        total_rows = first_rows.drop_duplicates(subset=['Current', 'Labeled Offset']).copy()
        total_rows['Sample'] = 'All data average'
        total_rows['Frequency'] = total_rows['Mean Frequency']
        total_rows['Amplitude'] = total_rows['Mean Amplitude']
        total_rows['Sample Normalized Amplitude'] = total_rows['Mean Sample Normalized Amplitude']
        total_rows['Sample Normalized Energy'] = total_rows['Mean Sample Normalized Energy']
        total_rows['Percent Energy Lost'] = total_rows['Mean Percent Energy Lost']
        total_rows['Energy'] = total_rows['Mean Energy']
        total_rows['Energy Frequency'] = total_rows['Mean Energy Frequency']

        # plot
        # first_rows   = self.normalize(first_rows,   ['Amplitude', 'Energy'])
        # average_rows = self.normalize(average_rows, ['Amplitude', 'Energy'])
        # total_rows   = self.normalize(total_rows,   ['Amplitude', 'Energy'])

        # self.plot_sample_profiles(first_rows, average_rows, total_rows)
        # self.plot_mean_frequency_profiles(total_rows, offset_order)
        self.plot_mean_spatial_energy(total_rows, offset_order)
        # self.plot_mean_sample_norm_amp(total_rows, offset_order)
        # self.plot_sample_norm_amp_profiles(offset_order, n_samples=3)
        self.plot_mean_sample_norm_energy(total_rows, offset_order)
        # self.plot_sample_norm_energy_profiles(offset_order, n_samples=3)
        self.plot_mean_sample_norm_energydiff(total_rows, offset_order)

##### main execution block #####
meta = []
debug_results = []
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

                # 1. Prepare images and extract metadata
                metadata, im, im_fft = PrepareImages(image_path, file, sample).run()

                # 2. Calculate beam profile and spatial energy stats
                amp, freq, energy, freq_energy, bin_counts = BeamStats(im, metadata.iloc[0], im_fft).run()
    
                metadata['Amplitude'] = [amp.tolist()]
                metadata['Frequency'] = [freq.tolist()]
                metadata['Energy'] = [energy.tolist()]
                metadata['Energy Frequency'] = [freq_energy.tolist()]
                metadata['Energy Bin Counts'] = [bin_counts.tolist()]
                meta.append(metadata)

                # save FFT per image
                cv2.imwrite(path + outputs + '/' + sample + '/' + file + '/' + image_name + '.jpeg', 
                            cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

##### aggregate metadata #####
all_metadata = pd.concat(meta, ignore_index=True)

##### compute sample normalized profiles #####
normalizer = SampleNormalizedAmplitude(metadata_df=all_metadata)
all_metadata = normalizer.run()

energy_normalizer = SampleNormalizedEnergy(metadata_df=all_metadata)
all_metadata = energy_normalizer.run()

##### save metadata #####
all_metadata.to_csv(path + outputs + '/all_meta_data.csv', index=False)

##### plot #####
plotter = BeamPlotter(metadata_df = all_metadata, output_path = path + outputs)
plotter.run()