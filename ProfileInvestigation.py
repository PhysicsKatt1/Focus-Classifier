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
from scipy import curve_fit

##### globals #####
path = r'/Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
test = '/Offsets_Test'
border_crop = 3
sample = 'Shower Drains'

output_data = '/all_metadata_' + sample + '.csv'
os.makedirs(path + outputs + '/' + sample, exist_ok=True)

##### define classes #####
class PrepareImages():
    def __init__(self, img):
        self.img = img
        self.im = Image.open(self.img)

    def extract_metadata(self):
        metadata_text = self.im.tag_v2.get(34682)
        xml_root = ET.fromstring(self.im.tag_v2.get(34683))
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read_file(StringIO(metadata_text))

        im_name = self.img.removesuffix('.tif')
        im_name = im_name.split('data_')

        metadata = {'Image': im_name[1], 'HFW': config['IBeam']['HFW'], 'ResolutionX': config['Image']['ResolutionX'],
            'ResolutionY': config['Image']['ResolutionY'], 'DwellTime': config['Scan']['Dwelltime'],
            'Voltage': config['IBeam']['HV'], 'Current': config['IBeam']['BeamCurrent'],
            'L2': xml_root.find('Optics/FibL2Voltage').text}

        return metadata

    def crop_DC(self):
        im_arr = cv2.normalize(np.array(self.im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im_arr  = im_arr[border_crop:-border_crop, border_crop:-border_crop]
        im_arr = im_arr - np.mean(im_arr)
        
        return im_arr

    def fft_imgs(self, image):
        im_fft = np.fft.fft2(image)
        # im_fft = np.fft.fftshift(im_fft)  # uncomment only to plot fft
        # im = np.log1p(np.abs(im_fft))     # uncomment only to plot fft
        im = np.abs(im_fft)

        return im, im_fft

    def run(self):
        metadata = self.extract_metadata()
        im_arr = self.crop_DC()
        im, im_fft = self.fft_imgs(im_arr)

        return metadata, im, im_fft

class BeamStats():
    def __init__(self, im, metadata, im_fft):
        self.im = im
        self.metadata = metadata
        self.im_fft = im_fft

    def beam_profiles(self):
        ny, nx = self.im_fft.shape

        # normalize pixel size
        hfw = float(self.metadata['HFW'])
        resolution_x = int(self.metadata['ResolutionX'])
        dx = hfw / resolution_x

        # FFT spatial frequencies in physical units
        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)
        fx, fy = np.meshgrid(fx, fy)

        # radial spatial frequency
        radius = np.sqrt(fx**2 + fy**2)
        df = 1 / (max(nx, ny) * dx)
        radial_bin = np.floor(radius / df).astype(int)
        counts = np.bincount(radial_bin.ravel())
        sums = np.bincount(radial_bin.ravel(),weights=self.im.ravel())

        # calculate frequency and amplitude
        valid = counts > 0
        amp = sums[valid] / counts[valid]
        amp = amp /np.max(amp)  # normalized for visualization 
        freq = np.arange(len(counts))[valid] * df

        return amp, freq

    def beam_energy(self):
        ny, nx = self.im_fft.shape

        # normalize pixel size
        hfw = float(self.metadata['HFW'])
        resolution_x = int(self.metadata['ResolutionX'])
        dx = hfw / resolution_x

        # FFT spatial frequencies in physical units
        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)
        fx, fy = np.meshgrid(fx, fy)

        # radial spatial frequency
        radius = np.sqrt(fx**2 + fy**2)
        df = 1 / (max(nx, ny) * dx)
        radial_bin = np.floor(radius / df).astype(int)

        # calculate power and energy
        power = np.abs(self.im_fft) ** 2
        counts = np.bincount(radial_bin.ravel())
        sums = np.bincount(radial_bin.ravel(), weights=power.ravel())

        valid = counts > 0
        energy = sums[valid]
        freq_energy = np.arange(len(counts))[valid] * df

        return energy, freq_energy

    def run(self):
        amp, freq = self.beam_profiles()
        energy, freq_energy = self.beam_energy()

        return amp, freq, energy, freq_energy

class MTFPipeline:
    def __init__(self, metadata_df, f_core_max=0.1e6, f_mid_max=1.0e6, f_tail_max=2.0e6):
        self.df = metadata_df.copy()
        self.f_core_max = f_core_max
        self.f_mid_max = f_mid_max
        self.f_tail_max = f_tail_max

    @staticmethod
    def analytical_rmtf(f, w_mid, sigma_core, sigma_mid, r0):
        w_core = 1.0 - w_mid
        H_core = np.exp(-2 * (np.pi**2) * (sigma_core**2) * (f**2))
        H_mid  = np.exp(-2 * (np.pi**2) * (sigma_mid**2) * (f**2))
        return w_core * H_core + w_mid * H_mid

    def fit_single_rmtf(self, freq, rmtf_data):
        mask = (~np.isnan(rmtf_data)) & (~np.isnan(freq)) & (freq >= 0)
        f = np.asarray(freq)[mask]
        y = np.asarray(rmtf_data)[mask]

        if len(f) < 5:
            return None

        p0 = [0.4, 0.005, 0.05, 0.1]
        bounds = ([0.0, 1e-4, 1e-3, 1e-3], [1.0, 0.1, 1.0, 5.0])

        try:
            popt, pcov = curve_fit(self.analytical_rmtf, f, y, p0=p0, bounds=bounds, maxfev=10000)
            w_mid, sigma_core_um, sigma_mid_um, r0_um = popt
            perr = np.sqrt(np.diag(pcov))

            return {'w_core': 1.0 - w_mid, 'w_mid': w_mid, 'sigma_core_nm': sigma_core_um * 1000.0,
                'sigma_mid_nm': sigma_mid_um * 1000.0, 'r0_tail_nm': r0_um * 1000.0, 'popt': popt,
                'perr': perr}
        except Exception:
            return None

    def integrate_region_energy(self, freq, mtf_profile, f_min, f_max):
        mask = (freq >= f_min) & (freq <= f_max)
        if not np.any(mask) or len(freq[mask]) < 2:
            return 0.0
        return np.trapezoid(mtf_profile[mask], freq[mask])

    def run(self):
        results = []
        for (file_id, current), file_group in self.df.groupby(['File', 'Current']):
            # focused fits
            focused_row = file_group[file_group['Labeled Offset'] == 0]
            if focused_row.empty:
                continue

            f_mtf = np.asarray(focused_row.iloc[0]['MTF'], dtype=float)
            f_freq = np.asarray(focused_row.iloc[0]['Energy Frequency'], dtype=float)
            n_f = min(len(f_freq), len(f_mtf))
            f_freq_c, f_mtf_c = f_freq[:n_f], f_mtf[:n_f]
            res_focused = self.fit_single_rmtf(f_freq_c, f_mtf_c)

            e_core_f = self.integrate_region_energy(f_freq_c, f_mtf_c, 0.0, self.f_core_max)
            e_mid_f  = self.integrate_region_energy(f_freq_c, f_mtf_c, self.f_core_max, self.f_mid_max)
            e_tail_f = self.integrate_region_energy(f_freq_c, f_mtf_c, self.f_mid_max, self.f_tail_max)
            e_tot_f  = e_core_f + e_mid_f + e_tail_f

            # defocused fits
            for idx, row in file_group.iterrows():
                offset = row['Labeled Offset']
                d_mtf = np.asarray(row['MTF'], dtype=float)
                d_freq = np.asarray(row['Energy Frequency'], dtype=float)
                n_d = min(len(d_freq), len(d_mtf))
                d_freq_c, d_mtf_c = d_freq[:n_d], d_mtf[:n_d]
                res_defocused = self.fit_single_rmtf(d_freq_c, d_mtf_c)

                e_core_d = self.integrate_region_energy(d_freq_c, d_mtf_c, 0.0, self.f_core_max)
                e_mid_d  = self.integrate_region_energy(d_freq_c, d_mtf_c, self.f_core_max, self.f_mid_max)
                e_tail_d = self.integrate_region_energy(d_freq_c, d_mtf_c, self.f_mid_max, self.f_tail_max)
                e_tot_d  = e_core_d + e_mid_d + e_tail_d

                # energy deltas
                delta_e_core = e_core_d - e_core_f
                delta_e_mid  = e_mid_d - e_mid_f
                delta_e_tail = e_tail_d - e_tail_f
                delta_e_tot  = e_tot_d - e_tot_f

                delta_sig_core = (res_defocused['sigma_core_nm'] - res_focused['sigma_core_nm']) if (res_defocused and res_focused) else np.nan
                delta_sig_mid  = (res_defocused['sigma_mid_nm'] - res_focused['sigma_mid_nm']) if (res_defocused and res_focused) else np.nan

                results.append({
                    'File': file_id,
                    'Sample': row.get('Sample', None),
                    'Current': current,
                    'Labeled Offset': offset,
                    
                    # Absolute Energy Integrals
                    'E_Core': e_core_d,
                    'E_Mid': e_mid_d,
                    'E_Tail': e_tail_d,
                    'E_Total': e_tot_d,
                    
                    # Energy Deltas
                    'Delta_E_Core': delta_e_core,
                    'Delta_E_Mid': delta_e_mid,
                    'Delta_E_Tail': delta_e_tail,
                    'Delta_E_Total': delta_e_tot,
                    'Pct_Delta_E_Mid': (delta_e_mid / (e_mid_f + 1e-12)) * 100.0,
                    
                    # Fit Parameters & Deltas
                    'sigma_core_nm': res_defocused['sigma_core_nm'] if res_defocused else np.nan,
                    'sigma_mid_nm': res_defocused['sigma_mid_nm'] if res_defocused else np.nan,
                    'Delta_sigma_core_nm': delta_sig_core,
                    'Delta_sigma_mid_nm': delta_sig_mid,
                })

        return pd.DataFrame(results)

##### define functions #####
def mean(s):
    n = min(len(a) for a in s)
    return [np.mean([a[:n] for a in s], axis=0).tolist()] * len(s)

def profile_plot(freq, amp, sample, current):
    freq, amp = np.asarray(freq), np.asarray(amp)
    hue = np.broadcast_to(sample, freq.shape)
    dashes = {s: (1, 1) if s == 'All Data Mean' else '' for s in np.unique(hue)}
    pp = sns.relplot(kind = 'line', x = freq, y = amp, palette = 'cool', hue = hue, style = hue, dashes = dashes,
                     col = np.broadcast_to(np.asarray(current, dtype = float), freq.shape))
    pp.set_xlabels('')
    pp.set_ylabels('Normalized rMTF')
    pp.set_titles('Current = {col_name}')
    pp.figure.subplots_adjust(top=0.84)
    pp.figure.text(0.5, 0.015, 'Spatial Frequency (cycles/µm)', ha = 'center', va = 'center')
    plt.suptitle('Mean Relative Modulation Transfer Function', x = 0.5, y = 0.99, ha = 'center')

    return pp

##### call functions #####
meta = []
for folder in os.listdir(path + inputs):
    if '.DS_Store' in folder:
        continue
     
    for file in os.listdir(path + inputs + '/' + folder):
        if '.DS_Store' in file:
            continue
        
        sample = folder.removeprefix('Offsets_')
        os.makedirs(path + outputs + '/' + sample + '/' + file, exist_ok=True)

        for image in tqdm(os.listdir(path + inputs + '/' + folder + '/' + file), desc = file):
            if '.tif' in image:
                # prepare images
                image_name = image.removesuffix('.tif')

                metadata, im, im_fft = PrepareImages(path + inputs + '/' + folder + '/' + file + '/' + image).run()
                metadata = pd.DataFrame([metadata])

                # save data origin and labeled offset
                metadata['File'] = file
                metadata['Sample'] = sample

                metadata['Labeled Offset'] = metadata['Image'].str.split('__').str[1]
                offset =  int(metadata['Labeled Offset'].iloc[0].removesuffix('.0'))
                metadata['Labeled Offset'] = offset

                # calculate beam profiles, D50, and D90
                amp, freq, energy, freq_energy = BeamStats(im, metadata.iloc[0], im_fft).run()
    
                metadata['Amplitude'] = [amp.tolist()]
                metadata['Frequency'] = [freq.tolist()]
                metadata['Energy'] = [energy.tolist()]
                metadata['Energy Frequency'] = [freq_energy.tolist()]

                # save metadata 
                meta.append(metadata)

                # save FFT per image
                cv2.imwrite(path + outputs + '/' + sample + '/' + file + '/' + image_name + '.jpeg', 
                            cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                # # plot profile per image 
                # pp = profile_plot(freq, amp, sample, metadata['Current'].iloc[0], offset)
                # plt.savefig(path + outputs + '/' + sample + '/' + file + '/Profile_' + image_name + '.jpeg', 
                #                 bbox_inches = 'tight')
                # plt.close(pp.figure)
            
##### save metadata for all images #####
all_metadata = pd.concat(meta, ignore_index=True)
all_metadata.to_csv(path + outputs + '/all_meta_data.csv')

##### average profiles per defocus offset and beam #####
groups = all_metadata.groupby(['Current', 'Labeled Offset'])
all_metadata['Mean Frequency'] = groups['Frequency'].transform(mean)
all_metadata['Mean Amplitude'] = groups['Amplitude'].transform(mean)
all_metadata['Mean Energy'] = groups['Energy'].transform(mean)
all_metadata['Mean Energy Frequency'] = groups['Energy Frequency'].transform(mean)

offset_order = sorted(all_metadata['Labeled Offset'].dropna().unique())

first_rows = all_metadata.drop_duplicates(subset=['Sample', 'Current', 'Labeled Offset']).copy()

subset_groups = first_rows.groupby(['Current', 'Labeled Offset'])
first_rows['Subset Mean Frequency'] = subset_groups['Frequency'].transform(mean)
first_rows['Subset Mean Amplitude'] = subset_groups['Amplitude'].transform(mean)
first_rows['Subset Mean Energy'] = subset_groups['Energy'].transform(mean)

# take the mean only over a subset of beams for simplified plotting
average_rows = first_rows.drop_duplicates(subset=['Current', 'Labeled Offset']).copy()
average_rows['Sample'] = 'Average'
average_rows['Frequency'] = average_rows['Subset Mean Frequency']
average_rows['Amplitude'] = average_rows['Subset Mean Amplitude']
average_rows['Energy'] = average_rows['Subset Mean Energy']

# take the mean over all data
total_rows = first_rows.drop_duplicates(subset=['Current', 'Labeled Offset']).copy()
total_rows['Sample'] = 'All data average'
total_rows['Frequency'] = total_rows['Mean Frequency']
total_rows['Amplitude'] = total_rows['Mean Amplitude']
total_rows['Energy'] = total_rows['Mean Energy']
total_rows['Mean Energy Frequency'] = total_rows['Mean Energy Frequency']

# ##### plot sample profile for each sample, defocus offset, current and the corresponding mean profile #####
# plot_data = (pd.concat([first_rows, average_rows, total_rows])[['Sample', 'Current', 'Labeled Offset', 'Frequency', 'Amplitude']]
#              .explode(['Frequency', 'Amplitude'])
#              .astype({'Frequency': float, 'Amplitude': float}))

# for offset in sorted(plot_data['Labeled Offset'].unique()):
#     d = plot_data[plot_data['Labeled Offset'] == offset]
#     pp = profile_plot(d['Frequency'], d['Amplitude'], d['Sample'], d['Current'])
#     pp.savefig(path + outputs + '/Profiles_' + str(offset) + 'V.jpeg', bbox_inches = 'tight')
#     plt.close(pp.figure)

# ##### plot the mean frequency profile for each defocus offset per beam #####
# d = total_rows.copy()

# d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
# d = d.sort_values(['Current', 'Labeled Offset'])
# d = (d[['Current', 'Labeled Offset', 'Frequency', 'Amplitude']].explode(['Frequency', 'Amplitude'])
#     .astype({'Frequency': float, 'Amplitude': float}))

# pp = profile_plot(d['Frequency'], d['Amplitude'], d['Labeled Offset'].astype(str), d['Current'])
# pp.savefig(path + outputs + '/Mean_Frequency_Profiles.jpeg', bbox_inches='tight')
# plt.close(pp.figure)

# ##### plot mean energy profile for each defocus offset per beam #####
# d = total_rows.copy()

# d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=offset_order, ordered=True)
# d = d.sort_values(['Current', 'Labeled Offset'])

# plot_freq = []
# plot_energy = []
# plot_offset = []
# plot_current = []

# for _, row in d.iterrows():
#     freq = np.asarray(row['Energy Frequency'], dtype=float)
#     energy = np.asarray(row['Energy'], dtype=float)
#     n = min(len(freq), len(energy))

#     freq = freq[:n]
#     energy = energy[:n]

#     plot_freq.extend(freq)
#     plot_energy.extend(energy)
#     plot_offset.extend([str(row['Labeled Offset'])] * n)
#     plot_current.extend([row['Current']] * n)

# d = pd.DataFrame({'Current': plot_current, 'Labeled Offset': plot_offset, 'Energy Frequency': plot_freq,
#     'Energy': plot_energy})
# d['Energy_Norm'] = d.groupby(['Current', 'Labeled Offset'], observed=True)['Energy'].transform(
#     lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-12)))
# d['Labeled Offset'] = pd.Categorical(d['Labeled Offset'], categories=[str(x) for x in offset_order],
#     ordered=True)

# pp = profile_plot(d['Energy Frequency'], d['Energy_Norm'], d['Labeled Offset'], d['Current'])
# pp.savefig(path + outputs + '/Mean_Spatial_Energy.jpeg', bbox_inches='tight')
# plt.close(pp.figure)

##### calculate MTF and plot the mean per offset per beam #####
all_metadata['MTF'] = None
all_metadata['MTF Cutoff Freq'] = None

for (file, current), group in all_metadata.groupby(['File', 'Current']):
    focused = group[group['Labeled Offset'] == 0]
    focused_energy = np.asarray(focused['Energy'].iloc[0], dtype=float)
    focused_freq = np.asarray(focused['Energy Frequency'].iloc[0], dtype=float)

    # estimate noise floor from high-frequency tail
    tail = focused_energy[-max(int(0.1 * len(focused_energy)), 10):]
    noise_floor = np.median(tail)
    sig_focused_full = np.maximum(0, focused_energy - noise_floor)
    peak_focused = np.max(sig_focused_full) if np.max(sig_focused_full) > 0 else 1e-12
    signal_threshold = 0.03 * peak_focused
    below_threshold = np.where(sig_focused_full < signal_threshold)[0]
    valid_below = [idx for idx in below_threshold if idx > 2]
    
    if len(valid_below) > 0:
        cutoff_idx = valid_below[0]
    else:
        cutoff_idx = len(focused_energy)

    cutoff_idx = max(1, cutoff_idx)
    cutoff_freq = focused_freq[cutoff_idx - 1]

    # calculate MTF for every defocus offset
    for idx, row in group.iterrows():
        defocused_energy = np.asarray(row['Energy'], dtype=float)
        n = min(len(focused_energy), cutoff_idx, len(defocused_energy))
        if n == 0:
            continue

        focused_profile = focused_energy[:n]
        defocused_profile = defocused_energy[:n]
        sig_focused = np.maximum(0, focused_profile - noise_floor)
        sig_defocused = np.maximum(0, defocused_profile - noise_floor)

        raw_mtf = np.divide(sig_defocused, sig_focused, out=np.zeros_like(sig_defocused), 
            where=(sig_focused >= signal_threshold))
        raw_mtf = raw_mtf / np.max(raw_mtf)

        all_metadata.at[idx, 'MTF'] = raw_mtf.tolist()
        all_metadata.at[idx, 'MTF Cutoff Freq'] = cutoff_freq

#### average MTF by sample, current, and defocus offset #####
# sample_mtf = []

# for (sample_name, current, offset), group in all_metadata.groupby(['Sample', 'Current', 'Labeled Offset']):
#     mtf_profiles = [np.asarray(x, dtype=float) for x in group['MTF'] if isinstance(x, list)]
#     n = min(len(x) for x in mtf_profiles)
#     mtf_array = np.asarray([x[:n] for x in mtf_profiles], dtype=float)
#     sample_mean_mtf = np.mean(mtf_array, axis=0)
#     freq = np.asarray(group.iloc[0]['Energy Frequency'], dtype=float)[:n]
#     sample_mtf.append({'Sample': sample_name, 'Current': current, 'Labeled Offset': offset,
#         'Frequency': freq, 'MTF': sample_mean_mtf})

# plot_freq = []
# plot_mtf = []
# plot_offset = []
# plot_current = []

# for current in sorted(all_metadata['Current'].unique()):
#     current_samples = [x for x in sample_mtf if x['Current'] == current]

#     for offset in sorted(set(x['Labeled Offset'] for x in current_samples)):
#         offset_samples = [x for x in current_samples if x['Labeled Offset'] == offset]
#         n = min(min(len(x['MTF']) for x in offset_samples), min(len(x['Frequency']) for x in offset_samples))
#         mtf_array = np.asarray([x['MTF'][:n] for x in offset_samples], dtype=float)

#         mean_mtf = np.mean(mtf_array, axis=0)
#         mean_freq = np.asarray(offset_samples[0]['Frequency'], dtype=float)[:n]

#         plot_freq.extend(mean_freq)
#         plot_mtf.extend(mean_mtf)
#         plot_offset.extend([str(offset)] * n)
#         plot_current.extend([current] * n)

# pp = profile_plot(plot_freq, plot_mtf, plot_offset, plot_current)
# pp.savefig(path + outputs + '/Mean_MTF.jpeg', bbox_inches='tight')
# plt.close(pp.figure)

##### plot Ishitani's model fits and energy deltas #####
pipeline = MTFPipeline(
    metadata_df=all_metadata,
    f_core_max=0.1e6,
    f_mid_max=1.0e6,
    f_tail_max=2.0e6
)

df_pipeline_results = pipeline.run()
df_pipeline_results.to_csv(path + outputs + '/pipeline_results.csv', index=False)
