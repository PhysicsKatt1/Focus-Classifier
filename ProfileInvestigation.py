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
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

##### globals #####
path = r'/Users/trentstarkey/Desktop'
inputs = '/Offsets'
outputs = '/Offsets_FFTs'
test = '/Offsets_Test'

##### define helper functions #####
def min_nyquist(metadata):
    # A common frequency grid is limited at high frequency by the coarsest pixel spacing, which has the smallest
    # Nyquist cutoff, and at low frequency by the smallest field of view, which has the widest native bin spacing.
    dx = metadata['HFW'].astype(float) * 1e6 / metadata['ResolutionX'].astype(float)
    return  1 / (2 * dx.max()), 1 / (metadata['HFW'].astype(float).min() * 1e6)

def scintallator_atten(f, v):
    # 70e-9 s = tau1 and 300e-9 s = tau2, https://eureka.patsnap.com/materials/yag-scintillator-radiation-detection
    #https://www.google.com/search?q=model+high+frequency+signal+attenuation+from+a+E-T+detector+with+a+YAG+crystal&sca_esv=ea58cd80e616cff1&biw=1435&bih=754&sxsrf=AKL5yDjdw6rK0YthvZh1ha71Xxx2c_IG1w%3A1793309788744&ei=XLzjaujzLNKRm9cPyqCt0As&uact=5&oq=model+high+frequency+signal+attenuation+from+a+E-T+detector+with+a+YAG+crystal&gs_lp=Egxnd3Mtd2l6LXNlcnAaAhgCIk5tb2RlbCBoaWdoIGZyZXF1ZW5jeSBzaWduYWwgYXR0ZW51YXRpb24gZnJvbSBhIEUtVCBkZXRlY3RvciB3aXRoIGEgWUFHIGNyeXN0YWxIg8YCUOGaAVizxAJwC3gBkAEAmAFxoAHiOqoBBTc3LjExsAEAuAEDyAEA-AEBmAI_oALqMqgCEMICChAAGEcY1gQYsAPCAhcQLhjcBhi4BhjaBhjYAhjIAxiwA9gBAcICBxAjGOoCGCfCAhAQIxiiBxieBhjwBRjqAhgnwgIXEAAYgAQYigUYkQIY5wYY6gIYtALYAQLCAgQQIxgnwgILEAAYgAQYigUYkQLCAg0QABiABBiKBRhDGIsDwgIXEC4YgAQYsQMYgwEYiwMYmwMYqAMY0gPCAgoQABiABBiKBRhDwgILEC4Y-gYY8gYYjwfCAgoQLhiABBiKBRhDwgIWEC4YgAQYigUYQxixAxiLAxijAxioA8ICEBAAGIAEGIoFGEMYsQMYiwPCAggQLhiABBixA8ICFBAuGIAEGLEDGIMBGIsDGJkDGKgDwgILEAAYgAQYsQMYiwPCAggQABiABBixA8ICDRAAGIAEGIoFGEMYsQPCAhEQLhiABBixAxiLAxioAxilA8ICFxAuGIAEGLEDGMcBGNEDGIsDGKgDGNIDwgIXEC4YgAQYsQMY5QQYiwMYpQMYqAMYoAPCAhEQLhiABBixAxiLAxilAxioA8ICGRAuGIAEGIoFGEMYlwUY3AQY3gQY4ATYAQLCAgUQABiABMICCxAAGIAEGLQHGIsDwgILEC4YgAQYxwEYrwHCAgcQABgDGIsDwgIIEAAYgAQYiwPCAg4QLhiABBiLAxioAxiaA8ICBBAAGAPCAgkQABgWGB4YiwPCAgYQABgWGB7CAgsQABiABBiKBRiGA8ICBRAAGO8FwgIIEAAYgAQYogTCAgUQIRigAcICBRAhGKsCwgIFECEYnwXCAgQQIRgKmAMU4gMEGAAgXeIDBBgAIF7iAwQYACBf4gMEGAAgYOIDBBgAIGHiAwQYACBi4gMFEgExIEDxBcIZ7RdvDNFIiAYBkAYLugYECAEYGboGBggCEAEYAZIHBTIxLjQyoAeklgOyBwUxOS40MrgHwTLCBwYyLTYxLjLIB8kCgAgB&sclient=gws-wiz-serp
    # v converts the scintillator decay time into a spatial frequency response using the image acquisition scan velocity.
    return np.sqrt(0.91 / (1 + (2 * np.pi * f * v * 70e-9)**2) + (1 - 0.91) / (1 + (2 * np.pi * f * v * 300e-9)**2)) 

def gauss(f, A, f0):
    return A * np.exp(-f**2 / (2 * f0**2))

def white_noise(C):
    # White noise has a frequency-independent power spectral density represented by constant power.
    return C

def Denoise(df, low=2.0, high=100.):
    df['PreDenoised Amplitude'] = df['Amplitude']
    df['PreDenoised Energy'] = df['Energy']
    df['Noise Floor'] = None
    df['Fit Params'] = None 

    for _, group in df.groupby(['Current', 'Sample', 'File']):
        fits = []
        for idx, row in group.iterrows():
            # Focused profiles do not exhibit a sufficiently distinct noise-floor region and are therefore
            # not used to estimate the background noise power.
            if row['Labeled Offset'] != 0:
                f = np.asarray(row['Frequency'], dtype=float)
                P = np.asarray(row['Energy'], dtype=float)
                k = np.asarray(row['Energy Bin Counts'], dtype=float)
                w = (f >= low) & (f <= high)
                
                dx = float(row['HFW']) * 1e6 / int(row['ResolutionX'])
                v = dx / float(row['DwellTime'])
                # Both signal and noise are attenuated at high spatial frequencies by the finite temporal 
                # response of the scintillator.
                model = lambda f, A, f0, C: (gauss(f, A, f0) + white_noise(C)) * scintallator_atten(f, v)
   
                lo = [0., 0.1, 1.]
                hi = [np.inf, 20.0, 0.9 * P[w].max()]

                popt, _ = curve_fit(model, f[w],  P[w], sigma=P[w], 
                                    p0=[P[w][0], 1.0, 0.1 * (P[w] * scintallator_atten(f[w], v)).min()],
                                    bounds=(lo, hi), max_nfev=10000)

                fits.append(popt)
                df.at[idx, 'Fit Params'] = popt.tolist()

        for idx, row in group.iterrows():
            f = np.asarray(row['Frequency'], dtype=float)
            k = np.asarray(row['Energy Bin Counts'], dtype=float)
            dx = float(row['HFW']) * 1e6 / int(row['ResolutionX'])
            v = dx / float(row['DwellTime'])

            # The underlying white-noise power cab be estimated from the mean fitted noise coefficient
            # and the total noise power can be estimated by applying the frequency-dependent scintillator attenuation
            # to the mean fitted noise coefficient.
            N = np.mean([p[2] for p in fits]) * scintallator_atten(f, v)
            clean = np.asarray(row['Energy'], dtype=float) - N
            # Radially binned noise power has an approximate standard error proportional to 1 / sqrt(k),
            # where k is the number of frequency pixels in the bin. Because the FFT power of a real-valued
            # image is Hermitian symmetric, conjugate frequency pairs are redundant, giving an effective
            # independent sample count of approximately k / 2. Rather than removing all values below the
            # estimated noise floor, we can retain only values exceeding the noise floor by its estimated standard
            # error to account for uncertainty in the noise estimate and avoid overly aggressive subtraction.
            clean = np.where(clean > 3 * N / np.sqrt(k / 2), clean, np.nan) #/ np.sqrt(k / 2)

            df.at[idx, 'Noise Floor'] = N.tolist()
            df.at[idx, 'Energy'] = clean.tolist()
            df.at[idx, 'Amplitude'] = np.sqrt(clean).tolist()

    return df

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
            'Current': float(config['IBeam']['BeamCurrent']),
            'L2': xml_root.find('Optics/FibL2Voltage').text}
        return metadata

    def norm(self, dwell):
        # Normalize pixel intensity by dwell time to account for differences in per-pixel acquisition time between images.
        im_arr = np.array(self.im, dtype=np.float32) 
        im_arr = im_arr / dwell         

        return im_arr

    def periodic_component(self, image):    
        # Correct for FFT spectral leakage by removing discontinuities at the image boundaries 
        # before applying the Fourier transform, https://link.springer.com/article/10.1007/s10851-010-0227-1. 
        ny, nx = image.shape
        v = np.zeros_like(image)
        v[0, :] = image[-1, :] - image[0, :]
        v[-1, :] = -v[0, :]
        v[:, 0] += image[:, -1] - image[:, 0]
        v[:, -1] -= image[:, -1] - image[:, 0]
        cy = np.cos(2 * np.pi * np.arange(ny) / ny)[:, None]
        cx = np.cos(2 * np.pi * np.arange(nx) / nx)[None, :]
        denom = 2 * (cy + cx - 2)
        denom[0, 0] = 1
        s = np.fft.fft2(v) / denom
        s[0, 0] = 0
        return image - np.fft.ifft2(s).real

    def fft_imgs(self, image):
        im_fft = np.fft.fft2(self.periodic_component(image))
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
        metadata['Mean Counts'] = np.array(self.im, dtype=float).mean()

        im_arr = self.norm(float(metadata['DwellTime'].iloc[0]))
        im, im_fft = self.fft_imgs(im_arr)

        return metadata, im, im_fft

class BeamStats():
    def __init__(self, im, metadata, freq_max, freq_step):
        self.im = im
        self.metadata = metadata
        self.df = metadata.copy()
        # Define a common frequency grid for all images using the specified maximum frequency and frequency-bin width.
        self.freq_edges = np.arange(0, freq_max + freq_step, freq_step)
        self.freq = (self.freq_edges[:-1] + self.freq_edges[1:]) / 2

    def beam_profiles(self):
         # Calculate the spatial-frequency coordinates using the image's pixel spacing, and 
         # calculate the radial frequency of each FFT pixel.
        ny, nx = self.im.shape
        dx = (float(self.metadata['HFW']) * 1e6) / int(self.metadata['ResolutionX'])
        fx, fy = np.meshgrid(np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dx))
        radius = np.sqrt(fx**2 + fy**2)

        # Calculate the power contribution of each FFT pixel, accounting for the physical pixel area.
        freq_raw = radius.ravel()
        area = dx**2
        energy_raw = self.im.ravel()**2 * area

        # Assign FFT pixels to frequency bins and exclude frequencies outside the common grid and/or 
        # above the image's Nyquist frequency.
        bin_index = np.digitize(freq_raw, self.freq_edges) - 1
        valid = ((freq_raw >= self.freq_edges[0]) & (freq_raw < self.freq_edges[-1]) & (freq_raw < 1 / (2 * dx)))
        bin_index = bin_index[valid]
        energy_raw = energy_raw[valid]
        freq_raw = freq_raw[valid]

        # Count the number of FFT pixels in each frequency bin.
        n_bins = len(self.freq)
        bin_counts = np.bincount(bin_index, minlength=n_bins)

        # Calculate the mean power per FFT pixel in each frequency bin.
        energy_sum = np.bincount(bin_index, weights=energy_raw, minlength=n_bins)
        energy = np.divide(energy_sum, bin_counts, out=np.full(n_bins, np.nan), where=bin_counts > 0)
        # Convert mean power back to RMS FFT amplitude.
        amp = np.sqrt(energy / area)
        # Center bins to exclude freq = 0 since the DC component has been removed already and this is not physical,.
        freq = (self.freq_edges[:-1] + self.freq_edges[1:]) / 2

        # Calculate bin center frequencies, excluding zero frequency because the DC component has already been 
        # removed and does not represent physical beam structure.
        return amp[1:], freq[1:], energy[1:], bin_counts[1:]

    def run(self):
        return self.beam_profiles()
    
class SampNorm():
    def __init__(self, im, metadata):
        self.im = im
        self.metadata = metadata
        self.df = metadata.copy()

    def SampleNormalized(self):
        self.df['Sample Normalized Amplitude'] = None
        self.df['Sample Normalized Energy'] = None
        
        for (sample, file, current), group in self.df.groupby(['Sample', 'File', 'Current']):
            # Each file contains a sequence of images acquired at a single site and beam current, 
            # beginning at focus (zero offset) and progressing through increasing defocus offsets. 
            # Thus identifies the focused image in each site/current group as the reference profile 
            # for normalizing the remaining images.
            focused_row = group[group['Labeled Offset'] == 0].iloc[0]
            focused_amp = np.asarray(focused_row['Amplitude'], dtype=float).ravel()
            focused_e = np.asarray(focused_row['Energy'], dtype=float).ravel()

            for idx, row in group.iterrows():
                # The image is described by the convolution of the sample point-spread function (PSF) 
                # and the beam PSF. Defocus adds a blur kernel to the focused image such that:
                #   Focused image:  sample PSF * beam PSF
                #   Defocused image: blur kernel * sample PSF * beam PSF
                # Dividing the defocused profile by the focused profile in frequency space approximately cancels 
                # the shared sample and beam contributions, isolating the frequency response of defocus in terms
                # of the image blur kernel.
                amp = np.asarray(row['Amplitude'], dtype=float).ravel()
                e = np.asarray(row['Energy'], dtype=float).ravel()

                amp_ratio = amp / focused_amp
                e_ratio = e / focused_e

                self.df.at[idx, 'Sample Normalized Amplitude'] = amp_ratio.tolist()
                self.df.at[idx, 'Sample Normalized Energy'] = e_ratio.tolist()

        return self.df

    def fit_Kf(self, fmax=1.5):
        self.df['K Fit Params'] = None

        d = self.df[self.df['Labeled Offset'] != 0]
        for _, g in d.groupby(['Sample', 'Current', 'Labeled Offset']):
            f = np.asarray(g.iloc[0]['Frequency'], dtype=float)
            K = np.mean(np.stack(g['Sample Normalized Energy']), axis=0)
            ok = (f <= fmax) & np.isfinite(K)

            # f0 is the frequency where K has fallen to e^(−1/2) of A, so it measures how wide K is in frequency space.
            # A strongly blurred image kills high frequencies, so K is narrow and f0 is small. As blur shrinks toward 
            # zero, K gets wider, so f0 grows and goes to infinity when there is no blur at all. Fitting f0 near 
            # focus means the optimizer is chasing a parameter that runs off to infinity. Instead, fitting s = 1 /f0^2
            # lets s -> 0 as f0 -> inf, thereby allowing the optimizer to fit near focus.
            # We fix the Gaussian amplitude at 1.0 since focus / focus = 1.0 and since K(0) = 1 by convention
            # and because a common amplitude is required for an apples to apples comparison of how the Gaussian 
            # std changes with voltage offset. Also, σ_f = FWHM / (2√(2 ln 2)) ≈ FWHM / 2.3548
            # where σ_f = f0 = 1/√s. Since s = 1/σ_f², the initial guess for s is  s₀ = 8 ln 2 / FWHM².
            half = K[ok] <= 0.5 * K[ok].max()
            f_half = f[ok][half][0] if half.any() else f[ok].max()
            s0 = 8 * np.log(2) / (2 * f_half)**2
            
            popt, _ = curve_fit(lambda f, s: 1.0 * np.exp(-s * f**2 / 2), f[ok], K[ok],
                               p0=[s0], sigma=K[ok], bounds=((0), (np.inf)))

            for idx in g.index:
                self.df.at[idx, 'K Fit Params'] = popt.tolist()

    def fit_s_vs_V(self):
        self.df['sV Fit Params'] = None

        d = self.df.dropna(subset=['K Fit Params']).drop_duplicates(['Sample', 'Current', 'Labeled Offset']).copy()
        d['s'] = d['K Fit Params'].str[1]

        for current, g in d.groupby('Current'):
            m = g.groupby('Labeled Offset')['s'].mean()
            popt, _ = curve_fit(lambda V, a, b, c: a * V**2 + b * V + c, m.index.values, m.values)

            for idx in self.df.index[self.df['Current'] == current]:
                self.df.at[idx, 'sV Fit Params'] = popt.tolist()

    def run(self):
        self.df = self.SampleNormalized()
        self.fit_Kf()
        self.fit_s_vs_V()

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
        # plt.xlim(0, 10.)
        pp.set_titles('Current = {col_name}')
        pp.figure.subplots_adjust(top=0.84)
        pp.figure.text(0.5, 0.015, xlabel, ha='center', va='center')
        plt.suptitle(title, x=0.5, y=0.99, ha='center')
        return pp

    def plot_means(self, freq_col, val_col, ylabel, title, filename, drop_focus=False, log=True):
        rows = []
        for (current, offset), g in self.df.groupby(['Current', 'Labeled Offset']):
            f = np.mean(np.stack(g[freq_col]), axis=0)
            v = np.mean(np.stack(g[val_col]), axis=0)
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
                if row['Labeled Offset'] != 0:
                    freq = np.asarray(row['Frequency'], dtype=float)
                    energy = np.asarray(row['Sample Normalized Energy'], dtype=float)

                    n = min(len(freq), len(energy))

                    plot_freq.extend(freq[:n])
                    plot_energy.extend(energy[:n])
                    plot_offset.extend([row['Labeled Offset']] * n)
                    plot_current.extend([row['Current']] * n)
                    plot_trace_id.extend([str(row['Image'])] * n)

            plot_data = pd.DataFrame({'Frequency': plot_freq, 'Sample Normalized Energy': plot_energy,
                'Labeled Offset': plot_offset, 'Current': plot_current, 'Trace_ID': plot_trace_id})

            plot_data['Labeled Offset'] = pd.Categorical(plot_data['Labeled Offset'], 
                categories=offset_order, ordered=True)

            pp = sns.relplot(data=plot_data, kind='line', x='Frequency', y='Sample Normalized Energy', 
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

    def plot_denoise(self, n_samples,  random_state):
        d = self.df
        d = d.sample(n=min(n_samples, len(d)), random_state=random_state)

        ncols = min(3, len(d))
        nrows = int(np.ceil(len(d) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows),
                                 sharex=True, sharey=True, squeeze=False)

        for ax, (_, row) in zip(axes.ravel(), d.iterrows()):         
            f = np.asarray(row['Frequency'], dtype=float)
            raw = np.asarray(row['PreDenoised Energy'], dtype=float)
            clean = np.asarray(row['Energy'], dtype=float)
            N = np.asarray(row['Noise Floor'], dtype=float)

            ax.plot(f, raw, label='Original', color='darkturquoise')
            ax.plot(f, clean, label='Denoised', color='indigo')
            ax.plot(f, N, label='Noise', color='red', ls='--')
            ax.set_yscale('log')
            ax.set_title(f"{row['Sample']} File {row['File']}\n {row['Current']} pA, offset = {row['Labeled Offset']} V",
                         fontsize=8)
            ax.legend(fontsize=7)

        for ax in axes.ravel()[len(d):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel('Power')
        fig.suptitle('Denoised Profiles')
        fig.supxlabel('Spatial Frequency (cycles/µm)')
        fig.tight_layout()
        plt.savefig(self.output_path + f'/DenoisedSample.jpeg', bbox_inches='tight')
        plt.close(fig) 

    def plot_samples(self, n_samples, random_state):
        rng = np.random.default_rng(random_state)
        pairs = self.df[['Sample', 'File']].drop_duplicates()
        pairs = pairs.sample(n=min(n_samples, len(pairs)), random_state=random_state)

        for sample, file in pairs.itertuples(index=False):
            d = self.df[(self.df['Sample'] == sample) & (self.df['File'] == file)]
            offset = rng.choice(d.loc[d['Labeled Offset'] != 0, 'Labeled Offset'].unique())

            parts = [(d[d['Labeled Offset'] == 0], 'Amplitude','Focused'),
                     (d[d['Labeled Offset'] == offset], 'Amplitude', f'{offset} V'),
                     (d[d['Labeled Offset'] == offset], 'Sample Normalized Amplitude', f'{offset} V Samp Norm')]

            freq, amp, label, current = [], [], [], []
            for sub, col, name in parts:
                for _, row in sub.iterrows():
                    f = row['Frequency']
                    freq.extend(f)
                    amp.extend(row[col])
                    label.extend([name] * len(f))
                    current.extend([row['Current']] * len(f))

            pp = self.profile_plot(freq=freq, amp=amp, sample=label, current=current,
                                   ylabel='Magnitude', title=f'{sample} File: {file}')
            pp.set(yscale='log')
            pp.savefig(self.output_path + f'/Sample_{sample}_{file}.jpeg', bbox_inches='tight')
            plt.close(pp.figure)

    def plot_K_params(self):
        d = self.df.dropna(subset=['K Fit Params']).drop_duplicates(['Sample', 'Current', 'Labeled Offset']).copy()
        d['s'] = d['K Fit Params'].str[1]

        for p in ['s']:
            pp = sns.relplot(data=d, kind='line', x='Labeled Offset', y=p, col='Current',
                            marker='o', errorbar='sd',  palette=['lightblue'])
            pp.set_xlabels('Labeled Offset (V)')
            pp.set_ylabels(p)
            pp.set_titles('Current = {col_name}')
            pp.figure.subplots_adjust(top=0.84)
            pp.figure.suptitle(f'K(f) Gaussian Fit: {p} vs Offset')
            pp.savefig(self.output_path + f'/K_fit_{p}.jpeg', bbox_inches='tight')
            plt.close(pp.figure)

    def plot_K_fits(self, n_samples, random_state=None):
        d = self.df.dropna(subset=['K Fit Params'])
        groups = list(d.groupby(['Sample', 'Current', 'Labeled Offset']))
        rng = np.random.default_rng(random_state)
        picks = rng.choice(len(groups), size=min(n_samples, len(groups)), replace=False)

        ncols = min(3, len(picks))
        nrows = int(np.ceil(len(picks) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows),
                                sharex=True, squeeze=False)

        for ax, i in zip(axes.ravel(), picks):
            (sample, current, offset), g = groups[i]
            f = np.asarray(g.iloc[0]['Frequency'], dtype=float)
            K = np.mean(np.stack(g['Sample Normalized Energy']), axis=0)
            s = g.iloc[0]['K Fit Params']

            ax.plot(f, K, label='Mean K(f)', color='indigo')
            ax.plot(f, np.exp(-s * f**2 / 2), label=f'Fit: s={s:.2f}', color='red', ls='--')
            ax.set_title(f"{sample}\n {current} pA, offset = {offset} V", fontsize=8)
            ax.legend(fontsize=7)
            ax.set_ylim(0, 1.4)

        for ax in axes.ravel()[len(picks):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel('Sample Normalized Power')
        fig.suptitle('K(f) Gaussian Fits')
        fig.supxlabel('Spatial Frequency (cycles/µm)')
        fig.tight_layout()
        plt.savefig(self.output_path + '/K_fit_samples.jpeg', bbox_inches='tight')
        plt.close(fig)

    def plot_s_vs_V(self):
        d = self.df.dropna(subset=['K Fit Params']).drop_duplicates(['Sample', 'Current', 'Labeled Offset']).copy()
        d['s'] = d['K Fit Params'].str[1]

        pp = sns.relplot(data=d, kind='line', x='Labeled Offset', y='s', col='Current',
                         marker='o', errorbar='sd', palette=['lightblue'])

        V = np.linspace(d['Labeled Offset'].min(), d['Labeled Offset'].max(), 200)
        for current, ax in pp.axes_dict.items():
            a, b, c = self.df.loc[self.df['Current'] == current, 'sV Fit Params'].iloc[0]
            ax.plot(V, a * V**2 + b * V + c, color='red', ls='--', label=f'a={a:.4f}, b={b:.4f}, c={c:.2f}')
            ax.legend(fontsize=7)

        pp.set_xlabels('Labeled Offset (V)')
        pp.set_ylabels('s')
        pp.set_titles('Current = {col_name}')
        pp.figure.subplots_adjust(top=0.84)
        pp.figure.suptitle('s vs Offset with Quadratic Fit')
        pp.savefig(self.output_path + '/s_vs_V_fit.jpeg', bbox_inches='tight')
        plt.close(pp.figure)

    def run(self):

        # Plot means.
        self.plot_means('Frequency', 'Amplitude', 'Magnitude',
                        'Mean Spatial Frequency Distribution', '/Mean_Frequency_Profiles.jpeg')
        self.plot_means('Frequency', 'Energy', 'Power',
                        'Mean Power Distribution', '/Mean_Spatial_Energy.jpeg')
        self.plot_means('Frequency', 'Sample Normalized Amplitude', r'$\tilde{K}(f)$',
                        'Mean Sample Normalized Frequency Magnitude', '/Mean_samp_norm_amp.jpeg', drop_focus=True, log=False)   
        self.plot_means('Frequency', 'Sample Normalized Energy', r'$\tilde{K}(f)$',
                        'Mean Sample Normalized Power', '/Mean_samp_norm_energy.jpeg', drop_focus=True, log=False)

        # # Plot sampled profiles.
        # offset_order = sorted(self.df.loc[self.df['Labeled Offset'] != 0, 'Labeled Offset'].dropna().unique())
        # self.plot_sample_norm_amp_profiles(offset_order, n_samples=3)
        # self.plot_sample_norm_energy_profiles(offset_order, n_samples=3)
        # self.plot_samples(n_samples=3, random_state=None)
        # self.plot_denoise(n_samples=6, random_state=None)
        self.plot_K_fits(n_samples=6, random_state=None)

        # Plot models
        self.plot_K_params()
        self.plot_s_vs_V()


##### main execution block #####
images = []
metadata_records = []
for folder in os.listdir(path + inputs):
    if '.DS_Store' in folder:
        continue

    sample = folder.removeprefix('Offsets_')

    for file in os.listdir(path + inputs + '/' + folder):
        if '.DS_Store' in file:
            continue

        folder_path = path + inputs + '/' + folder + '/' + file

        for image in os.listdir(folder_path):
            if image.endswith('.tif'):
                image_path = folder_path + '/' + image
                processor = PrepareImages(image_path, file, sample)

                images.append((processor, sample, file, image))
                metadata_records.append(processor.extract_metadata())

freq_max, max_step = min_nyquist(pd.DataFrame(metadata_records))

##### process images #####
meta = []
for processor, sample, file, image in tqdm(images, desc='Processing images'):
    os.makedirs(path + outputs + '/' + sample + '/' + file, exist_ok=True)

    metadata, im, im_fft = processor.run()
    amp, freq, energy, bin_counts = BeamStats(im, metadata.iloc[0], freq_max, max_step).run()

    metadata['Amplitude'] = [amp.tolist()]
    metadata['Frequency'] = [freq.tolist()]
    metadata['Energy'] = [energy.tolist()]
    metadata['Energy Bin Counts'] = [bin_counts.tolist()]
    meta.append(metadata)

    '''
    # Save FFT per image.
    image_name = image.removesuffix('.tif')
    cv2.imwrite(path + outputs + '/' + sample + '/' + file + '/' + image_name + '.jpeg',
                cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    '''
##### aggregate metadata #####
all_metadata = pd.concat(meta, ignore_index=True)

# #### cleaning data #####
# print('Cleaning data.')
# all_metadata = Denoise(all_metadata)

##### sasmple normalizing #####
print('Sample normalizing.')
all_metadata = SampNorm(None, all_metadata).run()

##### save metadata #####
all_metadata.to_csv(path + outputs + '/all_meta_data.csv', index=False)

##### plot #####
print('Plotting.')
BeamPlotter(metadata_df = all_metadata, output_path = path + outputs).run()
