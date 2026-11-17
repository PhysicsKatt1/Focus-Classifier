##### imports #####
import os
import sys
import time
import shutil
import subprocess
import traceback
import warnings
import configparser
from io import StringIO
from pathlib import Path
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from PIL import Image
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import messagebox

##### config #####
# config.xml is expected next to this script unless a path is passed:  python Initiate.py /path/to/config.xml
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'config.xml'
config = ET.parse(CONFIG_PATH).getroot()
BASE = Path(config.find('InputPaths/Path1').text.strip()).expanduser()
PARENT_NAMES = [p.text.strip() for p in config.find('Parents')]
NAME_TO_ID = {n: i for i, n in enumerate(PARENT_NAMES)}
NUM_PARENTS = len(PARENT_NAMES)

# Filename tokens that identify each beam condition, in the same order as <Parents> in config.xml.
BEAM_TOKENS = ['30000.0V_0.09nA', '30000.0V_0.041nA', '30000.0V_0.75nA']

# Folder layout (relative to BASE). Edit here if your structure differs.
INPUT_DIR = BASE / 'Inputs'
OUTPUT_DIR = BASE / 'Outputs'
FOCUSED_DIR = OUTPUT_DIR / 'Focused'            # Outputs/Focused/<beam condition>/
DEFOCUSED_DIR = OUTPUT_DIR / 'Defocused'        # Outputs/Defocused/<beam condition>/
UNMATCHED_DIR = OUTPUT_DIR / 'Unmatched'        # images whose filename matches no beam condition
ERROR_DIR = OUTPUT_DIR / 'Errors'               # images that raised an exception
MODEL_PATH = BASE / 'UNet_3.1.keras'
RUN_CSV = BASE / 'runData.csv'
FOCAL_FIT_CSV = BASE / 'focal_point_vs_dV_fit_stats.csv'    # output of the analysis script

WD_UM = 4030.        # working distance used by the focal-point model in the analysis script
POLL_SECONDS = 1.0
IMG_EXT = '.tif'

for d in [INPUT_DIR, FOCUSED_DIR, DEFOCUSED_DIR, UNMATCHED_DIR, ERROR_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for p in PARENT_NAMES:
    (FOCUSED_DIR / p).mkdir(exist_ok=True)
    (DEFOCUSED_DIR / p).mkdir(exist_ok=True)

##### custom keras objects #####
keras.saving.get_custom_objects().clear()
@keras.saving.register_keras_serializable()
class Exp_relu(layers.Layer):
    def __init__(self, leak):
        super(Exp_relu, self).__init__()
        self.leak = leak

    def call(self, input):
        return K.maximum(input * tf.exp(input), self.leak)

    def get_config(self):
        return {'leak': self.leak}

@keras.saving.register_keras_serializable()
class Img_shift(layers.Layer):
    def call(self, input):
        return tf.signal.fftshift(input)

@keras.saving.register_keras_serializable()
class Inv_img_shift(layers.Layer):
    def call(self, input):
        return tf.signal.ifftshift(input)

@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, resize_x, resize_y):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.resize_x = resize_x
        self.resize_y = resize_y

    def call(self, input):
        resized = layers.Resizing(self.resize_x, self.resize_y)(input)
        return keras.ops.image.extract_patches(resized, size=self.patch_size)

    def get_config(self):
        return {'patch_size': self.patch_size, 'resize_x': self.resize_x, 'resize_y': self.resize_y}

@keras.saving.register_keras_serializable()
class Loss(tf.keras.losses.Loss):
    def call(self, yTrain, pred):
        error = yTrain - pred
        var = tf.math.reduce_variance(pred)
        entropy = K.categorical_crossentropy(pred * K.exp(error * var), yTrain)
        return entropy

##### popup window  #####
def output(message):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo("Defocus", message)
        root.destroy()
    except Exception:
        # Fallback to a native macOS dialog if tkinter is unavailable.
        safe = message.replace('\\', '\\\\').replace('"', '\\"')
        subprocess.run(['osascript', '-e', f'display dialog "{safe}" with title "Defocus" buttons {{"OK"}}'])

##### preprocessing #####
def read_tiff_meta(path):
    with Image.open(path) as im:
        ini_text = im.tag_v2.get(34682)
        xml_root = ET.fromstring(im.tag_v2.get(34683))
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read_file(StringIO(ini_text))
    return {'HFW': float(cfg['IBeam']['HFW']),
            'ResolutionX': int(cfg['Image']['ResolutionX']),
            'DwellTime': float(cfg['Scan']['Dwelltime']),
            'Voltage': float(cfg['IBeam']['HV']),
            'Current': float(cfg['IBeam']['BeamCurrent']),
            'L2': float(xml_root.find('Optics/FibL2Voltage').text)}

def labeled_offset_from_name(name):
    # Same parsing as the analysis script: text after 'data_', second '__' field.
    try:
        stem = os.path.splitext(name)[0]
        parts = stem.split('data_')
        stem = parts[1] if len(parts) > 1 else parts[0]
        return float(stem.split('__')[1])
    except Exception:
        return np.nan

def periodic_component(image):
    # Removes boundary discontinuities before the FFT to limit spectral leakage.
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

def fft_magnitude(path, meta):
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    # normalise by dwell time
    arr = arr / meta['DwellTime']                     
    return np.abs(np.fft.fft2(periodic_component(arr))) / arr.size

def radial_energy(mag, meta, edges):
    ny, nx = mag.shape
    dx = meta['HFW'] * 1e6 / meta['ResolutionX']
    fx, fy = np.meshgrid(np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dx))
    radius = np.sqrt(fx**2 + fy**2).ravel()
    energy_raw = mag.ravel()**2 * dx**2
    idx = np.digitize(radius, edges) - 1
    valid = (radius >= edges[0]) & (radius < edges[-1]) & (radius < 1 / (2 * dx))
    idx, energy_raw = idx[valid], energy_raw[valid]
    n_bins = len(edges) - 1
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=energy_raw, minlength=n_bins)
    energy = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
    freq = (edges[:-1] + edges[1:]) / 2
    # drop the DC bin
    return energy[1:], freq[1:]                         

##### fitted models from the analysis script #####
def _row_for_current(table, current):
    # Pick the Current in the table closest (log scale) to this image's beam current, average over samples.
    cur = pd.to_numeric(table['Current'], errors='coerce')
    nearest = cur.iloc[(np.log(cur / current)).abs().argmin()]
    return table[np.isclose(cur, nearest)].mean(numeric_only=True)

def inverse_dz_to_dV(dz, m, q):
    """Invert dz = m*dV + q*dV^2, taking the root on the branch through dV = 0."""
    if abs(q) < 1e-12:
        return dz / m
    disc = m**2 + 4 * q * dz
    if disc < 0:                       # dz outside the range the fit can produce: use the vertex
        return -m / (2 * q)
    roots = np.array([(-m + np.sqrt(disc)) / (2 * q), (-m - np.sqrt(disc)) / (2 * q)])
    return float(roots[np.abs(roots).argmin()])

##### running mean of focused L2 per beam condition #####
class FocusState:
    def __init__(self):
        self.n = {p: 0 for p in PARENT_NAMES}
        self.total = {p: 0.0 for p in PARENT_NAMES}
        self.paused = {p: False for p in PARENT_NAMES}
        # Rebuild from images already in the Focused folders so the mean survives restarts.
        for p in PARENT_NAMES:
            for f in newest_first(FOCUSED_DIR / p):
                try:
                    self.n[p] += 1
                    self.total[p] += read_tiff_meta(f)['L2']
                except Exception:
                    self.n[p] -= 1

    def mean(self, p):
        return self.total[p] / self.n[p] if self.n[p] else np.nan

    def add_focused(self, p, L2):
        self.n[p] += 1                 # a focused image resumes the mean if it was paused
        self.total[p] += L2
        self.paused[p] = False

    def pause(self, p):
        self.paused[p] = True          # no updates until the next focused image

def newest_first(folder):
    files = [f for f in Path(folder).glob('*') if f.suffix.lower() in IMG_EXT and not f.name.startswith('.')]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

##### helpers #####
def beam_from_name(name):
    for parent, token in zip(PARENT_NAMES, BEAM_TOKENS):
        if token in name:
            return parent
    return None

def file_is_ready(path):
    s1 = path.stat().st_size
    time.sleep(0.5)
    return s1 > 0 and s1 == path.stat().st_size

def move_in(src, dest_dir):
    dest = Path(dest_dir) / src.name
    shutil.move(str(src), str(dest))
    os.utime(dest, None)               # mtime = arrival time, so "newest focused" is well defined
    return dest

def append_row(row):
    pd.DataFrame([row]).to_csv(RUN_CSV, mode='a', index=False, header=not RUN_CSV.exists())

##### main processing #####
def process_image(path, model, state, sigma_table, focal_table):
    name = path.name
    beam = beam_from_name(name)
    if beam is None:
        print(f'{name}: no matching beam condition, moved to Unmatched')
        move_in(path, UNMATCHED_DIR)
        return

    meta = read_tiff_meta(path)

    # --- classify (input prep identical to the original script) ---
    img_h, img_w = model.input_shape[0][1:3]
    x = tf.keras.utils.load_img(str(path), color_mode='grayscale', target_size=(img_h, img_w))
    x = np.expand_dims(tf.keras.utils.img_to_array(x) / 255.0, axis=0)
    parent_onehot = tf.one_hot([NAME_TO_ID[beam]], NUM_PARENTS)
    probs = model.predict({'img': x, 'parent': parent_onehot}, verbose=0)[0]
    pred = int(np.argmax(probs))       # 0 = focused, 1 = defocused

    true_defocused = 0 if '__0.0__0.0__0.0__' in name else 1
    mean_before = state.mean(beam)

    if pred == 0:
        # ---------- focused ----------
        move_in(path, FOCUSED_DIR / beam)
        was_paused = state.paused[beam]
        state.add_focused(beam, meta['L2'])
        row.update({'mean_focused_L2_after_V': state.mean(beam), 'n_focused': state.n[beam],
                    'mean_paused_before': was_paused})
        print(f'{name}: FOCUSED ({beam}), running mean L2 = {state.mean(beam):.4f} V (n={state.n[beam]})')
    else:
        # ---------- defocused ----------
        dest = move_in(path, DEFOCUSED_DIR / beam)
        state.pause(beam)              # running mean frozen until the next focused image
        row.update({'mean_focused_L2_after_V': state.mean(beam), 'n_focused': state.n[beam],
                    'mean_paused_before': True})
        print(f'{name}: DEFOCUSED ({beam}), running mean paused')

        refs = newest_first(FOCUSED_DIR / beam)
        if not refs or state.n[beam] == 0:
            output(f'{beam}\n{name}\n\nClassified as DEFOCUSED, but there is no focused image yet for this '
                   f'beam condition, so the defocus cannot be quantified.')
        else:
            foc_path = refs[0]
            foc_meta = read_tiff_meta(foc_path)
            L2_focus = state.mean(beam)

            # dz from the first-order Einzel-lens model used in the analysis script:
            # z = WD * (L2_focus / L2)^2, dz = z - WD.
            dz = WD_UM * ((L2_focus / meta['L2'])**2 - 1)
            fit = _row_for_current(focal_table, meta['Current'])
            dV = inverse_dz_to_dV(dz, fit['dz/dV (um/V)'], fit['q (um/V^2)'])

    row = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'image': name, 'beam_condition': beam,
               'pred': pred, 'pred_label': 'defocused' if pred else 'focused',
               'prob_focused': float(probs[0]), 'prob_defocused': float(probs[1]),
               'true_label': true_defocused, 'correct': 'Yes' if pred == true_defocused else 'No',
               'labeled_offset_V': labeled_offset_from_name(name), 'L2_V': meta['L2'],
               'mean_focused_L2_before_V': mean_before, 'Offset': dV}

    append_row(row)

if __name__ == "__main__":
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    focal_table = pd.read_csv(FOCAL_FIT_CSV)
    state = FocusState()

    while True:
        # Re-read the run flag each pass so setting <Run>False</Run> stops the loop.
        if ET.parse(CONFIG_PATH).getroot().find('Params/Run').text.strip() != 'True':
            break
        pending = sorted(f for f in INPUT_DIR.iterdir()
                         if f.is_file() and f.suffix.lower() in IMG_EXT and not f.name.startswith('.'))
        if not pending:
            time.sleep(POLL_SECONDS)
            continue
        for path in pending:
            try:
                if not file_is_ready(path):
                    continue
                process_image(path, model, state, focal_table)
            except Exception:
                traceback.print_exc()
                if path.exists():
                    move_in(path, ERROR_DIR)