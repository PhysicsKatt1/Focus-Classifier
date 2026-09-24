import tensorflow as tf
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
from keras.saving import register_keras_serializable
import keras

##### globals #####
path = r'/Users/trentstarkey/Desktop'
image_dir = path + '/RegressionData_30kV_0.09nA_train_defocusOnly'
val_dir = path + '/RegressionData_30kV_0.09nA_val_defocusOnly'
csv_file = image_dir + '/labels.csv'
csv_file_val = val_dir + '/labels.csv'
test_dir = path + '/RegressionData_30kV_0.09nA_test'
csv_file_test = test_dir + '/labels.csv'

year = '2025'
width = 518
height = 448
epochs = 100
batch = 30
learning_rate = 1e-3
mod_name = 'Regression_mod_transfer_1.0'
tolerance = 3

BLUR_SIGMA = 28
BLUR_TRUNCATE = 3.0  

##### define functions #####
def gaussian_ksize(sigma, truncate=BLUR_TRUNCATE):
    radius = int(truncate * sigma + 0.5)
    return 2 * radius + 1 

BLUR_KSIZE = gaussian_ksize(BLUR_SIGMA)

def gaussian_kernel_1d(size, sigma):
    ax = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
    kernel = tf.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel

_GAUSSIAN_1D = gaussian_kernel_1d(BLUR_KSIZE, BLUR_SIGMA)
_KERNEL_H = tf.reshape(_GAUSSIAN_1D, [1, BLUR_KSIZE, 1, 1])  # blur along width
_KERNEL_V = tf.reshape(_GAUSSIAN_1D, [BLUR_KSIZE, 1, 1, 1])  # blur along height

def gaussian_blur(image_4d):
    blur = tf.nn.conv2d(image_4d, _KERNEL_H, strides=[1, 1, 1, 1], padding='SAME')
    blur = tf.nn.conv2d(blur, _KERNEL_V, strides=[1, 1, 1, 1], padding='SAME')
    return blur

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
class WeightedLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
        weight = tf.stop_gradient(tf.abs(y_true - y_pred)) + 1.0
        return tf.reduce_mean(loss(y_true, y_pred) * weight)

CUSTOM_OBJECTS = {'Exp_relu': Exp_relu, 'Img_shift': Img_shift,
    'Inv_img_shift': Inv_img_shift, 'WeightedLoss': WeightedLoss, 'Patches': Patches}

def get_label_bounds(*csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs)
    return combined['Defocus'].min(), combined['Defocus'].max()

class MinMaxLabelScaler:
    def __init__(self, min_val, max_val):
        self.min_val = np.float32(min_val)
        self.max_val = np.float32(max_val)

    def transform(self, labels):
        labels = np.asarray(labels, dtype=np.float32)
        return (labels - self.min_val) / (self.max_val - self.min_val + 1e-7)

    def inverse_transform(self, scaled_labels):
        scaled_labels = tf.cast(scaled_labels, tf.float32)
        return scaled_labels * (self.max_val - self.min_val) + self.min_val

    def labels(self, paths, df):
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        labels = df.loc[names, ['Defocus']].values.astype('float32')
        return labels

def normalize(image, label):
    image = tf.image.resize(image, [height, width])
    image = image[3:-3, 3:-3, :]

    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-7)

    image_4d = tf.expand_dims(image, axis = 0)
    blur = gaussian_blur(image_4d)
    blur = tf.squeeze(blur, axis = 0)
    image = image - blur

    image = tf.squeeze(image, axis = -1)
    image = tf.signal.fft2d(tf.cast(image, tf.complex64))
    image = tf.signal.fftshift(image)

    image = image - tf.reduce_mean(image, axis=(-2, -1), keepdims=True)

    mag = tf.abs(image)
    scale = tf.reduce_max(mag) + 1e-7
    image = image / tf.cast(scale, image.dtype)

    image = tf.expand_dims(image, axis = -1)

    return image, label


class CreateDatasets():
    def create_datasets(self):
        self.img_training = tf.keras.utils.image_dataset_from_directory(image_dir,
            labels=None, color_mode='grayscale', seed=1, validation_split=0.2,
            subset='training', shuffle=False, image_size=(height, width), batch_size=None)

        self.img_val = tf.keras.utils.image_dataset_from_directory(image_dir,
            labels=None, color_mode='grayscale', seed=1, validation_split=0.2,
            subset='validation', shuffle=False, image_size=(height, width), batch_size=None)

        self.img_val1 = tf.keras.utils.image_dataset_from_directory(val_dir,
            labels=None, color_mode='grayscale', seed=1, validation_split=0.2,
            subset='validation', shuffle=False, image_size=(height, width), batch_size=None)

        train_paths = self.img_training.file_paths
        val_paths = self.img_val.file_paths
        val1_paths = self.img_val1.file_paths

        df_train = pd.read_csv(csv_file).set_index('Image')
        df_train.index = df_train.index.astype(str)
        df_val1 = pd.read_csv(csv_file_val).set_index('Image')
        df_val1.index = df_val1.index.astype(str)

        return train_paths, val_paths, val1_paths, df_train, df_val1

    def concat_data_labels(self, train_paths, val_paths, val1_paths, df_train, df_val1, label_scaler):
        labels_train = label_scaler.labels(train_paths, df_train)
        labels_val = label_scaler.labels(val_paths, df_train)
        labels_val1 = label_scaler.labels(val1_paths, df_val1)

        labels_train = label_scaler.transform(labels_train)
        labels_val = label_scaler.transform(labels_val)
        labels_val1 = label_scaler.transform(labels_val1)

        labels_train = tf.data.Dataset.from_tensor_slices(labels_train)
        labels_val = tf.data.Dataset.from_tensor_slices(labels_val)
        labels_val1 = tf.data.Dataset.from_tensor_slices(labels_val1)

        train_dataset = tf.data.Dataset.zip((self.img_training, labels_train)).shuffle(len(train_paths),
            seed=1).map(normalize).batch(batch)
        val_dataset = tf.data.Dataset.zip((self.img_val, labels_val)).shuffle(len(val_paths),
            seed=1).map(normalize).batch(batch)
        val_dataset1 = tf.data.Dataset.zip((self.img_val1, labels_val1)).shuffle(len(val1_paths),
            seed=1).map(normalize).batch(batch)

        return train_dataset, val_dataset, val_dataset1


    def optimize_data(self, train_dataset, val_dataset, val_dataset1):
        print('Optimizing datasets.')
        AUTOTUNE = tf.data.AUTOTUNE

        train_dataset = train_dataset.prefetch(AUTOTUNE)
        val_dataset = val_dataset.prefetch(AUTOTUNE)
        val_dataset1 = val_dataset1.prefetch(AUTOTUNE)

        return train_dataset, val_dataset, val_dataset1

class Model():
    def transfer_learning(self):
        print('Loading model.')
        net = tf.keras.models.load_model('UNet_3.1.keras', custom_objects= CUSTOM_OBJECTS, compile = False)

        net.trainable = False
        # for layer in range(0, 27):
        #     net.layers[layer].trainable = False
        # net.layers[33].trainable = False

        return net

    def new_model(self, model):
        x = model.layers[50].output

        x = tf.keras.layers.AveragePooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu', name='Dense2')(x)
        x = layers.Dense(128, activation='relu', name='Dense3')(x)
        output = layers.Dense(1, name='Output')(x)

        transfer_model = tf.keras.Model(model.inputs, output)

        return transfer_model

def make_tolerance_accuracy(label_scaler, tolerance):
    def tolerance_accuracy(y_true, y_pred):
        predictions_raw = label_scaler.inverse_transform(y_pred)
        labels_raw = label_scaler.inverse_transform(y_true)

        hit = tf.abs(predictions_raw - labels_raw) <= tolerance

        return tf.reduce_mean(tf.cast(hit, tf.float32))
    tolerance_accuracy.__name__ = 'tolerance_accuracy'

    return tolerance_accuracy

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, val_dataset1, learning_rate,
        tolerance, label_scaler):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_dataset1 = val_dataset1
        self.tolerance = tolerance
        self.label_scaler = label_scaler
        self.all_val_data = tf.data.Dataset.concatenate(self.val_dataset, self.val_dataset1)

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate,  clipnorm=1.0)
        self.weighted_loss = WeightedLoss()

        self.model.compile(optimizer=self.optimizer, loss = self.weighted_loss,
                           metrics=[tf.keras.metrics.MeanMetricWrapper(
                               make_tolerance_accuracy(self.label_scaler, self.tolerance),
                                name='tolerance_accuracy')])

    def fit(self):
        history = self.model.fit(self.train_dataset, validation_data =  self.all_val_data,
            epochs = epochs)

        self.model.save(f'RegressionMod_{mod_name}.keras')

        return history

class Test():
    def build_test_dataset(self, test_dir, csv_file, height, width, batch_size):
        df = pd.read_csv(csv_file)
        filepaths = [os.path.join(test_dir, f"{int(row['Image'])}.jpeg") for _, row in df.iterrows()]
        labels = df['Defocus'].values.astype('float32')

        def load_image(path):
            img = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img, channels=1)

            img = tf.image.resize(img, [height, width])
            img = img[3:-3, 3:-3, :]

            img = tf.cast(img, tf.float32) / 255.0
            mean = tf.reduce_mean(img)
            std = tf.math.reduce_std(img)
            img = (img - mean) / (std + 1e-7)

            return img

        path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        return tf.data.Dataset.zip((image_ds, label_ds)).batch(batch_size)

    def predict_image(self, model, test_dataset, label_scaler, tolerance):
        correct, samples = 0, 0
        progress = tqdm(test_dataset, desc='Test')

        for images, raw_labels in progress:
            predictions = model(images, training=False)
            predictions_raw = label_scaler.inverse_transform(predictions)

            hit = tf.abs(predictions_raw - raw_labels) <= tolerance
            correct += tf.reduce_sum(tf.cast(hit, tf.int32)).numpy()
            samples += raw_labels.shape[0]

        return correct / samples

if __name__ == "__main__":
    cd = CreateDatasets()
    train_paths, val_paths, val1_paths, df_train, df_val1 = cd.create_datasets()
    defocus_min, defocus_max = get_label_bounds(csv_file)
    label_scaler = MinMaxLabelScaler(defocus_min, defocus_max)
    train_dataset, val_dataset, val_dataset1 = cd.concat_data_labels(train_paths,
                                        val_paths, val1_paths, df_train, df_val1, label_scaler)
    train_dataset, val_dataset, val_dataset1 = cd.optimize_data(train_dataset, val_dataset, val_dataset1)


    mod = Model()
    net = mod.transfer_learning()

    for i, layer in enumerate(net.layers):
        try:
            shape = layer.output_shape
        except AttributeError:
            shape = "n/a"
        print(f"{i:3d}  {layer.name:30s}  {layer.__class__.__name__:20s}  "
            f"trainable={layer.trainable}  out_shape={shape}")

    transfer_model = mod.new_model(net)

    trainer = Trainer(model=transfer_model, train_dataset=train_dataset, val_dataset=val_dataset,
         val_dataset1 = val_dataset1, learning_rate=learning_rate, tolerance=tolerance,
         label_scaler=label_scaler)
    trainer.fit()

    test_mod = tf.keras.models.load_model(f'RegressionMod_{mod_name}.keras',
                custom_objects=CUSTOM_OBJECTS)
    test_defocus_min, test_defocus_max = get_label_bounds(csv_file_test)
    test_label_scaler = MinMaxLabelScaler(test_defocus_min, test_defocus_max)
    test = Test()
    test_data = test.build_test_dataset(test_dir, csv_file_test, height, width, batch)
    accuracy = test.predict_image(test_mod, test_data, test_label_scaler, tolerance)
    print(f'Total accuracy: {accuracy}')