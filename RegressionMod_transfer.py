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
width = 512
height = 442
epochs = 15
batch = 10
learning_rate = 1e-2
mod_name = 'Regression_mod_transfer_1.0'
tolerance = 3

##### define classes #####
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

    def labels(self, paths, df):
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        labels = df.loc[names, ['Defocus']].values.astype('float32')

        return labels

    def concat_data_labels(self, train_paths, val_paths, val1_paths, df_train, df_val1):
        labels_train = tf.data.Dataset.from_tensor_slices(self.labels(train_paths, df_train))
        labels_val = tf.data.Dataset.from_tensor_slices(self.labels(val_paths, df_train))
        labels_val1 = tf.data.Dataset.from_tensor_slices(self.labels(val1_paths, df_val1))

        train_dataset = tf.data.Dataset.zip((self.img_training, labels_train)).shuffle(len(train_paths),
            seed=1).batch(batch)
        val_dataset = tf.data.Dataset.zip((self.img_val, labels_val)).shuffle(len(val_paths),
            seed=1).batch(batch)
        val_dataset1 = tf.data.Dataset.zip((self.img_val1, labels_val1)).shuffle(len(val1_paths),
            seed=1).batch(batch)

        return train_dataset, val_dataset, val_dataset1

    def optimize_data(self, train_dataset, val_dataset, val_dataset1):
        print('Optimizing datasets.')
        AUTOTUNE = tf.data.AUTOTUNE
        cache_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
        val_dataset = val_dataset.cache(f'/tmp/img_val_cache_{cache_id}').prefetch(AUTOTUNE)
        val_dataset1 = val_dataset1.cache(f'/tmp/img_val1_cache_{cache_id}').prefetch(AUTOTUNE)

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
        x = model.layers[28].output

        x = tf.keras.layers.AveragePooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(1, activation='relu')(x)

        transfer_model = tf.keras.Model(model.inputs, output)

        return transfer_model

def get_label_bounds(*csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs)
    return combined['Defocus'].min(), combined['Defocus'].max()
 
class MinMaxLabelScaler:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
 
    def transform(self, labels):
        return (labels - self.min_val) / (self.max_val - self.min_val + 1e-7)
 
    def inverse_transform(self, scaled_labels):
        return scaled_labels * (self.max_val - self.min_val) + self.min_val

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, label_scaler, learning_rate,
        tolerance, loss_fn, log_dir='logs'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_scaler = label_scaler
        self.tolerance = tolerance
        self.loss_fn = loss_fn
        self.log_dir = log_dir
 
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    def train_epoch(self, epoch, epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0
        progress = tqdm(enumerate(self.train_dataset), desc=f'Training {epoch + 1}/{epochs}')

        for i, (images, raw_labels) in progress:
            scaled_labels = self.label_scaler.transform(raw_labels)

            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = self.loss_fn(scaled_labels, predictions)

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            predictions_raw = self.label_scaler.inverse_transform(predictions)
            total_correct += tf.reduce_sum(tf.cast(
                tf.abs(predictions_raw - raw_labels) <= self.tolerance, tf.int32)).numpy()
            total_samples += raw_labels.shape[0]
            total_loss += loss.numpy()

        num_batches = i + 1

        return total_loss / num_batches, total_correct / total_samples
 
    def validate(self, epoch, epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0
        progress = tqdm(enumerate(self.val_dataset), desc=f'Validation {epoch + 1}/{epochs}')
 
        for i, (images, raw_labels) in progress:
            scaled_labels = self.label_scaler.transform(raw_labels)
            predictions = self.model(images, training=False)
            loss = self.loss_fn(scaled_labels, predictions)
 
            predictions_raw = self.label_scaler.inverse_transform(predictions)
            total_correct += tf.reduce_sum(tf.cast(
                tf.abs(predictions_raw - raw_labels) <= self.tolerance, tf.int32)).numpy()
            total_samples += raw_labels.shape[0]
            total_loss += loss.numpy()
 
        num_batches = i + 1
 
        return total_loss / num_batches, total_correct / total_samples

    def fit(self, epochs, save_path):
        best_val_loss = np.inf
 
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(epoch, epochs)
            val_loss, val_accuracy = self.validate(epoch, epochs)
 
            print(f'Epoch [{epoch + 1}/{epochs}] ', f'Train Accuracy: {train_accuracy}',
                f'Train Loss: {train_loss:.6f}', f'Val Accuracy: {val_accuracy}', f'Val Loss: {val_loss:.6f}')
 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_weights(save_path)
                np.save(f'{save_path}_label_bounds.npy',
                    np.array([self.label_scaler.min_val, self.label_scaler.max_val]))
 
        return

class Test():
    def build_test_dataset(self, test_dir, csv_file, height, width, batch_size):
        df = pd.read_csv(csv_file)
        filepaths = [os.path.join(test_dir, f"{int(row['Image'])}.jpeg") for _, row in df.iterrows()]
        labels = df['Defocus'].values.astype('float32')
    
        def load_image(path):
            img = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img, channels=1)
            img = tf.image.resize(img, [height, width])
    
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
    train_dataset, val_dataset, val_dataset1 = cd.concat_data_labels(train_paths, 
                                        val_paths, val1_paths, df_train, df_val1)
    train_dataset, val_dataset, val_dataset1 = cd.optimize_data(train_dataset, val_dataset, val_dataset1)

    mod = Model()
    net = mod.transfer_learning()
    transfer_model = mod.new_model(net)

    defocus_min, defocus_max = get_label_bounds(csv_file)
    label_scaler = MinMaxLabelScaler(defocus_min, defocus_max)

    trainer = Trainer(model=transfer_model, train_dataset=train_dataset, val_dataset=val_dataset,
        label_scaler=label_scaler, learning_rate=learning_rate, tolerance=tolerance,
        loss_fn=WeightedLoss())
    trainer.fit(epochs = epochs, save_path=f'RegressionMod_{mod_name}.keras')

    test_mod = tf.keras.models.load_model(f'RegressionMod_{mod_name}.keras',
                custom_objects=CUSTOM_OBJECTS)
    defocus_min, defocus_max = get_label_bounds(csv_file_test)
    label_scaler = MinMaxLabelScaler(defocus_min, defocus_max)
    test = Test()
    test_data = test.build_test_dataset(test_dir, csv_file_test, height, width, batch)
    accuracy = test.predict_image(test_mod, test_data, label_scaler, tolerance)
    print(f'Total accuracy: {accuracy}')
    