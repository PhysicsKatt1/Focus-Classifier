##### import #####
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import os.path
from datetime import datetime
import keras
import pandas as pd
from keras.saving import register_keras_serializable
from tqdm import tqdm
from PIL import Image
from keras import ops

##### globals #####
width = 512
height = 442
epochs = 5
batch = 3
learning_rate = 10e-4
model_name = 'UNet_3.1'
path = r'/Users/trentstarkey/Desktop'
sorted_path = path + '/TrainingData/30000.0V_0.09nA'
sorted_path1 = path + '/ValData/30000.0V_0.09nA'
test_path = path + '/TestData/Multifiducial_variableHFW/SortedJpegs'

##### define functions #####
class CreateDatasets():    
    def create_datasets(self):
        img_training = tf.keras.utils.image_dataset_from_directory(sorted_path,
            labels='inferred',
            color_mode='grayscale',
            seed=1,
            validation_split=0.2,
            subset='training',
            shuffle=True,
            image_size=(height, width),
            batch_size=batch, 
            label_mode = 'categorical')

        img_val = tf.keras.utils.image_dataset_from_directory(sorted_path,
            labels='inferred',
            color_mode='grayscale',
            seed=1,
            validation_split=0.2,
            subset='validation',
            shuffle=True,
            image_size=(height, width),
            batch_size=batch,
            label_mode = 'categorical')

        img_val1 = tf.keras.utils.image_dataset_from_directory(sorted_path1,
            labels='inferred',
            color_mode='grayscale',
            seed=1,
            validation_split=0.2,
            subset='validation',
            shuffle=True,
            image_size=(height, width),
            batch_size=batch,
            label_mode = 'categorical')

        return img_training, img_val, img_val1
    
    def optimize_data(self, img_training, img_val, img_val1):
        print('Optimizing datasets.')
        AUTOTUNE = tf.data.AUTOTUNE
        cache_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        img_training = img_training.cache().prefetch(AUTOTUNE)
        img_val = img_val.cache(f'/tmp/img_val_cache_{cache_id}').prefetch(AUTOTUNE)
        img_val1 = img_val1.cache(f'/tmp/img_val1_cache_{cache_id}').prefetch(AUTOTUNE)

        return img_training, img_val, img_val1

    def run(self):
        img_training, img_val, img_val1 = self.create_datasets()
        img_training, img_val, img_val1 = self.optimize_data(img_training, img_val, img_val1)

        return img_training, img_val, img_val1

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
        entropy = K.categorical_crossentropy(pred*K.exp(error*var), yTrain) 
        return entropy

class CreateModel():
    def class_num(self):
        focus_classes = set()

        for parent in os.listdir(sorted_path):
            parent_path = os.path.join(sorted_path, parent)

            for cls in os.listdir(parent_path):
                class_path = os.path.join(parent_path, cls)
                if os.path.isdir(class_path):
                    focus_classes.add(cls)

        num_labels = len(focus_classes)  

        return num_labels
    
    def create_model(self, num_labels):
        num_labels = num_labels
        inputs = tf.keras.Input(shape=(height, width, 1), batch_size=batch, name = 'image')
        # parents = tf.keras.Input(shape=(3,), name= 'parents')
        norm = tf.keras.layers.Rescaling(1 / 255., 0.)(inputs)

        x = layers.ZeroPadding2D((1, 1))(norm)
        x = layers.SeparableConv2D(256, (3, 3))(x)
        x = Exp_relu(leak = 0)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.BatchNormalization()(x)
    
        activation = x

        for filter in [256, 64, 8]:
            x = layers.ZeroPadding2D((1, 1))(x)
            x = layers.SeparableConv2D(filter, (3, 3))(x)
            x = Exp_relu(leak = 0)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.BatchNormalization()(x)

            res = layers.SeparableConv2D(filter, (3, 3))(activation)
            res = layers.Resizing(x.shape[1], x.shape[2])(res)
            x = layers.add([x, res])
            activation = x    

        for filter in [36]:
            x = layers.ZeroPadding2D((1, 1))(x)
            x = layers.Conv2DTranspose(filter, (3, 3))(x)
            x = Exp_relu(leak = 0)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D((2, 2))(x)

            res = layers.UpSampling2D((2, 2))(activation)
            res = layers.Conv2DTranspose(filter, (3, 3))(activation)
            res = layers.Resizing(x.shape[1], x.shape[2])(res)
            x = layers.add([x, res])
            activation = x    

        x = Patches(patch_size=4, resize_x=200, resize_y=200)(x) 

        for filter in [64]:
            x= layers.Dense(filter, activation='relu')(x)   

            res = layers.Dense(filter, activation='relu')(activation)
            res = layers.Resizing(x.shape[1], x.shape[2])(res)
            x = layers.add([x, res])
            
            activation = x  
    
        x = layers.Dropout(rate=0.05)(x)
        x = Img_shift()(x)

        for filter in [128]:
            x= layers.Dense(filter, activation='relu')(x)

            res = layers.Dense(filter, activation='relu')(activation)
            res = layers.Resizing(x.shape[1], x.shape[2])(res)
            x = layers.add([x, res])
            
            activation = x  

        x = Inv_img_shift()(x)

        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Flatten()(x)
        # x = layers.Concatenate()([x, parents])
        x= layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01))(x) 
        x= layers.Dense(1024, activation='relu')(x)
        output = layers.Dense(2, activation='softmax')(x)

        net = tf.keras.Model(inputs=inputs,outputs=output)

        return net

    def train_model_fit(self, model, training_data, val_data, val_data1):
        optimizer = tf.keras.optimizers.Lamb(learning_rate = learning_rate, beta_1=0.9, beta_2=0.5999, epsilon=1e-10)
        metric_accuracy = tf.keras.metrics.CategoricalAccuracy()
        metric_F1 = tf.keras.metrics.F1Score(threshold=None, average='weighted')
        metric_precision = tf.keras.metrics.Precision()
        metric_recall = tf.keras.metrics.Recall()

        all_val_data = tf.data.Dataset.concatenate(val_data, val_data1)

        model.compile(loss = Loss(), optimizer = optimizer, metrics = [metric_accuracy, metric_F1, metric_precision, metric_recall])
        history = model.fit(training_data, batch_size = batch,  validation_data = all_val_data.take(100), 
                            steps_per_epoch = 30, epochs = epochs, class_weight = {0: 0.7, 1: 0.3})
        
        model.save(path + '/' + model_name + '.keras')
        
        return history
    
    def run(self):
        num_labels = self.class_num()
        net = self.create_model(num_labels)
        history = self.train_model_fit(net, training_data, val_data, val_data1)

        return history


##### call fuctions ####
data = CreateDatasets()
training_data, val_data, val_data1 = data.run()

mod = CreateModel()
history = mod.run()
