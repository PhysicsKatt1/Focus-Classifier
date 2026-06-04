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
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import os.path
import keras
import pandas as pd
from keras.saving import register_keras_serializable
from tqdm import tqdm
from PIL import Image

##### globals #####
##### globals #####
year = '2025'
width = 512
height = 442
epochs = 5
batch = 3
title = ''
learning_rate = 10e6
path = r'/media/kat/d19098b8-7064-4621-8345-a76510e141a7/ThruFocusData/30kV90pA'
test_path = path + '/TestData/Multifiducial_variableHFW/SortedJpegs'
# test_path = path + '/TestData/Showerdrains_1535HFW/SortedJpegs'

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

def hard_vote():
    optimizer = tf.keras.optimizers.Lamb(learning_rate = learning_rate, beta_1=0.9, beta_2=0.5999, epsilon=1e-10)

    test_accuracy_list_in = []
    test_accuracy_list_not = []

    print('Loading model 1.')
    loaded_model1 = tf.keras.models.load_model(
        r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.1.keras')
    loaded_model1.compile(optimizer=optimizer, loss=Loss())

    print('Loading model 2.')
    loaded_model2 = tf.keras.models.load_model(
         r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.2.keras')
    loaded_model2.compile(optimizer=optimizer, loss=Loss())

    print('Loading model 3.')
    loaded_model3 = tf.keras.models.load_model(
         r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.3.keras')
    loaded_model3.compile(optimizer=optimizer, loss=Loss())

    # print('Loading model 4.')
    # loaded_model4 = tf.keras.models.load_model(
    #     r'/home/kitty/PycharmProjects/TR_ThruFocusSweeps/2ClassesPerBeamModel/TrainedModels/NotExtended_SingleInput/Arch6.keras',
    #     custom_objects={'exponential_relu': exponential_relu, 'fft_shift': fft_shift, 'ifft_shift': ifft_shift})
    # loaded_model4.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print('Making in focus predictions')
    pred_list_in1 = []
    pred_list_in2 = []
    pred_list_in3 = []
    pred_list_in4 = []
    for image, n in zip(os.listdir(test_path + '/In_focus'), tqdm(range(len(os.listdir(test_path + '/In_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/In_focus' + '/' + image))

        pred_in1 = np.argmax(loaded_model1.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_in1.append(pred_in1)

        pred_in2 = np.argmax(loaded_model2.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_in2.append(pred_in2)

        pred_in3 = np.argmax(loaded_model3.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_in3.append(pred_in3)

        # pred_in4 = np.argmax(loaded_model4.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        # pred_list_in4.append(pred_in4)

    pred_in = []
    for p1, p2, p3 in zip(pred_list_in1, pred_list_in2, pred_list_in3):
        pred = (p1+p2+p3)/3.
        pred_in.append(pred)

    df_in = pd.DataFrame({'pred_in': pred_in, 'pred1_in': pred_list_in1, 'pred2_in': pred_list_in2, 'pred3_in': pred_list_in3})
    df_in.to_csv(r'/home/kat/Data/in.csv')

    print('Making not in focus predictions')
    pred_list_not1 = []
    pred_list_not2 = []
    pred_list_not3 = []
    pred_list_not4 = []
    for image, n in zip(os.listdir(test_path + '/Not_in_focus'), tqdm(range(len(os.listdir(test_path + '/Not_in_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/Not_in_focus' + '/' + image))

        pred_not1 = np.argmax(loaded_model1.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_not1.append(pred_not1)

        pred_not2 = np.argmax(loaded_model2.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_not2.append(pred_not2)

        pred_not3 = np.argmax(loaded_model3.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_list_not3.append(pred_not3)

        # pred_not4 = np.argmax(loaded_model4.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        # pred_list_not4.append(pred_not4)

    pred_not = []
    for p1, p2, p3 in zip(pred_list_not1, pred_list_not2, pred_list_not3):
        pred = (p1+p2+p3) /3.
        pred_not.append(pred)

    df_not = pd.DataFrame({'pred_not': pred_not, 'pred1_not ': pred_list_not1, 'pred2_not': pred_list_not2, 'pred3_not ': pred_list_not3})
    df_not.to_csv(r'/home/kat/Data//not.csv')

    test_accuracy_list_in.append(((len(list(filter(lambda x: x <= 1/3., pred_in))) / len(pred_in))) * 100)
    test_accuracy_list_not.append((len(list(filter(lambda x: x > 1/3., pred_not))) / len(pred_not)) * 100)
    print('In focus test accuracy:', test_accuracy_list_in, 'Not in focus test accuracy:', test_accuracy_list_not)

    test_accuracy_list_in = np.asarray(test_accuracy_list_in)
    test_accuracy_list_not = np.asarray(test_accuracy_list_not)

    eval_df = pd.DataFrame({'In focus Evaluation Accuracy {}'.format(title): test_accuracy_list_in,
    'Not In focus Evaluation Accuracy {}'.format(title): test_accuracy_list_not})
    eval_df.to_csv(r'/home/kat/Data//LossAndAccuracy_eval_HardEnsemble.csv')

    return test_accuracy_list_in, test_accuracy_list_not

def soft_vote():
    test_accuracy_list_in = []
    test_accuracy_list_not = []

    optimizer = tf.keras.optimizers.Lamb(learning_rate = learning_rate, beta_1=0.9, beta_2=0.5999, epsilon=1e-10,)

    print('Loading model 1.')
    loaded_model1 = tf.keras.models.load_model(
        r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.1.keras')
    loaded_model1.compile(optimizer=optimizer, loss=Loss())

    print('Loading model 2.')
    loaded_model2 = tf.keras.models.load_model(
         r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.2.keras')
    loaded_model2.compile(optimizer=optimizer, loss=Loss())

    print('Loading model 3.')
    loaded_model3 = tf.keras.models.load_model(
         r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.3.keras')
    loaded_model3.compile(optimizer=optimizer, loss=Loss())

    # print('Loading model 4.')
    # loaded_model4 = tf.keras.models.load_model(
    #     r'/home/kitty/PycharmProjects/TR_ThruFocusSweeps/2ClassesPerBeamModel/TrainedModels/NotExtended_SingleInput/Arch6.keras',
    #     custom_objects={'exponential_relu': exponential_relu, 'fft_shift': fft_shift, 'ifft_shift': ifft_shift})
    # loaded_model4.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # make in focus predictions
    pred1_in = []
    pred2_in = []
    pred3_in = []
    pred4_in = []
    pred_list_in = []
    for image, n in zip(os.listdir(test_path + '/In_focus'), tqdm(range(len(os.listdir(test_path + '/In_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/In_focus' + '/' + image))

        pred1_in.append(loaded_model1.predict(image_to_predict[None, :, :], verbose='2'))
        pred2_in.append(loaded_model2.predict(image_to_predict[None, :, :], verbose='2'))
        pred3_in.append(loaded_model3.predict(image_to_predict[None, :, :], verbose='2'))
        # pred4_in.append(loaded_model4.predict(image_to_predict[None, :, :], verbose='2'))

        pred_in = pred1_in + pred2_in + pred3_in
        for p in pred_in:
            pred_list_in.append(np.argmax(p))

    pred1_not = []
    pred2_not = []
    pred3_not = []
    pred4_not = []
    pred_list_not = []
    for image, n in zip(os.listdir(test_path + '/Not_in_focus'), tqdm(range(len(os.listdir(test_path + '/Not_in_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/Not_in_focus' + '/' + image))

        pred1_not.append(loaded_model1.predict(image_to_predict[None, :, :], verbose='2'))
        pred2_not.append(loaded_model2.predict(image_to_predict[None, :, :], verbose='2'))
        pred3_not.append(loaded_model3.predict(image_to_predict[None, :, :], verbose='2'))
        # pred4_not.append(loaded_model4.predict(image_to_predict[None, :, :], verbose='2'))

        pred_not = pred1_not + pred2_not + pred3_not
        for p in pred_not:
            pred_list_not.append(np.argmax(p))


    test_accuracy_list_in.append(((len(list(filter(lambda x: x == 0, pred_list_in))) / len(pred_list_in))) * 100)
    test_accuracy_list_not.append(((len(list(filter(lambda x: x != 0, pred_list_not))) / len(pred_list_not))) * 100)

    print('In focus test accuracy:', test_accuracy_list_in, 'Not in focus test accuracy:', test_accuracy_list_not)
    soft_pred_df = pd.DataFrame({'In focus': test_accuracy_list_in, 'Not in focus': test_accuracy_list_not})
    soft_pred_df.to_csv(r'/home/kat/Data/LossAndAccuracy_eval_SoftEnsemble.csv')


    return test_accuracy_list_in, test_accuracy_list_not, soft_pred_df

# hard_vote()
soft_vote()
