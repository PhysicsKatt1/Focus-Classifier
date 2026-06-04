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
year = '2025'
width = 512
height = 442
epochs = 10
batch = 3
learning_rate = 10e-6
path = r'/media/kat/d19098b8-7064-4621-8345-a76510e141a7/ThruFocusData/30kV90pA'
# sorted_path = path + '/TrainingData/SortedJpegs_variableHFW_variableScanRes'
sorted_path = path + '/TrainingData/SortedJpegs_half'
sorted_path1 = path + '/ValData/SortedJpegs_variableHFW'
test_path = path + '/TestData/Multifiducial_variableHFW/SortedJpegs'
# test_path = path + '/TestData/Showerdrains_1535HFW/SortedJpegs'
# test_path = r'/media/kat/d19098b8-7064-4621-8345-a76510e141a7/ThruFocusData/MixedBeams/SortedJpegs'

training_loss_list = []
training_accuracy_list = []
training_F1_list = []
training_precision_list = []
training_recall_list = []
val_accuracy_list = []
val_loss_list = []
val_F1_list = []
val_precision_list = []
val_recall_list = []
test_accuracy_list_in = []
test_accuracy_list_not = []
test_F1_list = []
test_precision_list = []
test_recall_list = []

##### define functions #####
def creating_datasets():
    print('Creating training dataset.')
    img_training = tf.keras.utils.image_dataset_from_directory(sorted_path,
                                                               color_mode='grayscale', seed=1, validation_split=0.2,
                                                               subset='training', labels='inferred',
                                                               label_mode='categorical',
                                                               shuffle=True, image_size=(height, width),
                                                               batch_size=batch) 

    print('Creating validation dataset.')
    img_val = tf.keras.utils.image_dataset_from_directory(sorted_path,
                                                          color_mode='grayscale', seed=1, validation_split=0.2,
                                                          subset='validation', labels='inferred',
                                                          label_mode='categorical',
                                                          shuffle=True, image_size=(height, width), batch_size=batch)
                                                          

    print('Creating test dataset.')
    img_val1 = tf.keras.utils.image_dataset_from_directory(sorted_path1,
                                                           color_mode='grayscale', seed=1, labels='inferred',
                                                           label_mode='categorical', validation_split=0.4,subset='validation',
                                                           shuffle=True, image_size=(height, width), batch_size=batch)

    print('Optimizing datasets.')
    AUTOTUNE = tf.data.AUTOTUNE

    img_training = img_training.cache().shuffle(img_training.cardinality()).prefetch(buffer_size=AUTOTUNE)
    img_val = img_val.cache().shuffle(img_val.cardinality()).prefetch(buffer_size=AUTOTUNE)
    img_val1 = img_val1.cache().shuffle(img_val1.cardinality()).prefetch(buffer_size=AUTOTUNE)

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

# @keras.saving.register_keras_serializable()
# class Loss(tf.keras.losses.Loss):
#     def call(self, yTrain, pred):
#         error = K.sqrt(K.mean(K.square(pred - yTrain)))
#         entropy = K.categorical_crossentropy((pred + error), yTrain)
#         return entropy*K.exp(entropy)

@keras.saving.register_keras_serializable()
class Loss(tf.keras.losses.Loss):
    def call(self, yTrain, pred):
        error = yTrain - pred  
        var = tf.math.reduce_variance(pred)  
        entropy = K.categorical_crossentropy(pred*K.exp(error*var), yTrain) 
        return entropy
    
@keras.saving.register_keras_serializable()
class Encoder(layers.Layer):
    def __init__(self, n_patches, n_filters):
        super().__init__()
        self.n_patches = n_patches
        self.n_filters = n_filters
        self.projection = layers.Dense(units=n_filters)
        self.position_embedding = layers.Embedding(input_dim=n_patches, output_dim=n_filters)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.n_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        return {'n_patches': self.n_patches, 'n_filters': self.n_filters}

def create_model():
    num_labels = len(np.unique(os.listdir(sorted_path)))
    inputs = tf.keras.Input(shape=(height, width, 1), batch_size=batch)
    norm = tf.keras.layers.Rescaling(1 / 255., 0.)(inputs)
    x = layers.Resizing(442, 512)(norm)

    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.SeparableConv2D(256, (3, 3), activation='gelu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
   
    activation = x

    for filter in [256, 64, 8]:
        x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.SeparableConv2D(filter, (3, 3), activation='gelu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.BatchNormalization()(x)

        res = layers.SeparableConv2D(filter, (3, 3))(activation)
        res = layers.Resizing(x.shape[1], x.shape[2])(res)
        x = layers.add([x, res])
        activation = x    

    for filter in [36]:
        x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.Conv2DTranspose(filter, (3, 3), activation='gelu')(x)
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
    # x = Img_shift()(x)

    # for filter in [128]:
    #     x= layers.Dense(filter, activation='relu')(x)

    #     res = layers.Dense(filter, activation='relu')(activation)
    #     res = layers.Resizing(x.shape[1], x.shape[2])(res)
    #     x = layers.add([x, res])
        
    #     activation = x  

    # x = Inv_img_shift()(x)

    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x= layers.Dense(1024, activation='relu')(x)  #, kernel_regularizer=tf.keras.regularizers.l1_l2(0.1, 0.01)
    x= layers.Dense(512, activation='relu')(x)
    output = layers.Dense(num_labels, activation='softmax')(x)

    net = tf.keras.Model(inputs, output)

    return net

def train_model(training_data, val_data, val1_data, model):
    print('Beginning training.')

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metric_accuracy = tf.keras.metrics.CategoricalAccuracy()
    metric_F1 = tf.keras.metrics.F1Score(threshold=None, average='weighted')
    metric_precision = tf.keras.metrics.Precision()
    metric_recall = tf.keras.metrics.Recall()
    start_time = datetime.now()

    for epoch in range(epochs):
        for step, (xTrain, yTrain) in enumerate(training_data):
            with tf.GradientTape() as tape:
                training_output = model(xTrain, training=True)
                training_loss = loss(yTrain, training_output)

            gradient = tape.gradient(training_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradient, model.trainable_weights))
            metric_accuracy.update_state(yTrain, training_output)
            metric_F1.update_state(yTrain, training_output)
            metric_precision.update_state(yTrain, training_output)
            metric_recall.update_state(yTrain, training_output)

        training_accuracy = metric_accuracy.result()
        training_F1 = metric_F1.result()
        training_precision = metric_precision.result()
        training_recall = metric_recall.result()

        training_accuracy_list.append(training_accuracy)
        training_loss_list.append(training_loss)
        training_F1_list.append(training_F1)
        training_precision_list.append(training_precision)
        training_recall_list.append(training_recall)

        metric_accuracy.reset_state()
        metric_F1.reset_state()
        metric_precision.reset_state()
        metric_recall.reset_state()

        for (xVal, yVal), (xVal1, yVal1) in zip(val_data, val1_data):
            xVT = tf.concat([xVal, xVal1], axis=0)
            yVT = tf.concat([yVal, yVal1], axis=0)


            val_output = model(xVT, training=False)
            val_loss = loss(yVT, val_output)
            metric_accuracy.update_state(yVT, val_output)
            metric_F1.update_state(yVT, val_output)
            metric_precision.update_state(yVT, val_output)
            metric_recall.update_state(yVT, val_output)

        val_accuracy = metric_accuracy.result()
        val_F1 = metric_F1.result()
        val_precision = metric_precision.result()
        val_recall = metric_recall.result()

        val_accuracy_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        val_F1_list.append(val_F1)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)

        metric_accuracy.reset_state()
        metric_F1.reset_state()
        metric_precision.reset_state()
        metric_recall.reset_state()

        print('Epoch {}: '.format(epoch), 'Training Loss = {}, '.format(float(training_loss)),
              'Training Accuracy = {}, '.format(training_accuracy), 'Training F1 Score = {}, '.format(float(training_F1)),
              'Training Precision= {}, '.format(float(training_precision)), 'Training Recall = {}, '.format(float(training_recall)),
              'Validation Loss = {}, '.format(float(val_loss)), 'Validation Accuracy = {}, '.format(val_accuracy), 
              'Validation F1 Score = {}, '.format(float(val_F1)),
              'Validation Precision= {}, '.format(float(val_precision)), 'Validation Recall = {}, '.format(float(val_recall)))

    model.save( 
        r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/UNet_2.2.1.keras')
    end_time = datetime.now()
    print('Training and validation took {} s.'.format(end_time - start_time))

    return

def train_model_fit(model, training_data, val_data, val_data1):
    optimizer = tf.keras.optimizers.Lamb(learning_rate = learning_rate, beta_1=0.9, beta_2=0.5999, epsilon=1e-10)
    metric_accuracy = tf.keras.metrics.CategoricalAccuracy()
    metric_F1 = tf.keras.metrics.F1Score(threshold=None, average='weighted')
    metric_precision = tf.keras.metrics.Precision()
    metric_recall = tf.keras.metrics.Recall()

    all_val_data = tf.data.Dataset.concatenate(val_data, val_data1)

    model.compile(loss = Loss(), optimizer = optimizer, metrics = [metric_accuracy, metric_F1, metric_precision, metric_recall])
    history = model.fit(training_data, batch_size = batch,  validation_data = all_val_data, epochs = epochs, class_weight = {0: .6, 1: .4})
    
    model.save(
        r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/ExpRelu/UNet_elu.keras')
    
    return history

def test_model():
    print('Loading model.')
    optimizer = tf.keras.optimizers.Lamb(learning_rate = learning_rate, beta_1=0.9, beta_2=0.5999, epsilon=1e-10)    

    loaded_model = tf.keras.models.load_model(
        r'/home/kat/GitRepo/automated-tool-health-service/TR_ThruFocusSweeps/LinuxSystem/Trained_Models/CustomLoss_2xVal/ExpRelu/UNet_elu.keras')
    loaded_model.compile(loss = Loss(), optimizer = optimizer)

    print('Making in focus predictions')
    pred_list_in = []
    name_in = []
    for image, n in zip(os.listdir(test_path + '/In_focus'), tqdm(range(len(os.listdir(test_path + '/In_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/In_focus' + '/' + image))
        pred_in = np.argmax(loaded_model.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_in = np.asarray(pred_in)
        pred_list_in.append(pred_in)
        name_in.append(image)

    print('\n Making not in focus predictions.')
    pred_list_not = []
    name_not = []
    for image, n in zip(os.listdir(test_path + '/Not_in_focus'), tqdm(range(len(os.listdir(test_path + '/Not_in_focus'))))):
        image_to_predict = tf.convert_to_tensor(Image.open(test_path + '/Not_in_focus' + '/' + image))
        pred_not = np.argmax(loaded_model.predict(image_to_predict[None, :, :], verbose='2'), axis=1)
        pred_not = np.asarray(pred_not)
        pred_list_not.append(pred_not)
        name_not.append(image)

    df_in = pd.DataFrame({'pred':pred_list_in, 'image': name_in})
    # df_in.to_csv(r'/home/kat/Data/In_focus_pred.csv')
    df_not = pd.DataFrame({'pred': pred_list_not, 'image': name_not})
    # df_not.to_csv(r'/home/kat/Data/Not_in_focus_pred.csv')
    test_accuracy_list_in.append(((len(list(filter(lambda x: x == 0, pred_list_in))) / len(pred_list_in))) * 100)
    test_accuracy_list_not.append((len(list(filter(lambda x: x > 0, pred_list_not))) / len(pred_list_not)) * 100)
    print('In focus test accuracy:', test_accuracy_list_in, 'Not in focus test accuracy:', test_accuracy_list_not)


    return test_accuracy_list_in, test_accuracy_list_not

def main():
    img_training, img_val, img_val1 = creating_datasets()
    net = create_model()

    # train_model(img_training, img_val, img_val1, net)
    history = train_model_fit(net, img_training, img_val, img_val1)
    
    del img_training, img_val, img_val1
    gc.collect()

    return history 

##### call functions and save results for plotting #####
history = main()
test_model()

training_loss_list = history.history['loss']
training_accuracy_list = history.history['categorical_accuracy']
training_F1_list = history.history['f1_score']
training_recall_list = history.history['recall']
training_precision_list = history.history['precision']

val_loss_list = history.history['val_loss']
val_accuracy_list = history.history['val_categorical_accuracy']
val_F1_list = history.history['val_f1_score']
val_recall_list = history.history['val_recall']
val_precision_list = history.history['val_precision']

# training_loss_list = np.asarray(training_loss_list)
# training_accuracy_list = np.asarray(training_accuracy_list)
# training_F1_list = np.asarray(training_F1_list)
# training_recall_list = np.asarray(training_recall_list)
# training_precision_list = np.asarray(training_precision_list)

# val_loss_list = np.asarray(val_loss_list)
# val_accuracy_list = np.asarray(val_accuracy_list)
# val_F1_list = np.asarray(val_F1_list)
# val_recall_list = np.asarray(val_recall_list)
# val_precision_list = np.asarray(val_precision_list)

test_accuracy_list_in = np.asarray(test_accuracy_list_in)
test_accuracy_list_not = np.asarray(test_accuracy_list_not)
test_F1_list = np.asarray(test_F1_list)
test_recall_list = np.asarray(test_recall_list)
test_precision_list = np.asarray(test_precision_list)


model_df = pd.DataFrame({'Training Loss': training_loss_list,'Training Accuracy': training_accuracy_list, 'Training F1 Score': training_F1_list,
                         'Training Recall': training_recall_list, 'Training Precision': training_precision_list,
                        'Validation Loss': val_loss_list,'Validation Accuracy': val_accuracy_list, 'Validation F1 Score': val_F1_list,
                         'Validation Recall': val_recall_list, 'Validation Precision': val_precision_list})
eval_df = pd.DataFrame({'In focus Evaluation Accuracy': test_accuracy_list_in, 'Not In Focus Evaluation Accuracy': test_accuracy_list_not})
                        # 'Test F1 Score': test_F1_list, 'Test Recall': test_recall_list, 'Test Precision': test_precision_list})


model_df.to_csv(r'/home/kat/Data/LossAndAccuracy_UNet_elu.csv', index = False)
eval_df.to_csv(r'/home/kat/Data/LossAndAccuracy_eval_UNet_elu_Multifiducial_variableHFW.csv', index = False)