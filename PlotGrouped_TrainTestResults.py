import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##### globals #####
path = r'/Users/trentstarkey/Desktop'
folder = '/ExpRelu_TrainingData_sns'

def Plot_Test_Data():
    all_data = []
    test_set = ['LowkVMixed', 'Multifiducial', 'Showerdrains']

    #----- prepare data -----#
    for file in os.listdir(path + folder):
        if '.DS_Store' in file:
            continue

        if '.csv' not in file:
            continue

        data = pd.read_csv(path + folder + '/' + file)
        
        file = file.split('eval_')[1]
        data = data.rename(columns={'In focus Evaluation Accuracy': 'In Focus'})
        data = data.rename(columns={'Not In Focus Evaluation Accuracy': 'Not In Focus'})

        for ts in test_set:
            if ts in file:
                model = file.split(f'_{ts}')[0]
                data['Model'] = model
                data['Test Set'] = ts
        
        all_data.append(data)

    all_data = pd.concat(all_data, ignore_index = True)  
    all_data['Group'] = all_data['Model'].str.extract(r'(UNet(?:_\d+\.\d+\.\d+)?)')[0]
    all_data['Activation'] = all_data['Model'].str.extract(r'_(relu|elu|gelu)$')
    all_data['Activation'] = all_data['Activation'].fillna('ExpRelu')
    all_data.groupby('Group')

    #----- plot -----#
    g = sns.catplot(data = all_data, kind = 'bar', x = 'Test Set', y = 'Not In Focus', 
                hue = 'Activation', palette = 'cool', col = 'Group')

    g.set_titles('{col_name}')
    g.fig.suptitle('Not In Focus Test Accuracy', y = 0.99)
    g.set_axis_labels('', 'Accuracy')
    g.fig.supxlabel('Test Set')

    plt.savefig(path + folder + '/' + 'TestData.jpeg')

    return

def Plot_Training_Data():
    all_data = []

    #----- prepare data -----#
    for file in os.listdir(path + folder):
        if '.DS_Store' in file:
            continue

        if '.csv' not in file:
            continue

        data = pd.read_csv(path + folder + '/' + file)
        data['Epochs'] = range(1, len(data)+1)
        
        file = file.split('LossAndAccuracy_')[1]
        data['Model'] = file.removesuffix('.csv')
        all_data.append(data)

    all_data = pd.concat(all_data, ignore_index = False)
    all_data['Group'] = all_data['Model'].str.extract(r'(UNet(?:_\d+\.\d+\.\d+)?)')[0]
    all_data['Activation'] = all_data['Model'].str.extract(r'_(relu|elu|gelu)$')
    all_data['Activation'] = all_data['Activation'].fillna('ExpRelu')
    all_data.groupby('Group')

    #----- plot -----#
    g = sns.relplot(data = all_data, kind = 'line', x = 'Epochs', y = 'Training Loss', 
                hue = 'Activation', palette = 'cool', col = 'Group')

    g.set_titles('{col_name}')
    g.fig.suptitle('Training Loss', y = 0.99)
    g.set_axis_labels('', 'Loss')
    g.fig.supxlabel('Epoch')

    plt.savefig(path + folder + '/' + 'Training_Loss.jpeg')

    return

##### call functions #####
# Plot_Test_Data()
Plot_Training_Data()

