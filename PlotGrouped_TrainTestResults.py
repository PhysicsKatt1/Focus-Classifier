import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

##### globals #####
epochs = 15
batch = 3
learning_rate = 10e-6
path = r'/home/kat/Data'
folder = '/FFTshift/2.2.4'

cmap = LinearSegmentedColormap.from_list('cmap', ['blue', 'royalblue', 'indigo', 'slateblue'])
                                               # , 'darkgreen', 'lightgreen', 'darkred', 'lightcoral', 'darkmagenta', 'plum', 'darkorange', 'burlywood'])
labels = ['Class A novel relu', 'Class B novel relu','Class A elu', 'Class B elu', 'Class A gelu', 'Class B gelu', 'Class A relu', 'Class B relu' ]

##### define functions #####
def plot_eval_data():
    model_list = []
    csv_list = []
    index_list = []
    header_list = []
    stats_list = []
    for file in sorted(os.listdir(path + folder)):
        dfs = []
        for csv in sorted(os.listdir(path + folder + '/' + file)):
            if 'eval' in csv:
                imported = pd.read_csv(path + folder + '/' + file + '/' + csv)
                label = csv.removeprefix('LossAndAccuracy_eval_')
                if 'Multifiducial' in label:
                    label = label[:label.index('_Multifiducial')]
                if 'Showerdrains' in label:
                    label = label[:label.index('_Showerdrains')]
                if 'MixedBeams' in label:
                    label = label[:label.index('_MixedBeams')]

                #rename columns
                for column in imported :
                    if 'Not' in column:
                        imported.rename({column:'Not In Focus'}, axis=1, inplace=True)
                    else:
                        imported.rename({column:'In Focus'}, axis=1, inplace=True)
                dfs.append(imported)
                csv_list.append(csv)

        header_list.append(label)
        model_df = pd.concat(dfs)
        model_df = model_df.groupby('In Focus').first().reset_index()
        model_list.append(model_df)

    # get index labels
    for names in csv_list:
        names = names.removeprefix('LossAndAccuracy_eval')
        if 'Multifiducial' in names:
            name = names[names.index('Multifiducial'):names.index('.csv')]
            index_list.append(name)
        if 'Showerdrains' in names:
            name = names[names.index('Showerdrains'):names.index('.csv')]
            index_list.append(name)
        if 'MixedBeams' in names:
            name = names[names.index('MixedBeams'):names.index('.csv')]
            index_list.append(name)

    index_list = index_list[:len(os.listdir(path + folder + '/' + file))-1]
   
    for model, label in zip(model_list, header_list):
        for column in model:
                if 'Not' in column:
                    model.rename({column:label+ ' Not In Focus'}, axis=1, inplace=True) 
                else:
                    model.rename({column:label+ ' In Focus'}, axis=1, inplace=True) 

        model.index = index_list

        not_in_foc = []
        in_foc = []
        diff = []
        not_in_foc_mean = []
        not_in_foc_std = []
        not_in_foc_var = []
        in_foc_mean = []
        in_foc_std = []
        in_foc_var = []
        for column in model:
            if 'Not' in column:
                not_in_foc.append(model[column])
            else:
                in_foc.append(model[column])
        for i, j, in zip(not_in_foc, in_foc):
            diff.append(abs(i-j))

        mean_diff = []
        std_diff = []
        var_diff = []
        mean_diff.append(np.mean(diff))
        std_diff.append(np.std(diff))
        var_diff.append(np.var(diff))
        not_in_foc_mean.append(np.mean(not_in_foc))
        not_in_foc_std.append(np.std(not_in_foc))
        not_in_foc_var.append(np.var(not_in_foc))
        in_foc_mean.append(np.mean(in_foc))
        in_foc_std.append(np.std(in_foc))
        in_foc_var.append(np.var(in_foc))


        df = pd.DataFrame({'In Focus Mean': in_foc_mean, 'In Focus Std': in_foc_std, 'In Focus Var': in_foc_var,
                            'Not In Focus Mean': not_in_foc_mean, 'Not In Focus Std': not_in_foc_std, 'Not In Focus Var': not_in_foc_var,
                            'Mean Interclass Delta': mean_diff, 'Stddev Interclass Delta': std_diff, 'Var Interclass Delta': var_diff})
        stats_list.append(df)
    all_stats = pd.concat(stats_list, axis=0)
    all_stats.index = header_list
    all_stats.to_csv(path + '/EvalStats_PerClass_2.2.4.csv')


    # plot
    pd.concat(model_list, axis = 1).plot.bar(title = 'Evaluation Results Per Model', xlabel = 'Test Set', ylabel = 'Percent Accuracy',
                    colormap=cmap, figsize=(10, 8)) #Paired
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(path + '/EvalAccuracy_2.2.4.png')

    return header_list

def plot_training_data(headers):
    train_accuracy_list = []
    train_loss_list = []
    val_accuracy_list = []
    val_loss_list = []
    train_accuracy_df = pd.DataFrame()
    train_loss_df = pd.DataFrame()
    val_accuracy_df = pd.DataFrame()
    val_loss_df = pd.DataFrame()

    for data in sorted(os.listdir(path + folder)):
        for csv in sorted(os.listdir(path + folder  + '/' + data)):
            if 'eval' not in csv:
                df = pd.read_csv(path + folder + '/' + data + '/' + csv)

                for column in df:
                    if 'Training Accuracy' in column:
                                train_accuracy_list.append(df[column])
                    if 'Training Loss' in column:
                                train_loss_list.append(df[column])
                    if 'Validation Accuracy' in column:
                                val_accuracy_list.append(df[column])
                    if 'Validation Loss' in column:
                                val_loss_list.append(df[column])

    for list, header in zip(train_accuracy_list, headers):
        train_accuracy_df[header + ' Training'] = list

    for list, header in zip(train_loss_list, headers):
        train_loss_df[header + ' Training'] = list

    for list, header in zip(val_accuracy_list, headers):
        val_accuracy_df[header + ' Validation'] = list

    for list, header in zip(val_loss_list, headers):
        val_loss_df[header + ' Validation'] = list

    train_accuracy_df = train_accuracy_df.fillna(0)
    val_accuracy_df = val_accuracy_df.fillna(0)
          
    # # plot accuracy data in one plot per model
    # ax= train_accuracy_df.plot(subplots=True, kind='line', legend=False, xlabel='Epoch',
    #                        ylabel='Accuracy', ylim=[0.5, 1.1], color = 'indigo', figsize = (13, 13), title = 'Training and Validation Accuracy')
    # val_accuracy_df.plot(ax=ax, subplots=True, kind='line', legend=False, xlabel='Epoch',
    #                      linestyle='--', ylabel='Accuracy', color = 'gray', title = headers, ylim=[0.5, 1.1])
    # plt.legend(['Training', 'Validation'], labelcolor = 'black', bbox_to_anchor = (1.13, 7.0))
    # plt.tight_layout()
    # plt.savefig(path + '/TrainingValAccuracyPerModel.png')

    # # plot accuracy data in a single plot
    # ax = train_accuracy_df.plot(subplots=False, kind='line', legend=True, xlabel='Epoch',
    #                             ylabel='Accuracy', figsize=(13, 13), # ylim=[-0.1, 1.1]
    #                             title='Training and Validation Accuracy', colormap=cmap)
    # val_accuracy_df.plot(ax=ax, subplots=False, kind='line', legend=True, xlabel='Epoch',
    #                      linestyle='--', ylabel='Accuracy', title='Training and Validation Accuracy', colormap=cmap)
    # plt.legend(bbox_to_anchor=(1.0, 0.2))
    # plt.tight_layout()
    # plt.savefig(path + '/TrainingValAccuracyPerModel_2.2.4.png')

    # # plot loss data
    # ax= train_loss_df.plot(subplots=True, kind='line', legend=False, xlabel='Epoch',
    #                        ylabel='Accuracy', ylim=[0, 1], figsize = (13, 13), title = 'Training and Validation Loss')
    # val_loss_df.plot(ax=ax, subplots=True, kind='line', legend=False, xlabel='Epoch',
    #                      linestyle='--', ylabel='Loss', title = headers, ylim=[0, 1])
    # plt.legend(['Training', 'Validation'], labelcolor = 'black', bbox_to_anchor = (1.13, 7.0))
    # plt.tight_layout()
    # plt.savefig(path + '/TrainingValLossPerModel.png')

    #plot loss data in a single plot
    ax = train_loss_df.plot(subplots=False, kind='line', legend=True, xlabel='Epoch',
                                ylabel='Loss', figsize=(13, 13),    # ylim=[-0.1, 600], 
                                title='Training and Validation Loss', colormap=cmap) 
    val_loss_df.plot(ax=ax, subplots=False, kind='line', legend=True, xlabel='Epoch',
                         linestyle='--', ylabel='Loss', title='Training and Validation Loss', colormap=cmap) #ylim=[-0.1, 600]
    plt.legend(bbox_to_anchor=(1.0, 0.2))
    plt.tight_layout()
    plt.savefig(path + '/TrainingValLossPerModel_2.2.4.png')

    return


##### call functions #####
header_list = plot_eval_data()
plot_training_data(header_list)