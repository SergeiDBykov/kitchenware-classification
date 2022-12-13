import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

repo_path = os.path.dirname(os.path.abspath(__file__))


datapath = os.path.join(repo_path, 'data/')

available_classes = ['glass', 'cup', 'spoon', 'plate', 'knife', 'fork',]


train_labels = pd.read_csv(datapath + '/train.csv', index_col=0)

def show_train_example(cls = 'knife'):
    cls = cls.lower()
    if cls not in available_classes:
        raise ValueError('Class not available')
    else:
        #create figure with 3x2 sub-plots
        fig, axes = plt.subplots(2, 3, figsize=(15,10))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        axes = axes.flatten()
        print('Showing examples of class: {}'.format
                (cls))
        cls_df = train_labels[train_labels['label'] == cls]
        cls_df = cls_df.sample(6)
        for i in range(len(cls_df)):
            img = plt.imread(datapath + 'images/' + str(cls_df.index[i])+'.jpg')
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f'ID: {cls_df.index[i]}')
        plt.show()


