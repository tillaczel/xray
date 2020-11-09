import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

y_train = pd.read_csv('target_train.csv', index_col=0)
y_test = pd.read_csv('target_test.csv', index_col=0)
image_names = os.listdir(f'{os.getcwd()}/images')
image_names = [os.path.splitext(x)[0] for x in image_names]

ims_train = []
for im in y_train.index.to_numpy():
    try:
        ims_train.append(Image.open(f'images/{im}.png'))
    except:
        pass

ims_test = []
double_label_df = pd.DataFrame(columns=['1. label', '2. label'])
for im in y_test.index.to_numpy():
    try:
        ims_test.append(Image.open(f'images/{im}.png'))
    except:
        pass
    df = pd.DataFrame(y_test.loc[im].to_numpy()[np.newaxis], index=[im], columns=['1. label', '2. label'])
    double_label_df = double_label_df.append(df)

missing_image = list(set(list(y_train.index.to_numpy()))-set(image_names))
missing_image.sort()
print(missing_image)

missing_image = list(set(list(y_test.index.to_numpy(dtype='str')))-set(image_names))
missing_image.sort()
print(missing_image)

missing_label = list(set(image_names)-set(list(y_train.index.to_numpy()))-set(list(y_test.index.to_numpy(dtype='str'))))
missing_label.sort()
print(missing_label)
print(len(missing_label))

print(double_label_df.shape)
double_label_df.to_csv('double_label.csv')

