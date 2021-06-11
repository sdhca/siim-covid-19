# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm
from pathlib import Path
import re
from shutil import copyfile
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
input_path = Path('image_files/siim-covid-19-1-data-prep')
data_path = Path('data')
output_path = Path('image_files/masked')
train_df = pd.read_csv('data/train_scaled.zip')
dev_df = pd.read_csv('data/dev_scaled.zip')


# %%
train_df.head()


# %%
def load_model(path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    model.summary()
    return model

def load_image(path, new_size=(128, 128)):
    # Read image
    input_image = cv2.imread(path)
    # Scale Image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) / 255.0
    # Resize Image
    input_image = cv2.resize(input_image, new_size)
    # Add 1-channel
    input_image = input_image[..., np.newaxis]
    #convert to np.array
    image = np.array(input_image).reshape((1, new_size[0], new_size[1], 1))

    return image


# def main():
# %%
# activate_GPU(do_GPU=False)  # Steve, you can try this either way
plot_results = False
# load model
model = load_model('data/lungs.h5')

# %%
# datasets = pd.read_csv('../input/datasetscsv/datasets.csv')
datasets = pd.DataFrame.from_dict(
    {'Dataset': ['train_scaled', 'dev_scaled'],
    'RecordFile': [data_path/'train_scaled.zip', data_path/'dev_scaled.zip'],
    'ImageInput': [input_path/'train_png', input_path/'dev_png']}
)
datasets

# %%
# Produced 128x128 masked images
img_size = (128, 128)

for _, ds in tqdm(datasets.iterrows(), total=len(datasets), desc='Datasets'):
    # print(ds['Dataset'])
    dataset_name =ds.Dataset
    img_dir = Path(ds.ImageInput)
    data = pd.read_csv(ds.RecordFile)
    for _, row in tqdm(data.iterrows(), total=len(data), desc='{0} images'.format(dataset_name)):
    # for _, row in tqdm(data.head(n=4).iterrows(), total=len(data), desc='{0} images'.format(dataset_name)):
        image_file = re.sub(r'_image$', '', row.id) + '.png'
        png_in = img_dir/row.StudyInstanceUID/image_file
        out_name = '{0}_{1}x{2}'.format(dataset_name, img_size[0], img_size[1])
        png_out = output_path/out_name/row.StudyInstanceUID/image_file
        png_out.parent.mkdir(exist_ok=True, parents=True)

        # Load image
        img = load_image(str(png_in), img_size)
        # plt.imshow(img)
        # plt.show()

        # Predict lung mask
        mask = np.squeeze(model.predict(img))
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)  # this rounds fractions up or down
        # binary_mask = np.array(binary_mask)

        # Save masked image
        img = np.squeeze(img)
        masked_img = img * binary_mask
        png_data = (masked_img * 255).astype(np.uint8)
        cv2.imwrite(str(png_out), png_data)


        # Plot masks
        if plot_results:
            fig = plt.figure()
            fig.add_subplot(131)
            plt.imshow(img)
            fig.add_subplot(132)
            plt.imshow(binary_mask)
            fig.add_subplot(133)
            plt.imshow(png_data)
            plt.show()


# %%
def sort_images_into_2_classes(data_df, input_path, output_path, img_size, data_split_name='train'):
    # Create output directories
    base_dir = 'png_masked_2-class_{0}x{1}'.format(img_size[0], img_size[1])
    (output_path/base_dir/data_split_name/'positive').mkdir(parents=True, exist_ok=True)
    (output_path/base_dir/data_split_name/'negative').mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        png_name = re.sub(r'_image$', '', row.id) + '.png'
        src = input_path/row.StudyInstanceUID/png_name
        img_class = 'negative'
        if row.Opacity != 'negative':
            img_class = 'positive'
        dest = output_path/base_dir/data_split_name/img_class/'{0}-{1}'.format(row.StudyInstanceUID, png_name)
        copyfile(src, dest)

# %%
split_output = Path('image_files/2-classes')
sort_images_into_2_classes(train_df, output_path/'train_scaled_128x128', split_output, img_size, 'train')
sort_images_into_2_classes(dev_df, output_path/'dev_scaled_128x128', split_output, img_size, 'dev')

 # %%