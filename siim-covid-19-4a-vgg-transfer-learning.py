# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib import pyplot as plt
import re


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

image_size = (224, 224)
# image_size = (128, 128)


# %%
# df_input_path = Path('../input/siim-covid-19-3-positive-negative-class-split/data')
df_input_path = Path('data')
image_input_path = Path('image_files/2-classes/png_2-class_1024x1024')


# %%
datagen_shuffle=True  # If shuffle is False, it looks like the generator will go through the negative directory first and the positive directory second
batch_size=20
# datagen = ImageDataGenerator(rescale=1./255)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             width_shift_range=[-20, 20],
                             height_shift_range=[-20, 20],
                             rotation_range = 30)
train_it = datagen.flow_from_directory(image_input_path/'train', class_mode='binary', shuffle=datagen_shuffle, target_size=image_size, batch_size=batch_size)
dev_it = datagen.flow_from_directory(image_input_path/'dev', class_mode='binary', shuffle=datagen_shuffle, target_size=image_size, batch_size=batch_size)
print(train_it.class_indices)


# %%
# images, labels = next(train_it)
# print(labels)
# print(np.unique(labels))
# print(images[0].shape)
# print(np.min(images[0]), np.max(images[1]))
# plt.subplot(131)
# plt.imshow(images[0][:, :, 0])
# plt.subplot(132)
# plt.imshow(images[0][:, :, 1])
# plt.subplot(133)
# plt.imshow(images[0][:, :, 2])
# plt.show()


# %%
# Getting the ordered list of filenames for the images
image_files = pd.Series(train_it.filenames)

image_files = list(image_files.str.split('\\', expand=True)[1].str[:-4])
print(image_files[:5])
# print(image_files)

# For exploration later if we want to include patient sex or other classification data as input a model:
# This is supposed to set the order the table containing class labels or other data in the same order as the images in the data generator
train_df = pd.read_csv(df_input_path/'train_scaled.zip')
train_df['study-image'] = train_df.StudyInstanceUID + '-' + train_df.id.str.replace('_image', '')
train_df = train_df.set_index('study-image')
# train_df.head()
# Sorting the structured data into the same order as the images
train_df_sorted = train_df.reindex(image_files)
train_df_sorted.head()


# %%
# %% Load and modify VGG16 model
new_input = Input(shape=(image_size[0], image_size[1], 3))
base_model = VGG16(include_top=False, input_tensor=new_input)
base_model.trainable = False
# for layer in base_model.layers:
#     layer.trainable = False
flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(class1)
model = Model(inputs=base_model.inputs, outputs=output)
# model.compile()  # Compile after making any changes including making the base_model trainable
model.summary()


# %%
# %%
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 30
history = model.fit(train_it, epochs=epochs, validation_data=dev_it)
# model.fit(train_it, steps_per_epoch=16, validation_data=dev_it, validation_steps=8)


# %%
# _, acc = model.evaluate(dev_it, steps=len(dev_it))
# print('> %.3f' % (acc * 100.0))
history.history.keys()


# %%
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['binary_accuracy'], color='blue', label='train')
    plt.plot(history.history['val_binary_accuracy'], color='orange', label='test')
    # save plot to file
#     filename = sys.argv[0].split('/')[-1]
    plt.show()
#     pyplot.savefig(filename + '_plot.png')
#     pyplot.close()
summarize_diagnostics(history)


# %%
# epochs = 20
# history2 = model.fit(train_it, epochs=epochs, validation_data=dev_it)
# summarize_diagnostics(history2)


# %%
model.save('data/VGG_transfer.h5')

# %%

