# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
import re

# %%
#This might not be necessary, I need it for my GPU
def activate_GPU(do_GPU=True):
    physical_devices = tf.config.list_physical_devices('GPU')

    if do_GPU:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        # Disable first GPU
        try:
            tf.config.set_visible_devices(physical_devices[1:], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            # Logical device was not created for first GPU
            assert len(logical_devices) == len(physical_devices) - 1
        except:
            # Insame device or cannot modify virtual devices once initialized.
            pass

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
activate_GPU(do_GPU=False)  # Steve, you can try this either way
plot_results = False
# load model
model = load_model('./data/lungs.h5')

 # %%
# Produced 128x128 masked images
img_size = (128, 128)
data_dir = Path('./data')
out_dir = Path('./image_files/')

datasets = pd.read_csv(data_dir/'datasets.csv')
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
        png_out = out_dir/out_name/row.StudyInstanceUID/image_file
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
# main()