# %%
import numpy as np
import pandas as pd
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# %% Load and modify VGG16 model
new_input = Input(shape=(128, 128, 3))
base_model = VGG16(include_top=False, input_tensor=new_input)
base_model.trainable = False
# for layer in base_model.layers:
#     layer.trainable = False
flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(4, activation='softmax')(class1)
model = Model(inputs=base_model.inputs, outputs=output)
# model.compile()  # Compile after making any changes including making the base_model trainable
model.summary()
# %%

# Load image and convert to numpy array
image = load_img('image.png', target_size=(128, 128))
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
