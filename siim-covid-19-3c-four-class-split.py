# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from pathlib import Path
import re
from shutil import copyfile

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
output_path = Path('image_files/4-classes/')
train_df = pd.read_csv('data/train_scaled.zip')
dev_df = pd.read_csv('data/dev_scaled.zip')


# %%
train_df.head()


# %%
def sort_images_into_4_classes(data_df, input_path, output_path, data_split_name='train'):
    # Create output directories
    base_dir = 'png_4-class_1024x1024'
    # "negative", "typical", "indeterminate", "atypical"
    
    (output_path/base_dir/data_split_name/'negative').mkdir(parents=True, exist_ok=True)
    (output_path/base_dir/data_split_name/'typical').mkdir(parents=True, exist_ok=True)
    (output_path/base_dir/data_split_name/'indeterminate').mkdir(parents=True, exist_ok=True)
    (output_path/base_dir/data_split_name/'atypical').mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        png_name = re.sub(r'_image$', '', row.id) + '.png'
        src = input_path/row.StudyInstanceUID/png_name
        img_class = row.Opacity
        dest = output_path/base_dir/data_split_name/img_class/'{0}-{1}'.format(row.StudyInstanceUID, png_name)
        copyfile(src, dest)


# %%
sort_images_into_4_classes(train_df, input_path/'train_png', output_path, 'train')
sort_images_into_4_classes(dev_df, input_path/'dev_png', output_path, 'dev')
# %%
