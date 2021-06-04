# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from skimage import segmentation, feature, future, morphology, measure, transform
from scipy import ndimage
# import skimage
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from functools import partial
from PIL import Image
from pathlib import Path
from math import sqrt
from tqdm import tqdm
import re

# %%
# full_img = data.skin()
train_df = pd.read_csv('train_scaled.csv')
train_df.head()
train_png_dir = Path('train_png')
full_img = Image.open(train_png_dir/'000c9c05fd14'/'51759b5579bc.png')
# full_img.show()

# img = full_img[:900, :900]
img = np.array(full_img)
plt.imshow(full_img, cmap='Greys')
plt.show()

# %%
# Build an array of labels for training the segmentation.
# Here we use rectangles but visualization libraries such as plotly
# (and napari?) can be used to draw a mask on the image.
training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
# lung
training_labels[200:600, 200:300] = 2
# body
training_labels[800:1000, 200:600] = 1
# air
training_labels[0:50, 0:50] = 3


sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        multichannel=True)
features = features_func(img)
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
# clf = AdaBoostClassifier()
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
ax[0].contour(training_labels)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()


# %%
def plot_segmentation(image, training_boundaries, segmentation_result):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    ax[0].imshow(segmentation.mark_boundaries(image, segmentation_result, mode='thick'))
    ax[0].contour(training_boundaries)
    ax[0].set_title('Image, mask and segmentation boundaries')
    ax[1].imshow(segmentation_result)
    ax[1].set_title('Segmentation')
    fig.tight_layout()


# %%  # Merge air and body segments, set lung=1, not-lung=0
result2 = result.copy()
result2[result2 == 3] = 1
result2 = result2 - 1
plot_segmentation(img, training_labels, result2)

# # %%  Fill holes
# result3 = ndimage.binary_fill_holes(result2, structure=np.ones((5,5)))
# plot_segmentation(img, training_labels, result3)

# %%  # Remove small holes
result3 = morphology.remove_small_holes(result2==1, area_threshold=100)
result3 = morphology.remove_small_objects(result3, min_size=1000)
plot_segmentation(img, training_labels, result3)

# %%  # Use Watershed to split lungs from shoulders
distance = ndimage.distance_transform_edt(result3)
# plt.imshow(distance)
# plt.show()

coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=result3)
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndimage.label(mask)
# result4 = segmentation.watershed(-distance, markers=10, mask=result3, watershed_line=True, compactness=0)
result4 = segmentation.watershed(-np.array(distance), markers=4, mask=result3, watershed_line=False, compactness=0)
plot_segmentation(img, training_labels, result4)

# %%
distance = ndimage.distance_transform_edt(result4)
labelled = np.zeros(result4.shape)
labelled[result4 > 0] = 1
properties = measure.regionprops(label_image=labelled.astype(int), intensity_image=distance)
centroid = properties[0].centroid
print(centroid)
# plt.imshow(distance)
# plt.show()
# distance_from_centroid = np.fromfunction(lambda i, j: np.linalg.norm((i - centroid[0], j - centroid[1])), distance.shape)
# Not sure why np.linalg.norm couldn't be used
distance_from_centroid = np.fromfunction(lambda i, j: (i-centroid[0])**2 + (j-centroid[1])**2, distance.shape)
distance_from_centroid = np.sqrt(distance_from_centroid)
distance2 = distance - 3 * distance_from_centroid * np.max(distance) / np.max(distance_from_centroid)
plt.imshow(distance2)
plt.show()

result5 = segmentation.watershed(-distance2, markers=10, mask=result4, watershed_line=False, compactness=0.15)
plot_segmentation(img, training_labels, result5)

# %%
result6 = segmentation.clear_border(result5)
plot_segmentation(img, training_labels, result6)


# %%
mask = np.zeros(result6.shape)
mask[result6 > 0] = 1
plt.imshow(mask)
plt.show()

# %%
plt.imshow(img, 'Greys')
plt.imshow(mask, 'brg', alpha=0.3)
plt.show()


# %%
where = np.array(np.where(mask))
x1, y1 = np.min(where, axis=1)
x2, y2 = np.max(where, axis=1)
h = y2 - y1
w = x2 - x1

if h > w:
    x2 += h-w
else:
    y2 += w-h

img_crop = img[x1:x2, y1:y2]
mask_crop = mask[x1:x2, y1:y2]
plt.imshow(img_crop, 'Greys')
plt.imshow(mask_crop, 'brg', alpha=0.3)
plt.show()

# %%
img_masked = np.ma.array(img_crop, mask=1-mask_crop)
plt.imshow(img_masked, 'Greys')
plt.show()

# %%
img_masked = img_masked - np.min(img_masked)
img_masked = img_masked / np.max(img_masked)
img_masked = 1 - img_masked
plt.imshow(img_masked, 'Greys')
plt.show()

# %%
img_small = transform.resize(img_masked, (224, 224))
plt.imshow(img_small, 'Greys')
plt.show()

# %%
show_debug_images = False
# for i in tqdm(range(100)):
for i in tqdm(range(100, 1000)):
    png_224 = Path('train_png_224')
    png_224.mkdir(exist_ok=True)
    study = train_df.StudyInstanceUID[i]
    image = re.sub(r'_image$', '', train_df.id[i]) + '.png'

    full_img = Image.open(train_png_dir/study/image)
    img = np.array(full_img)
    # plt.imshow(img, cmap='Greys')
    # plt.show()

    features_new = features_func(img)
    result = future.predict_segmenter(features_new, clf)
    if show_debug_images is True:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
        ax[0].set_title('Image')
        ax[1].imshow(result)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        plt.show()

    # %%  # Merge air and body segments, set lung=1, not-lung=0
    result2 = result.copy()
    result2[result2 == 3] = 1
    result2 = result2 - 1
    if show_debug_images is True:
        plot_segmentation(img, training_labels, result2)

    # # %%  Fill holes
    # result3 = ndimage.binary_fill_holes(result2, structure=np.ones((5,5)))
    # plot_segmentation(img, training_labels, result3)

    # %%  # Remove small holes
    result3 = morphology.remove_small_holes(result2==1, area_threshold=100)
    result3 = morphology.remove_small_objects(result3, min_size=1000)
    if show_debug_images is True:
        plot_segmentation(img, training_labels, result3)

    # %%  # Use Watershed to split lungs from shoulders
    distance = ndimage.distance_transform_edt(result3)
    # plt.imshow(distance)
    # plt.show()

    coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=result3)
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers, _ = ndimage.label(mask)
    # result4 = segmentation.watershed(-distance, markers=10, mask=result3, watershed_line=True, compactness=0)
    result4 = segmentation.watershed(-np.array(distance), markers=4, mask=result3, watershed_line=False, compactness=0)
    if show_debug_images is True:
        plot_segmentation(img, training_labels, result4)


    distance = ndimage.distance_transform_edt(result4)
    labelled = np.zeros(result4.shape)
    labelled[result4 > 0] = 1
    properties = measure.regionprops(label_image=labelled.astype(int), intensity_image=distance)
    centroid = properties[0].centroid
    # print(centroid)
    # plt.imshow(distance)
    # plt.show()
    # distance_from_centroid = np.fromfunction(lambda i, j: np.linalg.norm((i - centroid[0], j - centroid[1])), distance.shape)
    # Not sure why np.linalg.norm couldn't be used
    distance_from_centroid = np.fromfunction(lambda i, j: (i-centroid[0])**2 + (j-centroid[1])**2, distance.shape)
    distance_from_centroid = np.sqrt(distance_from_centroid)
    distance2 = distance - 3 * distance_from_centroid * np.max(distance) / np.max(distance_from_centroid)
    if show_debug_images is True:
        plt.imshow(distance2)
        plt.show()

    result5 = segmentation.watershed(-distance2, markers=10, mask=result4, watershed_line=False, compactness=0.15)
    if show_debug_images is True:
        plot_segmentation(img, training_labels, result5)


    result6 = segmentation.clear_border(result5)
    if show_debug_images is True:
        plot_segmentation(img, training_labels, result6)


    mask = np.zeros(result6.shape)
    mask[result6 > 0] = 1
    if show_debug_images is True:
        plt.imshow(mask)
        plt.show()

    plt.imshow(img, 'Greys')
    plt.imshow(mask, 'brg', alpha=0.3)
    plt.savefig(png_224/'{0}-{1}'.format(study, image))
    if show_debug_images is True:
        plt.show()




# %%
