import numpy as np
import tensorflow as tf
import cv2 as cv2
import matplotlib.pyplot as plt

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

def load_images(path_to_images):
    images = []

    for path in path_to_images:

        # Read image
        input_image = cv2.imread(path)

        # Scale Image
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) / 255.0

        # Resize Image
        input_image = cv2.resize(input_image, (128, 128))

        # Add 1-channel
        input_image = input_image[..., np.newaxis]

        # Add to list of images
        images.append(input_image)

    #convert to np.array
    images = np.array(images)

    return images


def main():
    activate_GPU(do_GPU=False)  # Steve, you can try this either way
    plot_results = True

    path_to_images = ['./image_files/siim-covid-19-1-data-prep/train_png/000c9c05fd14/51759b5579bc.png',
                      './image_files/siim-covid-19-1-data-prep/train_png/0b89a95b6733/e2515b943e1e.png']

    # load model
    model = load_model('./data/lungs.h5')

    # load images.  paths is a list of paths to images
    input_images = load_images(path_to_images)

    # Run model
    masks = model.predict(input_images)

    # round up or down to 0 or 1
    binary_masks = []
    for j,mask in enumerate(masks):
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)  # this rounds fractions up or down
        binary_masks.append(binary_mask)
    binary_masks = np.array(binary_masks)


    # Plot masks
    if plot_results:
        n = len(path_to_images)
        fig = plt.figure()

        for j in range(n):

            # Plot original Image
            fig.add_subplot(n, 3, 1+j*3)
            plt.imshow(input_images[j])

            # Plot mask
            fig.add_subplot(n, 3, 2+j*3)
            plt.imshow(binary_masks[j])

            # Plot mask
            fig.add_subplot(n, 3, 3+j*3)
            plt.imshow(binary_masks[j,:,:]*input_images[j,:,:,0])

        # Show mask
        fig.canvas.manager.set_window_title('Image and Mask')
        plt.show()

main()
