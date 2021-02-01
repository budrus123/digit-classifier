from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

image_width = 32
image_height = 32
on_threshold = 200
number_of_neighbors = 6

def resize_image(image):
    '''
    Function that takes an image, and returns
    the re-sized version of the image (32x32), as
    this is the needed size for the machine learner.
    This is also the same size that was used in the
    training dataset.
    :param image: The image to be resized
    :return: The resized image
    '''
    width = image_width
    height = image_height
    img = image.resize((width, height), Image.ANTIALIAS)
    return img

def load_image(file_name):
    '''
    Function to load an image from a file name.
    :param file_name: Name of image file name to be opened
    :return: Opened image
    '''
    img = Image.open(file_name)
    return img

def jpg_to_bit_map(image):
    '''
    Function that converts a jpg input image, into
    a bmp image used in the classification process.
    :param image:  Image file in jpg format
    :return: Converted image in bmp format
    '''
    ary = np.array(image)
    r, g, b = np.split(ary, 3, axis=2)
    r = r.reshape(-1)
    g = r.reshape(-1)
    b = r.reshape(-1)
    bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2],
                      zip(r, g, b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float), 255)
    im = Image.fromarray(bitmap.astype(np.uint8))
    return im

def count_on_pixels(image_values, block_number):
    '''
    This is a function that takes a block in the bitmap image
    and counts how many pixels in that block are turned on.
    Pixel are considered on if they are below a certain threshold.
    :param image_values: The bitmap values for the image
    :param block_number: Block number sequence. Int from 0 ..< 64
    :return: Count of pixels that are on
    '''
    row = int(block_number / 8)
    column = block_number % 8
    start_x = column * 4
    end_x = start_x + 4
    start_y = row * 4
    end_y = start_y + 4
    sub_array = image_values[start_y:end_y, start_x:end_x]
    count = 0
    for i in range(4):
        for j in range(4):
            if sub_array[i][j] < on_threshold:
                count += 1
    return count

def create_instance_data(image_values):
    '''
    Function to create a dataset instance. Dataset instances are
    arrays of length 64 that all contain an integer ranging from 0 to 16.
    The original image size is 32x32. Instances are created by taking distinct
    4x4 blocks and counting how many of the 16 pixels are actually on.
    :param image_values: The bitmap values for the image. Array of numbers from 0 ... 255
    :return: Dataset instance (array of 64 values).
    '''
    instance_array = np.zeros(64)
    for i in range(64):
        on_pixels_count = count_on_pixels(image_values, i)
        instance_array[i] = on_pixels_count
    return instance_array


def main():
    # Loading the digits dataset
    digits = datasets.load_digits()
    # Creating the knn learner with number_of_neighbors
    knn = KNeighborsClassifier(n_neighbors = number_of_neighbors)
    # Getting the features data and the target data and training the model
    X = digits.data
    y = digits.target
    knn.fit(X, y)

    # Opening and formatting the image to correct shape
    image = load_image('examples/number_9.jpg')
    image_btmp = jpg_to_bit_map(image)
    resized_image = resize_image(image_btmp)
    values = np.array(resized_image.getdata())
    values = values.reshape(32, 32)

    # Creating the instance for input image
    instance = create_instance_data(values)
    # Predicting the outcome of the instance
    prediction = knn.predict([instance])
    print('Prediction of the digit is: ' + str(prediction[0]))
    # plt.imshow(instance.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()

if __name__ == '__main__':
    main()

