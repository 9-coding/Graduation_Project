import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import glob
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('model')
# Let plot our model

image_paths = sorted(glob.glob('dataset/png_images/IMAGES/*.png'))
mask_paths = sorted(glob.glob('dataset/png_masks/MASKS/*.png'))

# 이미지 경로의 파일 존재 여부를 확인합니다.
if image_paths and mask_paths:
    print("Files exists")
else:
    print("Files don't exist")


def load_image(image_path):
    file = tf.io.read_file(image_path)
    image = tf.image.decode_png(file, channels=3)
    return image

def load_mask(mask_path):
    file = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(file, channels=1)
    return mask

# image_paths와 mask_paths는 이미지 파일의 경로를 담고 있는 리스트로 가정합니다.
images = [load_image(image_path) for image_path in image_paths]
masks = [load_mask(mask_path) for mask_path in mask_paths]
print(len(images), len(masks))

# Sample few images and visualize
plt.figure(figsize=(16, 5))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(images[i])
    plt.colorbar()
    plt.axis('off')
plt.show()

# Sample corresponding masks and visualize
NORM = mpl.colors.Normalize(vmin=0, vmax=58)

# plot masks
plt.figure(figsize=(16, 5))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(masks[i], cmap='jet', norm=NORM)
    plt.colorbar()
    plt.axis('off')
plt.show()

# Resize data as the model expects
def resize_and_normalize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)/255.0
    # resize the image
    image = tf.image.resize(image, (128, 128))
    return image

def resize_mask(mask):
    # resize the mask
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return mask

X = [resize_and_normalize_image(image) for image in images]
y = [resize_mask(mask) for mask in masks]
print(len(X), len(y))

# Visualize a resized image and a resized mask
plt.figure(figsize=(16, 5))
# plot an image
plt.subplot(1, 2, 1)
plt.imshow(X[25])
plt.colorbar()

# plot a mask
plt.subplot(1, 2, 2)
plt.imshow(y[25])
plt.colorbar()
plt.show()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=16)

# Data Argumentation Codes
def brigtness(img, mask):
    # adjust brigtness of image, don't alter in mask
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask

def hue(img, mask):
    # adjust hue of image, don't alter in mask
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask

def crop(img, mask):
    # crop both image and mask identically
    img = tf.image.central_crop(img, 0.7)
    mask = tf.image.central_crop(mask, 0.7)
    img = tf.image.resize(img, (128, 128))
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return img, mask

def flip_horizontal(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

def flip_vertical(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

def rotate(img, mask):
    # rotate both image and mask identically
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask

def preprocessing_data(train_X, test_X, train_y, test_y, BATCH_SIZE=64, BUFFER_SIZE=1000):
    train_X = tf.data.Dataset.from_tensor_slices(train_X.copy())
    test_X = tf.data.Dataset.from_tensor_slices(test_X.copy())
    train_y = tf.data.Dataset.from_tensor_slices(train_y.copy())
    test_y = tf.data.Dataset.from_tensor_slices(test_y.copy())
    # zip images and masks
    train = tf.data.Dataset.zip((train_X, train_y))
    test = tf.data.Dataset.zip((test_X, test_y))
    # Perform augmentation on train data only
    a = train.map(brigtness)
    b = train.map(hue)
    c = train.map(crop)
    d = train.map(flip_horizontal)
    e = train.map(flip_vertical)
    f = train.map(rotate)
    # Concatenate every new augmented data
    for aug_data in [a, b, c, d, e, f]:
        train = train.concatenate(aug_data)
    # shuffle and group the train set into batches
    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    # do a prefetch to optimize processing
    train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # group the test set into batches
    test = test.batch(BATCH_SIZE)
    return train, test

train, test = preprocessing_data(train_X, test_X, train_y, test_y)

# 임시 코드
image_paths = sorted(glob.glob('dataset/png_images/IMAGES/*.png'))
mask_paths = sorted(glob.glob('dataset/png_masks/MASKS/*.png'))

csv_file = pd.read_csv('dataset/labels.csv')
class_names = list(tuple(csv_file['label_list']))
class_names[0] = 'background'
THRESHOLD = 0.6  # You can modify this value based on your requirement


def closest_color(rgb_values):
    # 0~255지만 입력을 고려하여 20~ 200으로 조정
    colors = {
        'Red': [255, 0, 0],
        'Green': [0, 255, 0],
        'Blue': [0, 0, 255],
        'Black': [0, 0, 0],
        'white': [255, 255, 255],
        'Yellow': [255, 255, 0],
        'Pink': [255, 192, 203]
    }

    min_distance = float('inf')
    closest = None

    for name, color in colors.items():
        distance = np.linalg.norm(np.array(rgb_values) - np.array(color))
        if distance < min_distance:
            min_distance = distance
            closest = name

    return closest


def predict_and_visualize(model, test, num_examples=5):
    img, mask = next(iter(test))
    pred = model.predict(img)
    for i in range(num_examples):
        # plot the predicted mask

        plt.figure(figsize=(20, 15))
        plt.subplot(1, 3, 1)
        probabilities = tf.nn.softmax(pred[i])  # Getting softmax probabilities
        predict = tf.argmax(pred[i], axis=-1)
        predict = np.array(predict)
        mask_high_confidence = np.max(probabilities, axis=-1) > THRESHOLD  # Creating a mask for high confidence pixels
        predict[~mask_high_confidence] = 0  # Assigning a background class to low confidence pixels
        plt.imshow(predict, cmap='jet')
        plt.axis('off')

        # Adding unique class labels present in the prediction
        unique_labels = np.unique(predict)
        # title = 'Prediction: ' + ', '.join([class_names[label] for label in unique_labels if label != 0])
        title = 'Prediction'
        plt.title(title)

        # plot the groundtruth mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask[i], cmap='jet')
        plt.axis('off')
        plt.title('Ground Truth')

        # plot the actual image
        plt.subplot(1, 3, 3)
        plt.imshow(img[i])
        plt.axis('off')
        plt.title('Actual Image')

        plt.show()

        for n in unique_labels:
            position = np.where(predict == n)
            index = random.choice(list(zip(position[0], position[1])))
            img_array = np.array(img[i])
            color = img_array[index[0], index[1]] * 255
            color_name = closest_color(color)
            if n == 0 or n == 41:  # ignore background and skin
                continue

            print(color_name, end=" ")
            print(class_names[n])
            print(color)
            print()


predict_and_visualize(model, test)