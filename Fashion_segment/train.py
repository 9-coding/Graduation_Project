import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow_examples.models.pix2pix import pix2pix

print("GPU: ", tf.config.list_physical_devices('GPU'))

# 이미지와 마스크의 파일 경로를 가져옵니다.
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

# Sample corresponding masks and visualize
NORM = mpl.colors.Normalize(vmin=0, vmax=58)

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

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

from tensorflow_examples.models.pix2pix import pix2pix

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 59

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# configure the training parameters and train the model

TRAIN_LENGTH = 800
EPOCHS = 500
BATCH_SIZE = 128
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE

model_history = model.fit(train,
                          validation_data=test,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          epochs=EPOCHS,
                          verbose=2)

def plot_metrics(metric_name, title, ylim=5):
    '''plots a given metric from the model history'''
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(model_history.history[metric_name],color='blue',label=metric_name)
    plt.plot(model_history.history['val_' + metric_name], color='green',label='val_' + metric_name)
    plt.show()

# Plot the training and validation loss and accuracy
plot_metrics("loss", title="Training vs Validation Loss", ylim=10)
plot_metrics("accuracy", title="Training vs Validation Accuracy", ylim=1)

model.save('model')