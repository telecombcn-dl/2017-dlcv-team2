import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import keras
from keras.datasets import cifar10

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_cam

import model_cifar10 as cnn

num_classes = 10
dropouts = [0.25, 0.25, 0.5]
filepath='weights.cifar10_e100_CNN.hdf5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = cnn.define_CNN(x_train, num_classes, dropouts, 1) 
print(model.summary())
model.load_weights(filepath, by_name=True)
print('Model loaded.')


# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'fc2_out'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

heatmaps = []
#for path in image_paths:
for i in range(5):
    seed_img = utils.load_img(path, target_size=(32, 32))
    print(seed_img.shape)
    seed_img = x_train[np.random.randint(i*13, 50000)]
    print(seed_img.shape)
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Activation map')
plt.savefig('activation_map_cnn')
