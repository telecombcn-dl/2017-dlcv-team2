
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation

import model_cifar10 as cnn

import keras
from keras.datasets import cifar10 


# Build the VGG16 network with ImageNet weights
num_classes = 10
dropouts = [0.25, 0.25, 0.5]
filepath='weights.cifar10_e100_CNN.hdf5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = cnn.define_CNN(x_train, num_classes, dropouts, 1)  
model.load_weights(filepath, by_name=True)

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'activation_1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Generate three different images of the same output index.
vis_images = []
for idx in [1, 1, 1]:
    img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=500)
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images)    
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.savefig('visualization_dense_cnn_predictions')

