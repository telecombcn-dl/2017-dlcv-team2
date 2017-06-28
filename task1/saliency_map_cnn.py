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
from vis.visualization import visualize_saliency

import model_cifar10 as cnn

num_classes = 10
dropouts = [0.25, 0.25, 0.5]
filepath='weights.cifar10_e100_CNN.hdf5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

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

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    'https://mathcass.files.wordpress.com/2016/07/2012-12-30-09-58-28.jpg?w=1374'
    'https://raw.githubusercontent.com/heuritech/convnets-keras/master/examples/dog.jpg'
    'https://blogs.voanews.com/student-union/files/2012/01/airplane-flickr-shyb.jpg'
    'https://avatars1.githubusercontent.com/u/4365777?v=3&s=400'
    'https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Atelopus_zeteki1.jpg/440px-Atelopus_zeteki1.jpg'
    #"http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    #"http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    #"https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    #"http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
    #"http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
]

from PIL import Image
from scipy.misc import toimage

heatmaps = []
for i in range(5):
    #seed_img = utils.load_img(path, target_size=(32, 32))
    seed_img = x_train[np.random.randint(i*5, 50000)]
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.savefig('saliency_map_cnn')
