
# coding: utf-8

# # Task 1.1 ARCHITECTURE

# In[1]:

from __future__ import print_function

import os
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as K

# inline plots
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'


# ### Loading the CIFAR-10 dataset

# In[2]:

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Models definition

# ### Model CNN: 5 Conv Layers + 3 Fully Connected Layers

# In[3]:


def define_CNN(input_dataset, num_classes, dropouts, summary):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_dataset.shape[1:], name='conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='relu1'))
    
    model.add(Conv2D(32, (3, 3), name='conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='relu_conv2'))
    
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool1'))
    model.add(Dropout(dropouts[0], name='dropout1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='relu_conv3'))
    
    model.add(Conv2D(64, (3, 3), name='conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='relu_conv4'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool2'))
    model.add(Dropout(dropouts[1], name='dropout2'))

    model.add(Flatten())
    model.add(Dense(512, name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu', name='relu_fc1'))
    model.add(Dropout(dropouts[2], name='dropout3'))
    
    model.add(Dense(num_classes, name='fc2_out'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    if summary:
        model.summary()
    
    return model


# # TASK 1.2 TRAINING

# #### Hyperparameters

# In[12]:

batch_size = 512
num_classes = 10
epochs = 350
dropouts = [0.25, 0.25, 0.5]
learning_rate = 0.001
decay_rate = 1e-6
data_augmentation = True
model_def = 'CNN'


# #### Model defintion

# In[13]:

model = define_CNN(x_train, num_classes, dropouts, 1)     

# from keras.utils import plot_model
# plot_model(model, to_file='model_cnn.png', show_shapes=True)   


# #### Optimizer definition

# In[14]:

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=learning_rate, decay=decay_rate)


# #### Training process

# In[ ]:

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(
        x_train, 
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test))


# #### Saving weights

# In[8]:

# checkpoint
if data_augmentation == True:
    filepath = 'weights.cifar10_e' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_da_bn_X' + '.hdf5'
else:
    filepath = 'weights.cifar10_e' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] +  '_' + model_def + '.hdf5'
    
model.save_weights(filepath)


# #### Accuracy plot

# In[9]:

print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'], '-o')
plt.plot(history.history['val_acc'], '-o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
if data_augmentation == True:
    plt.savefig('acc_hist_' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_da_bn_X')
else:
    plt.savefig('acc_hist_' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_nbn')
plt.show()


# #### Loss plot

# In[10]:

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
if data_augmentation == True:
    plt.savefig('loss_hist_' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_da_bn_X')
else:
    plt.savefig('loss_hist_' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_nbn')
plt.show()


# ## Evaluation

# In[ ]:

if data_augmentation == True:
    filepath = 'weights.cifar10_e' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '_da_bn_X' + '.hdf5'
else:
    filepath = 'weights.cifar10_e' + str(epochs) + '_' + str(dropouts[2]).split('.')[-1] + '_' + model_def + '.hdf5'
    
model.load_weights(filepath, by_name=True)


# In[11]:

# initiate RMSprop optimizer
'''
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
'''

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))


# In[ ]:



