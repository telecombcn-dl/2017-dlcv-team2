from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

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