
from __future__ import print_function

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import layers,models
import h5py
from keras.models import Model
from keras.layers import Layer


batch_size = 1
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 224, 224

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

p='mnist224_half.hdf5';

db=h5py.File(p);


X, y = db['images'],db['labels'];

# Split train and valid

i = int(db["images"].shape[0] * 0.75)

x_train, x_test, y_train, y_test = X[:i],X[i:],y[:i],y[i:];


print(x_train.shape)

print(y_train.shape)

num_sequence=2;

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3,num_sequence, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3,num_sequence, img_rows, img_cols)
    input_shape = (3,num_sequence, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],num_sequence, img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0],num_sequence, img_rows, img_cols, 3)
    input_shape = (num_sequence,img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[0])




"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
"""

classes = 10;
pooling = 'No'

include_top = False;

input_shape =  [num_sequence,img_rows, img_cols, 3];


img_input = layers.Input(shape=input_shape)

x = TimeDistributed(layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1'))(img_input)
x = TimeDistributed(layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2'))(x)
x = TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

# Block 2
x = TimeDistributed(layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1'))(x)
x = TimeDistributed(layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2'))(x)
x = TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

# Block 3
x = TimeDistributed(layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1'))(x)
x = TimeDistributed(layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2'))(x)
x = TimeDistributed(layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3'))(x)
x = TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

# Block 4
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1'))(x)
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2'))(x)
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv3'))(x)
x = TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

# Block 5
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv1'))(x)
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv2'))(x)
x = TimeDistributed(layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv3'))(x)
x = TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)

if include_top:
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
else:
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    elif pooling =='No':
    	x = x;

# Ensure that the model takes into account
# any potential predecessors of `input_tensor`.
#if input_tensor is not None:
#    inputs = keras_utils.get_source_inputs(input_tensor)
#else:
inputs = img_input
# Create model.
base_model = models.Model(inputs, x, name='vgg16')

weights ='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5';

base_model.load_weights(weights)

class MeanLayer(Layer):

    def __init__(self, **kwargs):

        
        super(MeanLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(MeanLayer, self).build(input_shape)
  
       
    def call(self, inputs):

        print(inputs.shape)

        output=K.mean(inputs,axis=1)

        print(output.shape)

        return output

       

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])






class FCHeadNet:
  @staticmethod
  def build(baseModel, classes, D):
    # initialize the head model that will be placed on top of
    # the base, then add a FC layer
    headModel = baseModel.output
    headModel = TimeDistributed(layers.Flatten(name='flatten'))(headModel)
    headModel = TimeDistributed(layers.Dense(D, activation='relu', name='fc1'))(headModel)
    headModel = TimeDistributed(layers.Dense(D, activation='relu', name='fc2'))(headModel)
    #headModel = MeanLayer()(headModel)
    #headModel = layers.Dense(classes, activation='softmax', name='predictions')(headModel)
    headModel = TimeDistributed(layers.Dense(classes, activation='softmax', name='predictions'))(headModel)
    headModel = MeanLayer()(headModel)
    # add a softmax layer
    #headModel = layers.Dense(classes, activation="softmax")(headModel)
    #headModel = layers.Dense(classes, activation='softmax', name='predictions2')(headModel)

    # return the model
    return headModel


head_model = FCHeadNet.build(base_model, 10, 256)

model = Model(inputs=base_model.input, outputs=head_model)

Dont_Want_to_train_all = True;


if (Dont_Want_to_train_all):

	for layer in base_model.layers:
		layer.trainable = False



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])