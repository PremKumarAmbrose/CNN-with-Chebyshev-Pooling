import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Lambda
from tensorflow.keras import activations
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#Preparation of MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
if keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)


#set number of categories
num_category = 10


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)


##Functional Model Layers
input_1=tf.keras.Input(shape=(28,28,1))                       
x = layers.Conv2D(32,3,activation='relu',						
	input_shape=input_shape)(input_1)
x = x = layers.Conv2D(64,3,activation='relu')(x)


##Chebyshev pooling  starts here
x_squared = layers.Lambda(lambda x: x**2)(x)
u=layers.AveragePooling2D(pool_size=(2, 2))(x)
temp = layers.AveragePooling2D(pool_size=(2, 2))(x_squared)
u_squared = layers.Lambda(lambda x: x**2) (u)
sigma_squared = layers.Subtract()([temp,u_squared])
t = layers.MaxPooling2D(pool_size=(2, 2))(x)
t = layers.Activation(activations.softplus)(t)
denominator_0=layers.Subtract()([t,u])
denominator_0 = layers.Lambda(lambda x: x**2) (denominator_0)
denominator = layers.Add()([sigma_squared,denominator_0])
output = Lambda(lambda x: x[0]/x[1])([sigma_squared,denominator])
##Chebyshev pooling stops here

output = layers.Dropout(0.25)(output)
output = layers.Flatten()(output)
output = layers.Dense(128,activation='relu')(output)
output = layers.Dropout(0.5)(output)
output = layers.Dense(num_category,activation='softmax')(output)
model = keras.Model(inputs = input_1, 
	outputs = output, name = "Chebyshev_pooling")


##Compiles the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
model.save("model.h5")            ##generates a h5 format model

##model_fit
batch_size = 128
num_epoch = 10
model_log = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_test, y_test))

model.save("model.h5")            ##generates a h5 format model
