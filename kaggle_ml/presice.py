
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
# Any results you write to the current directory are saved as output.
from keras.layers import Layer
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import os
import cv2
import string
import numpy as np
from IPython.core.debugger import set_trace
#Init main values
symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)
import matplotlib.pyplot as plt
import keras

dir = 'C:\\NeuroBrave\\Scrapy\\kaggle_ml\\data_presice\\samples'


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show(block=False)


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def create_model():
    img = layers.Input(shape=img_shape)  # Get image as an input and process it through some Convs

    x = layers.Conv2D(16,
                      (3, 3),
                      activation='relu',
                      padding='same',
                      name='Conv1')(img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    x = layers.Conv2D(32,
                      (3, 3),
                      activation='relu',
                      padding='same',
                      name='Conv1')(img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Second conv block
    x = layers.Conv2D(64,
                      (3, 3),
                      activation='relu',
                      padding='same',
                      name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.BatchNormalization()(x)
    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs

    new_shape = ((50 // 4), (200 // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)
    # Get flattened vector and make 5 branches from it. Each branch will predict one letter
    flat = layers.Flatten()(x)
    outs = []

    for _ in range(5):
        rbf = RBFLayer(12, 0.5)(flat)  # addition of RBF layer
        dens1 = layers.Dense(64, activation='sigmoid')(rbf)
        drop = layers.Dropout(0.2)(dens1)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)

        outs.append(res)

    # Compile model and return it
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    #  set_trace();

    return model


model = create_model()


def preprocess_data():
    n_samples = len(os.listdir(dir))
    X = np.zeros((n_samples, 50, 200, 1))  # 1070*50*200
    y = np.zeros((5, n_samples, num_symbols))  # 5*1070*36

    for i, pic in enumerate(os.listdir(dir)):
        # Read image as grayscale
        img = cv2.imread(os.path.join(dir, pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs

    # Return final data
    return X, y


X, y = preprocess_data()
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]
model.summary()

history = LossHistory()
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]],validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]), batch_size=32,
                 epochs=10,verbose=0,callbacks=[history])
score = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]], verbose=1)
def plot_accuracy(history):
    # summarize history for accuracy
    histo = history.history
    train_hist = np.mean(np.column_stack((histo['dense_1_accuracy'],histo['dense_3_accuracy'],histo['dense_5_accuracy'],histo['dense_7_accuracy'],histo['dense_9_accuracy'])),1)
    val_hist = np.mean(np.column_stack((histo['val_dense_1_accuracy'],histo['val_dense_3_accuracy'],histo['val_dense_5_accuracy'],histo['val_dense_7_accuracy'],histo['val_dense_9_accuracy'])),1)
    np.mean([histo['dense_1_accuracy'],histo['dense_3_accuracy']])
    plt.figure()
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block = False)
# Define function to predict captcha
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt, sum(probs) / 5


import matplotlib.pyplot as plt
img=cv2.imread(os.path.join(dir,'33f7m.png'),cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print(['33f7m Predict: ' , predict(os.path.join(dir,'33f7m.png'))])
print(['2b827 Predict: ' , predict(os.path.join(dir,'2b827.png'))])

history.loss_plot('epoch')
plot_accuracy(hist)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model1.png',show_shapes=True)
print('a')
