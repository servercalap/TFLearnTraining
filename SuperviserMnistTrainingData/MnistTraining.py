from __future__ import division,print_function,absolute_import
import tflearn
import numpy as np
import scipy


import tflearn.datasets.mnist as minst

X, Y, testX, testY = minst.load_data(one_hot=True)

input_layer = tflearn.input_data(shape=[None, 784])


hidden_layer1 = tflearn.fully_connected(input_layer, 128, activation='relu', regularizer='L2', weight_decay=0.001)

dropoutFirst = tflearn.dropout(hidden_layer1, 0.8)
hidden_layer2 = tflearn.fully_connected(dropoutFirst, 128, activation='relu', regularizer='L2', weight_decay=0.001)

dropoutSecond = tflearn.dropout(hidden_layer2, 0.8)


softmax = tflearn.fully_connected(dropoutSecond, 10, activation='softmax')

sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)

top_key = tflearn.metrics.Top_k(3)
training =tflearn.regression(softmax, optimizer=sgd, metric=top_key, loss='categorical_crossentropy')

model = tflearn.DNN(training, tensorboard_verbose=0)


model.fit(X, Y, n_epoch=10, validation_set=(testX, testY), show_metric=True, run_id='dense_model')

data = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("six.png", flatten=True)))
data = np.array(data).reshape(1, 784)
print(model.predict(data))
