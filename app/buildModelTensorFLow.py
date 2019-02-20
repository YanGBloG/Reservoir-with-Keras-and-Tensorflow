import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 0: ALL, 1: WARNING + ERR, 2: ERR, 3: NOTHING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

class RegressionModel():
    def __init__(self, link_to_dataset, list_of_hidden_nodes, test_size, learning_rate, epochs, batch_size):
        self.link_to_dataset = link_to_dataset
        self.list_of_hidden_nodes = list_of_hidden_nodes
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = self.get_input_shape()
        self.X = tf.placeholder('float', [None, self.input_shape])
        self.Y = tf.placeholder('float', [None, 1])
        self.losses = []
        self.val_losses = []

    def split_into_train_and_test_set(self):
        data = pd.read_csv(self.link_to_dataset).values
        X_g = data[:, 0:-1]
        y_g = data[:, -1]
        scaler = StandardScaler()
        X_new = scaler.fit_transform(X_g)
        X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_g, test_size = self.test_size)
        return X_train, X_valid, y_train, y_valid

    def get_input_shape(self):
        X_train, X_valid, y_train, y_valid = self.split_into_train_and_test_set()
        return X_train.shape[1]

    def networkx(self):
        input_layer = tf.Variable(tf.random_normal([self.input_shape, self.list_of_hidden_nodes[0]]))
        output_layer = tf.Variable(tf.random_normal([self.list_of_hidden_nodes[-1], 1]))

        weights = {}
        biases = {}

        output_bias = tf.Variable(tf.random_normal([1]))

        for i in range(len(self.list_of_hidden_nodes) - 1):
            weights['h{}'.format(i+1)] = tf.Variable(tf.random_normal([self.list_of_hidden_nodes[i], self.list_of_hidden_nodes[i+1]]))

        for i in range(len(self.list_of_hidden_nodes)):
            biases['b{}'.format(i)] = tf.Variable(tf.random_normal([self.list_of_hidden_nodes[i]]))

        if len(self.list_of_hidden_nodes) == 1:
            last_layer = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 2:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            last_layer = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 3:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            last_layer = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 4:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            last_layer = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 5:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            layer_4 = tf.nn.relu(layer_4)

            last_layer = tf.add(tf.matmul(layer_4, weights['h4']), biases['b4'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 6:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            layer_4 = tf.nn.relu(layer_4)

            layer_5 = tf.add(tf.matmul(layer_4, weights['h4']), biases['b4'])
            layer_5 = tf.nn.relu(layer_5)

            last_layer = tf.add(tf.matmul(layer_5, weights['h5']), biases['b5'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 7:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            layer_4 = tf.nn.relu(layer_4)

            layer_5 = tf.add(tf.matmul(layer_4, weights['h4']), biases['b4'])
            layer_5 = tf.nn.relu(layer_5)

            layer_6 = tf.add(tf.matmul(layer_5, weights['h5']), biases['b5'])
            layer_6 = tf.nn.relu(layer_6)

            last_layer = tf.add(tf.matmul(layer_6, weights['h6']), biases['b6'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 8:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            layer_4 = tf.nn.relu(layer_4)

            layer_5 = tf.add(tf.matmul(layer_4, weights['h4']), biases['b4'])
            layer_5 = tf.nn.relu(layer_5)

            layer_6 = tf.add(tf.matmul(layer_5, weights['h5']), biases['b5'])
            layer_6 = tf.nn.relu(layer_6)

            layer_7 = tf.add(tf.matmul(layer_6, weights['h6']), biases['b6'])
            layer_7 = tf.nn.relu(layer_7)

            last_layer = tf.add(tf.matmul(layer_7, weights['h7']), biases['b7'])
            last_layer = tf.nn.relu(last_layer)

        elif len(self.list_of_hidden_nodes) == 9:
            layer_1 = tf.add(tf.matmul(self.X, input_layer), biases['b0'])
            layer_1 = tf.nn.relu(layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
            layer_2 = tf.nn.relu(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
            layer_3 = tf.nn.relu(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
            layer_4 = tf.nn.relu(layer_4)

            layer_5 = tf.add(tf.matmul(layer_4, weights['h4']), biases['b4'])
            layer_5 = tf.nn.relu(layer_5)

            layer_6 = tf.add(tf.matmul(layer_5, weights['h5']), biases['b5'])
            layer_6 = tf.nn.relu(layer_6)

            layer_7 = tf.add(tf.matmul(layer_6, weights['h6']), biases['b6'])
            layer_7 = tf.nn.relu(layer_7)

            layer_8 = tf.add(tf.matmul(layer_7, weights['h7']), biases['b7'])
            layer_8 = tf.nn.relu(layer_8)

            last_layer = tf.add(tf.matmul(layer_8, weights['h8']), biases['b8'])
            last_layer = tf.nn.relu(last_layer)

        output_mat = tf.add(tf.matmul(last_layer, output_layer), output_bias)

        return output_mat, output_bias

    def next_batch(self, num, data, label):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        label_shuffle = [label[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(label_shuffle).reshape(-1, 1)

    def trainModel(self):
        X_train, X_valid, y_train, y_valid = self.split_into_train_and_test_set()
        model, w_out = self.networkx()

        loss = tf.reduce_mean(tf.square(self.Y - model))
        optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.epochs):
                x_batch, y_batch = self.next_batch(self.batch_size, X_train, y_train)
                sess.run([train, loss], feed_dict={self.X: x_batch, self.Y: y_batch})

                self.losses.append(sess.run(loss, feed_dict={self.X: x_batch, self.Y: y_batch}))
                self.val_losses.append(sess.run(loss, feed_dict={self.X: X_valid, self.Y: y_valid.reshape(-1, 1)}))
                print('Epoch: {}'.format(i), 'Loss: {:.4f}'.format(self.losses[i]), 'Val_loss: {:.4f}'.format(self.val_losses[i]))
        
            print('Optimization Finished!')


    def LossAndValidation(self):
        return self.losses, self.val_losses
