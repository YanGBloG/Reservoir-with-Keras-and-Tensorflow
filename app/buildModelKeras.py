import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 0: ALL, 1: WARNING + ERR, 2: ERR, 3: NOTHING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

##import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation


class RegressionModel():
    def __init__(self, link_to_dataset, list_of_hidden_nodes, test_size, learning_rate, epochs, batch_size, opt_func):
        self.link_to_dataset = link_to_dataset
        self.list_of_hidden_nodes = list_of_hidden_nodes
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt_func = opt_func
        self.history = []
        self.model = []
        self.input_shape = self.get_input_shape()

    def split_to_train_and_test_set(self):
        data = pd.read_csv(self.link_to_dataset).values
        X_g = data[:, 0:-1]
        y_g = data[:, -2:-1]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_g)

        X_s = X_sc[0:1000]
        X_test = X_sc[1000:]

        y_s = y_g[0:1000].flatten()
        y_test = y_g[1000:].flatten()

        X_train, X_valid, y_train, y_valid = train_test_split(X_s, y_s, test_size = self.test_size)

        depth = data[1000:][:, 0:1]
        
        return X_train, X_valid, y_train, y_valid, X_test, y_test, depth

    def get_input_shape(self):
        X_train, X_valid, y_train, y_valid, _, _, _ = self.split_to_train_and_test_set()
        return X_train.shape[1]

    def networkx(self):
        nodes_list = self.list_of_hidden_nodes
        condition = len(nodes_list)
        
        model = Sequential()
        if condition == 1:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))

        elif condition == 2:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))

        elif condition == 3:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))

        elif condition == 4:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))

        elif condition == 5:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))
            model.add(Dense(nodes_list[4], activation='relu'))

        elif condition == 6:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))
            model.add(Dense(nodes_list[4], activation='relu'))
            model.add(Dense(nodes_list[5], activation='relu'))

        elif condition == 7:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))
            model.add(Dense(nodes_list[4], activation='relu'))
            model.add(Dense(nodes_list[5], activation='relu'))
            model.add(Dense(nodes_list[6], activation='relu'))

        elif condition == 8:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))
            model.add(Dense(nodes_list[4], activation='relu'))
            model.add(Dense(nodes_list[5], activation='relu'))
            model.add(Dense(nodes_list[6], activation='relu'))
            model.add(Dense(nodes_list[7], activation='relu'))

        elif condition == 9:
            model.add(Dense(nodes_list[0], input_shape=(self.input_shape, ), activation='relu'))
            model.add(Dense(nodes_list[1], activation='relu'))
            model.add(Dense(nodes_list[2], activation='relu'))
            model.add(Dense(nodes_list[3], activation='relu'))
            model.add(Dense(nodes_list[4], activation='relu'))
            model.add(Dense(nodes_list[5], activation='relu'))
            model.add(Dense(nodes_list[6], activation='relu'))
            model.add(Dense(nodes_list[7], activation='relu'))
            model.add(Dense(nodes_list[8], activation='relu'))
            
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=self.opt_func)
        return model

    def trainModel(self):
        X_train, X_valid, y_train, y_valid, X_test, y_test, depth = self.split_to_train_and_test_set()
        model = self.networkx()
        history = model.fit(X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=[X_valid, y_valid],
                    verbose=1)
        if not self.history:
            self.history.append(history)
            self.model.append(model)
        else:
            self.history = []
            self.model = []
            self.history.append(history)
            self.model.append(model)

    def model_and_history(self):
        model = self.model[0]
        history = self.history[0]
        return model, history

 