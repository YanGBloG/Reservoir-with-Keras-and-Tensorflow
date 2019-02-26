import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from GasPropertiesCalculation import *
from WaterPropertiesCalculation import *

from nxArchitecture import DrawNN
from nxArchitecture import NeuralNetwork as nn

from PopupWindowScreen import PopupWindow
from PopupWindowScreen import ShowTrainLog

from buildModelKeras import RegressionModel

from Conventional import Typical

import ImagesResources


norm_font = QFont()
norm_font.setFamily("Helvetica")
norm_font.setPointSize(10)

large_font = QFont()
large_font.setFamily("Helvetica")
large_font.setPointSize(12)

norm_font_bold = QFont()
norm_font_bold.setFamily("Helvetica")
norm_font_bold.setPointSize(10)
norm_font_bold.setBold(True)

class cnmApp(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)
        QMainWindow.setObjectName(self, "MainWindow")

        self.setMinimumSize(QSize(1275, 625))    
        self.setWindowTitle("Reservoir Engineering with TensorFlow")
        self.setWindowIcon(QIcon(':/fig/touch.png'))

        ### Add-in Parameters
        self.file_link = []
        self.history = []
        self.model = []
        self.predicted_data = []

        ### GUI
        self.statusBar()
        mainMenu = self.menuBar()
        
        Menu = mainMenu.addMenu('Menu')
        Open = Menu.addAction('Open')
        Save = Menu.addAction('Save')
        Exit = Menu.addAction('Exit')
        
        File = mainMenu.addMenu('File')
        Edit = mainMenu.addMenu('Edit')
        View = mainMenu.addMenu('View')
        Properties = mainMenu.addMenu('Properties')
        Help = mainMenu.addMenu('Help')


        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.setFont(norm_font)

        self.typical = Typical()
        self.ann = QWidget()
        self.results = QWidget()

        self.tabs.addTab(self.typical, "Typical")
        self.tabs.addTab(self.ann, "Artificial")
        self.tabs.addTab(self.results, "Results")

### FIRST TAB DISPLAY

### SECOND TAB DISPLAY
        self.batch_size_label = QLabel('Batch size:', self.ann)
        self.batch_size_label.setGeometry(QRect(10, 0, 71, 31))
        self.current_batch_size = QLineEdit('0', self.ann)
        self.current_batch_size.setGeometry(QRect(80, 6, 31, 22))
        self.current_batch_size.setReadOnly(True)
        self.current_batch_size.setAlignment(Qt.AlignRight)
        self.current_batch_size.setStyleSheet("border:none")
        
        self.slider_to_set_batch_size = QSlider(Qt.Horizontal, self.ann)
        self.slider_to_set_batch_size.setGeometry(QRect(10, 40, 131, 22))
        self.slider_to_set_batch_size.setCursor(QCursor(Qt.PointingHandCursor))
        self.slider_to_set_batch_size.setMinimum(0)
        self.slider_to_set_batch_size.setMaximum(100)
        self.slider_to_set_batch_size.setTickPosition(QSlider.TicksBelow)
        self.slider_to_set_batch_size.setTickInterval(11)
        self.slider_to_set_batch_size.valueChanged.connect(self.batch_size_change)
        
        self.learning_rate_label = QLabel('Learning rate', self.ann)
        self.learning_rate_label.setGeometry(QRect(170, 0, 81, 31))
        
        self.current_learning_rate = QLineEdit(self.ann)
        self.current_learning_rate.setGeometry(QRect(170, 40, 131, 22))
        self.current_learning_rate.setAlignment(Qt.AlignRight)
        self.current_learning_rate.setText('0.0001')

        self.regularization_rate_label = QLabel('Regular rate', self.ann)
        self.regularization_rate_label.setGeometry(QRect(330, 0, 121, 31))
        
        self.current_regularization_rate = QLineEdit(self.ann)
        self.current_regularization_rate.setGeometry(QRect(330, 40, 131, 22))
        self.current_regularization_rate.setAlignment(Qt.AlignRight)
        self.current_regularization_rate.setText('0.001')

        self.activation_label = QLabel('Activation', self.ann)
        self.activation_label.setGeometry(QRect(490, 0, 71, 31))

        activation_list = [ 'sigmoid', 'relu', 'tanh', 'linear', 'softmax']
        self.activation_entry = QComboBox(self.ann)
        self.activation_entry.setGeometry(QRect(490, 40, 131, 22))
        for item in activation_list:
            self.activation_entry.addItem(item)

        self.optimizer_label = QLabel('Optimizer', self.ann)
        self.optimizer_label.setGeometry(QRect(650, 0, 91, 31))
        
        optimizer_function_list = ['adam', 'adagrad', 'adadelta', 'rmsprop', 'sgd']
        self.optimizer_function_entry = QComboBox(self.ann)
        self.optimizer_function_entry.setGeometry(QRect(650, 40, 131, 22))
        for item in optimizer_function_list:
            self.optimizer_function_entry.addItem(item)

        self.loss_function_label = QLabel('Loss', self.ann)
        self.loss_function_label.setGeometry(QRect(810, 0, 91, 31))
        
        loss_function_list = ['Cross Enropy', 'mse']
        self.loss_function_entry = QComboBox(self.ann)
        self.loss_function_entry.setGeometry(QRect(810, 40, 131, 22))
        for item in loss_function_list:
            self.loss_function_entry.addItem(item)

        self.regularization_label = QLabel('Regularization', self.ann)
        self.regularization_label.setGeometry(QRect(970, 0, 91, 31))
        
        regular_list = ['None', 'L1', 'L2']
        self.regular_entry = QComboBox(self.ann)
        self.regular_entry.setGeometry(QRect(970, 40, 131, 22))
        for item in regular_list:
            self.regular_entry.addItem(item)
        
        self.problem_type_label = QLabel('Problem type', self.ann)
        self.problem_type_label.setGeometry(QRect(1130, 0, 91, 31))
        
        problem_type_list = ['Regression', 'Classification']
        self.problem_type_entry = QComboBox(self.ann)
        self.problem_type_entry.setGeometry(QRect(1130, 40, 131, 22))
        for item in problem_type_list:
            self.problem_type_entry.addItem(item)

        self.epoch_label = QLabel('Epoch:', self.ann)
        self.epoch_label.setGeometry(QRect(10, 72, 51, 31))
        self.current_epoch = QLineEdit('0', self.ann)
        self.current_epoch.setGeometry(QRect(50, 77, 41, 22))
        self.current_epoch.setReadOnly(True)
        self.current_epoch.setAlignment(Qt.AlignRight)
        self.current_epoch.setStyleSheet("border:none")
        
        self.slider_to_set_epoch = QSlider(Qt.Horizontal, self.ann)
        self.slider_to_set_epoch.setGeometry(QRect(10, 110, 131, 22))
        self.slider_to_set_epoch.setCursor(QCursor(Qt.PointingHandCursor))
        self.slider_to_set_epoch.setMinimum(0)
        self.slider_to_set_epoch.setMaximum(1500)
        self.slider_to_set_epoch.setTickPosition(QSlider.TicksBelow)
        self.slider_to_set_epoch.setTickInterval(100)
        self.slider_to_set_epoch.valueChanged.connect(self.number_of_epochs_change)

        return_icon = QIcon()
        return_icon.addPixmap(QPixmap(":/fig/return.png"), QIcon.Normal, QIcon.Off)        
        self.returnbutton = QPushButton(self.ann)
        self.returnbutton.setGeometry(QRect(170, 90, 31, 31))
        self.returnbutton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.returnbutton.setStyleSheet("QPushButton {color: #333; border-radius: 15px; border-style: outset; padding: 5px; }"
                                        "QPushButton:pressed { background-color: #D9D9D8}")
        self.returnbutton.setIcon(return_icon)
        self.returnbutton.setIconSize(QSize(25, 25))
            
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap(":/fig/icons_play.png"), QIcon.Normal, QIcon.Off)        
        self.playbutton = QPushButton(self.ann)
        self.playbutton.setGeometry(QRect(210, 80, 51, 51))
        self.playbutton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.playbutton.setStyleSheet("QPushButton { background-color: #F0F0F0; border-radius: 25px; border-style: outset;}"
                                      "QPushButton:pressed { background-color: #608B9B}")
        self.playbutton.setIcon(play_icon)
        self.playbutton.setIconSize(QSize(55, 55))
        self.playbutton.clicked.connect(self.train_model_process)

        stop_icon = QIcon()
        stop_icon.addPixmap(QPixmap(":/fig/stop.png"), QIcon.Normal, QIcon.Off)
        self.stopbutton = QPushButton(self.ann)
        self.stopbutton.setGeometry(QRect(270, 90, 31, 31))
        self.stopbutton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.stopbutton.setStyleSheet("QPushButton {color: #333; border-radius: 15px; border-style: outset; padding: 5px; }"
                                      "QPushButton:pressed { background-color: #D9D9D8}")
        self.stopbutton.setIcon(stop_icon)
        self.stopbutton.setIconSize(QSize(40, 40))
        self.stopbutton.clicked.connect(self.viewLossWithRealTime)

        plus_icon = QIcon()
        plus_icon.addPixmap(QPixmap(":/fig/plus.png"), QIcon.Normal, QIcon.Off)
        self.plus_layer = QPushButton(self.ann)
        self.plus_layer.setGeometry(QRect(340, 80, 51, 51))
        self.plus_layer.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.plus_layer.setStyleSheet("QPushButton { background-color: #F0F0F0; border-radius: 25px; border-style: outset; color: #333}"
                                      "QPushButton:pressed { background-color: #608B9B}")
        self.plus_layer.setIcon(plus_icon)
        self.plus_layer.setIconSize(QSize(40, 40))
        self.plus_layer.clicked.connect(self.add_model_hidden_layer)
        self.plus_layer.clicked.connect(self.makeNetworkArchitecture)

        minus_icon = QIcon()
        minus_icon.addPixmap(QPixmap(":/fig/minus.png"), QIcon.Normal, QIcon.Off)
        self.minus_layer = QPushButton(self.ann)
        self.minus_layer.setGeometry(QRect(400, 80, 51, 51))
        self.minus_layer.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_layer.setStyleSheet("QPushButton { background-color: #F0F0F0; border-radius: 25px; border-style: outset; color: #333}"
                                       "QPushButton:pressed { background-color: #608B9B}")
        self.minus_layer.setIcon(minus_icon)
        self.minus_layer.setIconSize(QSize(40, 40))
        self.minus_layer.clicked.connect(self.minus_model_hidden_layer)
        self.minus_layer.clicked.connect(self.makeNetworkArchitecture)

        add_nodes_icon = QIcon()
        add_nodes_icon.addPixmap(QPixmap(":/fig/add_neurons.png"), QIcon.Normal, QIcon.Off)

        minus_nodes_icon = QIcon()
        minus_nodes_icon.addPixmap(QPixmap(":/fig/minus_neurons.png"), QIcon.Normal, QIcon.Off)

    ### Input layer
        self.add_neuron_input = QPushButton(self.ann)
        self.add_neuron_input.setGeometry(QRect(468, 70, 31, 31))
        self.add_neuron_input.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_input.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_input.setIcon(add_nodes_icon)
        self.add_neuron_input.setIconSize(QSize(25, 25))
        self.add_neuron_input.clicked.connect(self.add_neuron_to_input_layer)
        self.add_neuron_input.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_input = QPushButton(self.ann)
        self.minus_neuron_input.setGeometry(QRect(502, 70, 31, 31))
        self.minus_neuron_input.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_input.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                              "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_input.setIcon(minus_nodes_icon)
        self.minus_neuron_input.setIconSize(QSize(25, 25))
        self.minus_neuron_input.clicked.connect(self.minus_neuron_to_input_layer)
        self.minus_neuron_input.clicked.connect(self.makeNetworkArchitecture)

        self.inputLayer_nodes = QLineEdit(self.ann)
        self.inputLayer_nodes.setGeometry(QRect(470, 110, 61, 21))
        self.inputLayer_nodes.setReadOnly(True)
        self.inputLayer_nodes.setAlignment(Qt.AlignCenter)
        self.inputLayer_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")
        self.inputLayer_nodes.setText('1')

    ### Hidden layer 1
        self.add_neuron_hl1 = QPushButton(self.ann)
        self.add_neuron_hl1.setGeometry(QRect(548, 70, 31, 31))
        self.add_neuron_hl1.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl1.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl1.setIcon(add_nodes_icon)
        self.add_neuron_hl1.setIconSize(QSize(25, 25))
        self.add_neuron_hl1.setEnabled(False)
        self.add_neuron_hl1.clicked.connect(self.add_neuron_to_hidden_layer_1)
        self.add_neuron_hl1.clicked.connect(self.makeNetworkArchitecture)
        
        self.minus_neuron_hl1 = QPushButton(self.ann)
        self.minus_neuron_hl1.setGeometry(QRect(582, 70, 31, 31))
        self.minus_neuron_hl1.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl1.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl1.setIcon(minus_nodes_icon)
        self.minus_neuron_hl1.setIconSize(QSize(25, 25))
        self.minus_neuron_hl1.setEnabled(False)
        self.minus_neuron_hl1.clicked.connect(self.minus_neuron_to_hidden_layer_1)
        self.minus_neuron_hl1.clicked.connect(self.makeNetworkArchitecture)

        self.hidden1_nodes = QLineEdit(self.ann)
        self.hidden1_nodes.setGeometry(QRect(550, 110, 61, 21))
        self.hidden1_nodes.setReadOnly(True)
        self.hidden1_nodes.setAlignment(Qt.AlignCenter)
        self.hidden1_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 2
        self.add_neuron_hl2 = QPushButton(self.ann)
        self.add_neuron_hl2.setGeometry(QRect(628, 70, 31, 31))
        self.add_neuron_hl2.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl2.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl2.setIcon(add_nodes_icon)
        self.add_neuron_hl2.setIconSize(QSize(25, 25))
        self.add_neuron_hl2.setEnabled(False)
        self.add_neuron_hl2.clicked.connect(self.add_neuron_to_hidden_layer_2)
        self.add_neuron_hl2.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl2 = QPushButton(self.ann)
        self.minus_neuron_hl2.setGeometry(QRect(662, 70, 31, 31))
        self.minus_neuron_hl2.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl2.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl2.setIcon(minus_nodes_icon)
        self.minus_neuron_hl2.setIconSize(QSize(25, 25))
        self.minus_neuron_hl2.setEnabled(False)
        self.minus_neuron_hl2.clicked.connect(self.minus_neuron_to_hidden_layer_2)
        self.minus_neuron_hl2.clicked.connect(self.makeNetworkArchitecture)

        self.hidden2_nodes = QLineEdit(self.ann)
        self.hidden2_nodes.setGeometry(QRect(630, 110, 61, 21))
        self.hidden2_nodes.setReadOnly(True)
        self.hidden2_nodes.setAlignment(Qt.AlignCenter)
        self.hidden2_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 3
        self.add_neuron_hl3 = QPushButton(self.ann)
        self.add_neuron_hl3.setGeometry(QRect(708, 70, 31, 31))
        self.add_neuron_hl3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl3.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl3.setIcon(add_nodes_icon)
        self.add_neuron_hl3.setIconSize(QSize(25, 25))
        self.add_neuron_hl3.setEnabled(False)
        self.add_neuron_hl3.clicked.connect(self.add_neuron_to_hidden_layer_3)
        self.add_neuron_hl3.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl3 = QPushButton(self.ann)
        self.minus_neuron_hl3.setGeometry(QRect(742, 70, 31, 31))
        self.minus_neuron_hl3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl3.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl3.setIcon(minus_nodes_icon)
        self.minus_neuron_hl3.setIconSize(QSize(25, 25))
        self.minus_neuron_hl3.setEnabled(False)
        self.minus_neuron_hl3.clicked.connect(self.minus_neuron_to_hidden_layer_3)
        self.minus_neuron_hl3.clicked.connect(self.makeNetworkArchitecture)

        self.hidden3_nodes = QLineEdit(self.ann)
        self.hidden3_nodes.setGeometry(QRect(710, 110, 61, 21))
        self.hidden3_nodes.setReadOnly(True)
        self.hidden3_nodes.setAlignment(Qt.AlignCenter)
        self.hidden3_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 4
        self.add_neuron_hl4 = QPushButton(self.ann)
        self.add_neuron_hl4.setGeometry(QRect(788, 70, 31, 31))
        self.add_neuron_hl4.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl4.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl4.setIcon(add_nodes_icon)
        self.add_neuron_hl4.setIconSize(QSize(25, 25))
        self.add_neuron_hl4.setEnabled(False)
        self.add_neuron_hl4.clicked.connect(self.add_neuron_to_hidden_layer_4)
        self.add_neuron_hl4.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl4 = QPushButton(self.ann)
        self.minus_neuron_hl4.setGeometry(QRect(822, 70, 31, 31))
        self.minus_neuron_hl4.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl4.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl4.setIcon(minus_nodes_icon)
        self.minus_neuron_hl4.setIconSize(QSize(25, 25))
        self.minus_neuron_hl4.setEnabled(False)
        self.minus_neuron_hl4.clicked.connect(self.minus_neuron_to_hidden_layer_4)
        self.minus_neuron_hl4.clicked.connect(self.makeNetworkArchitecture)
        
        self.hidden4_nodes = QLineEdit(self.ann)
        self.hidden4_nodes.setGeometry(QRect(790, 110, 61, 21))
        self.hidden4_nodes.setReadOnly(True)
        self.hidden4_nodes.setAlignment(Qt.AlignCenter)
        self.hidden4_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 5
        self.add_neuron_hl5 = QPushButton(self.ann)
        self.add_neuron_hl5.setGeometry(QRect(868, 70, 31, 31))
        self.add_neuron_hl5.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl5.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl5.setIcon(add_nodes_icon)
        self.add_neuron_hl5.setIconSize(QSize(25, 25))
        self.add_neuron_hl5.setEnabled(False)
        self.add_neuron_hl5.clicked.connect(self.add_neuron_to_hidden_layer_5)
        self.add_neuron_hl5.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl5 = QPushButton(self.ann)
        self.minus_neuron_hl5.setGeometry(QRect(902, 70, 31, 31))
        self.minus_neuron_hl5.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl5.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl5.setIcon(minus_nodes_icon)
        self.minus_neuron_hl5.setIconSize(QSize(25, 25))
        self.minus_neuron_hl5.setEnabled(False)
        self.minus_neuron_hl5.clicked.connect(self.minus_neuron_to_hidden_layer_5)
        self.minus_neuron_hl5.clicked.connect(self.makeNetworkArchitecture)
        
        self.hidden5_nodes = QLineEdit(self.ann)
        self.hidden5_nodes.setGeometry(QRect(870, 110, 61, 21))
        self.hidden5_nodes.setReadOnly(True)
        self.hidden5_nodes.setAlignment(Qt.AlignCenter)
        self.hidden5_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 6
        self.add_neuron_hl6 = QPushButton(self.ann)
        self.add_neuron_hl6.setGeometry(QRect(948, 70, 31, 31))
        self.add_neuron_hl6.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl6.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl6.setIcon(add_nodes_icon)
        self.add_neuron_hl6.setIconSize(QSize(25, 25))
        self.add_neuron_hl6.setEnabled(False)
        self.add_neuron_hl6.clicked.connect(self.add_neuron_to_hidden_layer_6)
        self.add_neuron_hl6.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl6 = QPushButton(self.ann)
        self.minus_neuron_hl6.setGeometry(QRect(982, 70, 31, 31))
        self.minus_neuron_hl6.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl6.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl6.setIcon(minus_nodes_icon)
        self.minus_neuron_hl6.setIconSize(QSize(25, 25))
        self.minus_neuron_hl6.setEnabled(False)
        self.minus_neuron_hl6.clicked.connect(self.minus_neuron_to_hidden_layer_6)
        self.minus_neuron_hl6.clicked.connect(self.makeNetworkArchitecture)
        
        self.hidden6_nodes = QLineEdit(self.ann)
        self.hidden6_nodes.setGeometry(QRect(950, 110, 61, 21))
        self.hidden6_nodes.setReadOnly(True)
        self.hidden6_nodes.setAlignment(Qt.AlignCenter)
        self.hidden6_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 7
        self.add_neuron_hl7 = QPushButton(self.ann)
        self.add_neuron_hl7.setGeometry(QRect(1028, 70, 31, 31))
        self.add_neuron_hl7.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl7.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl7.setIcon(add_nodes_icon)
        self.add_neuron_hl7.setIconSize(QSize(25, 25))
        self.add_neuron_hl7.setEnabled(False)
        self.add_neuron_hl7.clicked.connect(self.add_neuron_to_hidden_layer_7)
        self.add_neuron_hl7.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl7 = QPushButton(self.ann)
        self.minus_neuron_hl7.setGeometry(QRect(1062, 70, 31, 31))
        self.minus_neuron_hl7.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl7.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl7.setIcon(minus_nodes_icon)
        self.minus_neuron_hl7.setIconSize(QSize(25, 25))
        self.minus_neuron_hl7.setEnabled(False)
        self.minus_neuron_hl7.clicked.connect(self.minus_neuron_to_hidden_layer_7)
        self.minus_neuron_hl7.clicked.connect(self.makeNetworkArchitecture)
        
        self.hidden7_nodes = QLineEdit(self.ann)
        self.hidden7_nodes.setGeometry(QRect(1030, 110, 61, 21))
        self.hidden7_nodes.setReadOnly(True)
        self.hidden7_nodes.setAlignment(Qt.AlignCenter)
        self.hidden7_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Hidden layer 8
        self.add_neuron_hl8 = QPushButton(self.ann)
        self.add_neuron_hl8.setGeometry(QRect(1108, 70, 31, 31))
        self.add_neuron_hl8.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_hl8.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_hl8.setIcon(add_nodes_icon)
        self.add_neuron_hl8.setIconSize(QSize(25, 25))
        self.add_neuron_hl8.setEnabled(False)
        self.add_neuron_hl8.clicked.connect(self.add_neuron_to_hidden_layer_8)
        self.add_neuron_hl8.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_hl8 = QPushButton(self.ann)
        self.minus_neuron_hl8.setGeometry(QRect(1142, 70, 31, 31))
        self.minus_neuron_hl8.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_hl8.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_hl8.setIcon(minus_nodes_icon)
        self.minus_neuron_hl8.setIconSize(QSize(25, 25))
        self.minus_neuron_hl8.setEnabled(False)
        self.minus_neuron_hl8.clicked.connect(self.minus_neuron_to_hidden_layer_8)
        self.minus_neuron_hl8.clicked.connect(self.makeNetworkArchitecture)
        
        self.hidden8_nodes = QLineEdit(self.ann)
        self.hidden8_nodes.setGeometry(QRect(1110, 110, 61, 21))
        self.hidden8_nodes.setReadOnly(True)
        self.hidden8_nodes.setAlignment(Qt.AlignCenter)
        self.hidden8_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")

    ### Output layer
        self.add_neuron_output_layer = QPushButton(self.ann)
        self.add_neuron_output_layer.setGeometry(QRect(1188, 70, 31, 31))
        self.add_neuron_output_layer.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.add_neuron_output_layer.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.add_neuron_output_layer.setIcon(add_nodes_icon)
        self.add_neuron_output_layer.setIconSize(QSize(25, 25))
        self.add_neuron_output_layer.clicked.connect(self.add_neuron_to_output_layer)
        self.add_neuron_output_layer.clicked.connect(self.makeNetworkArchitecture)

        self.minus_neuron_output_layer = QPushButton(self.ann)
        self.minus_neuron_output_layer.setGeometry(QRect(1222, 70, 31, 31))
        self.minus_neuron_output_layer.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.minus_neuron_output_layer.setStyleSheet("QPushButton {color: #333; border-radius: 15px; padding: 5px; background-color: #F0F0F0}"
                                            "QPushButton:pressed { background-color: #D9D9D8}")
        self.minus_neuron_output_layer.setIcon(minus_nodes_icon)
        self.minus_neuron_output_layer.setIconSize(QSize(25, 25))
        self.minus_neuron_output_layer.clicked.connect(self.minus_neuron_to_output_layer)
        self.minus_neuron_output_layer.clicked.connect(self.makeNetworkArchitecture)
        
        self.outputLayer_nodes = QLineEdit(self.ann)
        self.outputLayer_nodes.setGeometry(QRect(1190, 110, 61, 21))
        self.outputLayer_nodes.setReadOnly(True)
        self.outputLayer_nodes.setAlignment(Qt.AlignCenter)
        self.outputLayer_nodes.setStyleSheet("background-color: rgb(247, 247, 247);")
        self.outputLayer_nodes.setText('1')

    ### Theme widget tab 2
        self.lower_theme_three = QWidget(self.ann)
        self.lower_theme_three.setGeometry(QRect(0, 140, 1265, 415))
        self.lower_theme_three.setStyleSheet("background-color: rgb(247, 247, 247);")

        self.data_grid = QLabel('DATA', self.lower_theme_three)
        self.data_grid.setGeometry(QRect(20, 10, 61, 21))
        self.data_grid.setFont(large_font)

        self.class_label = QLabel('Classification', self.lower_theme_three)
        self.class_label.setGeometry(QRect(20, 40, 91, 21))

        iris_icon = QIcon()
        iris_icon.addPixmap(QPixmap(":/fig/iris.ico"), QIcon.Normal, QIcon.Off)
        self.iris_ds = QPushButton(self.lower_theme_three)
        self.iris_ds.setGeometry(QRect(20, 70, 51, 51))
        self.iris_ds.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.iris_ds.setIcon(iris_icon)
        self.iris_ds.setIconSize(QSize(50, 50))

        titanic_icon = QIcon()
        titanic_icon.addPixmap(QPixmap(":/fig/titanic.ico"), QIcon.Normal, QIcon.Off)
        self.titanic_ds = QPushButton(self.lower_theme_three)
        self.titanic_ds.setGeometry(QRect(80, 70, 51, 51))
        self.titanic_ds.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.titanic_ds.setIcon(titanic_icon)
        self.titanic_ds.setIconSize(QSize(45, 45))

        self.regression_label = QLabel('Regression', self.lower_theme_three)
        self.regression_label.setGeometry(QRect(20, 130, 91, 21))

        poro_icon = QIcon()
        poro_icon.addPixmap(QPixmap(":/fig/poro.ico"), QIcon.Normal, QIcon.Off)
        self.poro_ds = QPushButton(self.lower_theme_three)
        self.poro_ds.setGeometry(QRect(20, 160, 51, 51))
        self.poro_ds.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.poro_ds.setIcon(poro_icon)
        self.poro_ds.setIconSize(QSize(45, 45))

        houseprice_icon = QIcon()
        houseprice_icon.addPixmap(QPixmap(":/fig/houseprice.ico"), QIcon.Normal, QIcon.Off)
        self.houseprice_ds = QPushButton(self.lower_theme_three)
        self.houseprice_ds.setGeometry(QRect(80, 160, 51, 51))
        self.houseprice_ds.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.houseprice_ds.setIcon(houseprice_icon)
        self.houseprice_ds.setIconSize(QSize(45, 45))

        self.users_ds = QLabel('Users dataset', self.lower_theme_three)
        self.users_ds.setGeometry(QRect(20, 220, 91, 21))

        load_icon = QIcon()
        load_icon.addPixmap(QPixmap(":/fig/load_data.png"), QIcon.Normal, QIcon.Off)
        self.load_user_data = QPushButton('Load', self.lower_theme_three)
        self.load_user_data.setGeometry(QRect(20, 250, 91, 31))
        self.load_user_data.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.load_user_data.setIcon(load_icon)
        self.load_user_data.setIconSize(QSize(25, 25))
        self.load_user_data.clicked.connect(self.openFileNameDialog)
        
        self.test_size = QLabel('Test size:', self.lower_theme_three)
        self.test_size.setGeometry(QRect(20, 290, 61, 21))
        self.ts_size = QLineEdit('0%', self.lower_theme_three)
        self.ts_size.setGeometry(QRect(80, 290, 41, 21))
        self.ts_size.setReadOnly(True)
        self.ts_size.setAlignment(Qt.AlignRight)
        self.ts_size.setStyleSheet("background-color: rgb(247, 247, 247);border:none")
        
        self.slider_ts = QSlider(Qt.Horizontal, self.lower_theme_three)
        self.slider_ts.setGeometry(QRect(20, 320, 111, 22))
        self.slider_ts.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.slider_ts.setMinimum(0)
        self.slider_ts.setMaximum(100)
        self.slider_ts.setTickPosition(QSlider.TicksBelow)
        self.slider_ts.setTickInterval(11)
        self.slider_ts.valueChanged.connect(self.test_size_Change)

        self.save_model_button = QPushButton('SAVE MODEL', self.lower_theme_three)
        self.save_model_button.setGeometry(QRect(20, 360, 111, 31))
        self.save_model_button.setStyleSheet('QPushButton {color: #333; border: 2px; border-color: #2F3F73; border-style: outset; background-color: #F0F0F0;}'
                                             'QPushButton:pressed { background-color: #D9D9D8}')
        self.save_model_button.clicked.connect(self.save_model_after_train)

        ###
        ### Plot figure here
        # Frame 1
        self.network_config_label = QLabel('NETWORKS ARCHITECTURE', self.lower_theme_three)
        self.network_config_label.setGeometry(QRect(150, 10, 261, 21))
        self.network_config_label.setFont(large_font)
        
        fullscreen_icon = QIcon()
        fullscreen_icon.addPixmap(QPixmap(":/fig/fullScreen.png"), QIcon.Normal, QIcon.Off)

        self.netArch = QGraphicsView(self.lower_theme_three)
        self.netArch.setGeometry(QRect(150, 40, 541, 371))

        self.network_config_figure = plt.figure()
        self.network_architect_canvas = FigureCanvas(self.network_config_figure)

        self.nx_config_pseudo_layout = QWidget(self.lower_theme_three)
        self.nx_config_pseudo_layout.setGeometry(QRect(151, 41, 539, 369))

        nx_config_box = QVBoxLayout(self.nx_config_pseudo_layout)
        nx_config_box.addWidget(self.network_architect_canvas)
        
        self.fullScreen_net = QPushButton(self.lower_theme_three)
        self.fullScreen_net.setGeometry(QRect(660, 40, 31, 31))
        self.fullScreen_net.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.fullScreen_net.setStyleSheet("QPushButton {border: 2px; background-color: #F0F0F0; border-color: #2F3F73; border-style: outset}"
                                          "QPushButton:pressed { background-color: #D9D9D8}")
        self.fullScreen_net.setIcon(fullscreen_icon)
        self.fullScreen_net.setIconSize(QSize(30, 30))
        self.fullScreen_net.clicked.connect(self.zoomNetworkAchitecture)
        
        # Frame 2
        self.log_loss_label = QLabel('LOSS RECORD', self.lower_theme_three)
        self.log_loss_label.setGeometry(QRect(710, 10, 251, 21))
        self.log_loss_label.setFont(large_font)

        trainview_icon = QIcon()
        trainview_icon.addPixmap(QPixmap(":/fig/viewlog.png"), QIcon.Normal, QIcon.Off)

        self.log_loss = QGraphicsView(self.lower_theme_three)
        self.log_loss.setGeometry(QRect(710, 40, 541, 371))

        self.loss_record_figure = Figure()
        self.log_loss_canvas = FigureCanvas(self.loss_record_figure)

        self.log_loss_pseudo_layout = QWidget(self.lower_theme_three)
        self.log_loss_pseudo_layout.setGeometry(QRect(711, 41, 539, 369))

        log_loss_box = QVBoxLayout(self.log_loss_pseudo_layout)
        log_loss_box.addWidget(self.log_loss_canvas)

        self.trainview = QPushButton(self.lower_theme_three)
        self.trainview.setGeometry(QRect(1220, 40, 31, 31))
        self.trainview.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.trainview.setStyleSheet("QPushButton {border: 2px; background-color: #F0F0F0; border-color: #2F3F73; border-style: outset}"
                                     "QPushButton:pressed { background-color: #D9D9D8}")
        self.trainview.setIcon(trainview_icon)
        self.trainview.setIconSize(QSize(30, 30))
        self.trainview.clicked.connect(self.showTrainingLog)
        

### THIRD TAB DISPLAY
        self.y_label_index = QLineEdit(self.results)
        self.y_label_index.setGeometry(QRect(10, 10, 101, 31))

        self.load_trained_model = QPushButton('Load model', self.results)
        self.load_trained_model.setGeometry(QRect(120, 10, 101, 31))
        self.load_trained_model.clicked.connect(self.load_model_to_predict)

        self.load_new_data = QPushButton('Load data', self.results)
        self.load_new_data.setGeometry(QRect(230, 10, 101, 31))
        self.load_new_data.clicked.connect(self.load_predict_data)

        self.predict_data_button = QPushButton('Predict', self.results)
        self.predict_data_button.setGeometry(QRect(340, 10, 101, 31))
        self.predict_data_button.clicked.connect(self.model_predict)

        self.line = QFrame(self.results)
        self.line.setGeometry(QRect(5, 43, 435, 16))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.evaluation_label = QLabel('Evaluation', self.results)
        self.evaluation_label.setGeometry(QRect(3, 60, 441, 21))
        self.evaluation_label.setStyleSheet('background-color: #DEF0D8; border-radius: 10px; color: #387144;')
        self.evaluation_label.setFont(large_font)
        self.evaluation_label.setAlignment(Qt.AlignCenter)

        self.evaluation_view = QGraphicsView(self.results)
        self.evaluation_view.setGeometry(QRect(3, 90, 441, 425))

        self.evaluation_figure = Figure()
        self.evaluation_figure_canvas = FigureCanvas(self.evaluation_figure)

        self.evaluation_figure_layout = QWidget(self.results)
        self.evaluation_figure_layout.setGeometry(QRect(4, 91, 439, 423))

        evaluation_box = QVBoxLayout(self.evaluation_figure_layout)
        evaluation_box.addWidget(self.evaluation_figure_canvas)

        self.show_result_button = QPushButton('Show', self.results)
        self.show_result_button.setGeometry(QRect(120, 520, 101, 31))
        self.show_result_button.clicked.connect(self.plotPredictResult)
        self.show_result_button.clicked.connect(self.plotEvaluation)

        self.detail_button = QPushButton('Detail', self.results)
        self.detail_button.setGeometry(QRect(230, 520, 101, 31))
        self.detail_button.clicked.connect(self.showDetailResult)

        self.save_result_button = QPushButton('Save', self.results)
        self.save_result_button.setGeometry(QRect(340, 520, 101, 31))
        self.save_result_button.clicked.connect(self.SavePredictResult)

        self.result_view = QGraphicsView(self.results)
        self.result_view.setGeometry(QRect(446, 0, 821, 553))

        self.result_figure = Figure()
        self.result_figure_canvas = FigureCanvas(self.result_figure)

        self.result_figure_layout = QWidget(self.results)
        self.result_figure_layout.setGeometry(QRect(447, 1, 819, 551))

        result_figure_box = QVBoxLayout(self.result_figure_layout)
        result_figure_box.addWidget(self.result_figure_canvas)
        

### FIRST TAB CONFIGURATION
        

### SECOND TAB CONFIGURATION
    ## Top theme
    def batch_size_change(self):
        values = self.slider_to_set_batch_size.value()
        str_values = str(values)
        self.current_batch_size.setText('{}'.format(str_values))
        return values

    def number_of_epochs_change(self):
        values = self.slider_to_set_epoch.value()
        str_values = str(values)
        self.current_epoch.setText('{}'.format(str_values))
        return values

    def train_model_process(self):
        while True:
            try:
                link_to_dataset = self.file_link[0]
                test_size = float(self.test_size_Change())/100
                learning_rate = float(self.current_learning_rate.text())
                epochs = self.number_of_epochs_change()
                batch_size = self.batch_size_change()
                list_of_hidden_nodes = self.number_of_neurons_in_layer()[:-1]
                loss_func = self.loss_function_entry.currentText()
                optimizer = self.optimizer_function_entry.currentText()

                model = RegressionModel(link_to_dataset, list_of_hidden_nodes, test_size, learning_rate, epochs, batch_size, loss_func, optimizer)
                model.trainModel()
                model, history = model.model_and_history()
                if not self.model:
                    self.model.append(model)
                    self.history.append(history)
                else:
                    self.model = []
                    self.history = []
                    self.model.append(model)
                    self.history.append(history)
                
                self.popupmsg('Optimization process finished!', 'info')
                break
            except:
                self.popupmsg('Something wrong in your process', 'warning')
                break

    def add_model_hidden_layer(self):
        if self.add_neuron_hl1.isEnabled() == False:
            self.add_neuron_hl1.setEnabled(True)
            self.minus_neuron_hl1.setEnabled(True)
            self.hidden1_nodes.setText('1')
        elif self.add_neuron_hl2.isEnabled() == False:
            self.add_neuron_hl2.setEnabled(True)
            self.minus_neuron_hl2.setEnabled(True)
            self.hidden2_nodes.setText('1')
        elif self.add_neuron_hl3.isEnabled() == False:
            self.add_neuron_hl3.setEnabled(True)
            self.minus_neuron_hl3.setEnabled(True)
            self.hidden3_nodes.setText('1')
        elif self.add_neuron_hl4.isEnabled() == False:
            self.add_neuron_hl4.setEnabled(True)
            self.minus_neuron_hl4.setEnabled(True)
            self.hidden4_nodes.setText('1')
        elif self.add_neuron_hl5.isEnabled() == False:
            self.add_neuron_hl5.setEnabled(True)
            self.minus_neuron_hl5.setEnabled(True)
            self.hidden5_nodes.setText('1')
        elif self.add_neuron_hl6.isEnabled() == False:
            self.add_neuron_hl6.setEnabled(True)
            self.minus_neuron_hl6.setEnabled(True)
            self.hidden6_nodes.setText('1')
        elif self.add_neuron_hl7.isEnabled() == False:
            self.add_neuron_hl7.setEnabled(True)
            self.minus_neuron_hl7.setEnabled(True)
            self.hidden7_nodes.setText('1')
        elif self.add_neuron_hl8.isEnabled() == False:
            self.add_neuron_hl8.setEnabled(True)
            self.minus_neuron_hl8.setEnabled(True)
            self.hidden8_nodes.setText('1')

    def minus_model_hidden_layer(self):
        if self.add_neuron_hl8.isEnabled():
            self.add_neuron_hl8.setEnabled(False)
            self.minus_neuron_hl8.setEnabled(False)
            self.hidden8_nodes.setText('')
        elif self.add_neuron_hl7.isEnabled():
            self.add_neuron_hl7.setEnabled(False)
            self.minus_neuron_hl7.setEnabled(False)
            self.hidden7_nodes.setText('')
        elif self.add_neuron_hl6.isEnabled():
            self.add_neuron_hl6.setEnabled(False)
            self.minus_neuron_hl6.setEnabled(False)
            self.hidden6_nodes.setText('')
        elif self.add_neuron_hl5.isEnabled():
            self.add_neuron_hl5.setEnabled(False)
            self.minus_neuron_hl5.setEnabled(False)
            self.hidden5_nodes.setText('')
        elif self.add_neuron_hl4.isEnabled():
            self.add_neuron_hl4.setEnabled(False)
            self.minus_neuron_hl4.setEnabled(False)
            self.hidden4_nodes.setText('')
        elif self.add_neuron_hl3.isEnabled():
            self.add_neuron_hl3.setEnabled(False)
            self.minus_neuron_hl3.setEnabled(False)
            self.hidden3_nodes.setText('')
        elif self.add_neuron_hl2.isEnabled():
            self.add_neuron_hl2.setEnabled(False)
            self.minus_neuron_hl2.setEnabled(False)
            self.hidden2_nodes.setText('')
        elif self.add_neuron_hl1.isEnabled():
            self.add_neuron_hl1.setEnabled(False)
            self.minus_neuron_hl1.setEnabled(False)
            self.hidden1_nodes.setText('')

    def add_neuron_to_input_layer(self):
        number_of_neurons = int(self.inputLayer_nodes.text())
        self.inputLayer_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_input_layer(self):
        number_of_neurons = int(self.inputLayer_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.inputLayer_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.inputLayer_nodes.setText('1')

    def add_neuron_to_hidden_layer_1(self):
        number_of_neurons = int(self.hidden1_nodes.text())
        self.hidden1_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_1(self):
        number_of_neurons = int(self.hidden1_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden1_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden1_nodes.setText('1')

    def add_neuron_to_hidden_layer_2(self):
        number_of_neurons = int(self.hidden2_nodes.text())
        self.hidden2_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_2(self):
        number_of_neurons = int(self.hidden2_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden2_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden2_nodes.setText('1')

    def add_neuron_to_hidden_layer_3(self):
        number_of_neurons = int(self.hidden3_nodes.text())
        self.hidden3_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_3(self):
        number_of_neurons = int(self.hidden3_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden3_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden3_nodes.setText('1')

    def add_neuron_to_hidden_layer_4(self):
        number_of_neurons = int(self.hidden4_nodes.text())
        self.hidden4_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_4(self):
        number_of_neurons = int(self.hidden4_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden4_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden4_nodes.setText('1')

    def add_neuron_to_hidden_layer_5(self):
        number_of_neurons = int(self.hidden5_nodes.text())
        self.hidden5_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_5(self):
        number_of_neurons = int(self.hidden5_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden5_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden5_nodes.setText('1')

    def add_neuron_to_hidden_layer_6(self):
        number_of_neurons = int(self.hidden6_nodes.text())
        self.hidden6_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_6(self):
        number_of_neurons = int(self.hidden6_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden6_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden6_nodes.setText('1')

    def add_neuron_to_hidden_layer_7(self):
        number_of_neurons = int(self.hidden7_nodes.text())
        self.hidden7_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_7(self):
        number_of_neurons = int(self.hidden7_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden7_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden7_nodes.setText('1')

    def add_neuron_to_hidden_layer_8(self):
        number_of_neurons = int(self.hidden8_nodes.text())
        self.hidden8_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_hidden_layer_8(self):
        number_of_neurons = int(self.hidden8_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.hidden8_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.hidden8_nodes.setText('1')

    def add_neuron_to_output_layer(self):
        number_of_neurons = int(self.outputLayer_nodes.text())
        self.outputLayer_nodes.setText('{}'.format(str(number_of_neurons + 1)))

    def minus_neuron_to_output_layer(self):
        number_of_neurons = int(self.outputLayer_nodes.text())
        if number_of_neurons - 1 >= 1:
            self.outputLayer_nodes.setText('{}'.format(str(number_of_neurons - 1)))
        else:
            self.outputLayer_nodes.setText('1')


    ## Lower theme there     
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "","CSV Files (*.csv);;All Files (*.*)", options=options)
        if fileName:
            if self.file_link:
                self.file_link = []
                self.file_link.append(fileName)
            else:
                self.file_link.append(fileName)

    def test_size_Change(self):
        values = self.slider_ts.value()
        str_values = str(values)
        self.ts_size.setText('{}%'.format(str_values))
        return values

    def number_of_neurons_in_layer(self):
        neurons_of_input_layer = int(self.inputLayer_nodes.text())
        neurons_of_output_layer = int(self.outputLayer_nodes.text())
        neurons_of_hidden_layer_1 = self.hidden1_nodes.text()
        neurons_of_hidden_layer_2 = self.hidden2_nodes.text()
        neurons_of_hidden_layer_3 = self.hidden3_nodes.text()
        neurons_of_hidden_layer_4 = self.hidden4_nodes.text()
        neurons_of_hidden_layer_5 = self.hidden5_nodes.text()
        neurons_of_hidden_layer_6 = self.hidden6_nodes.text()
        neurons_of_hidden_layer_7 = self.hidden7_nodes.text()
        neurons_of_hidden_layer_8 = self.hidden8_nodes.text()

        if neurons_of_hidden_layer_8:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), int(neurons_of_hidden_layer_4), int(neurons_of_hidden_layer_5), int(neurons_of_hidden_layer_6), int(neurons_of_hidden_layer_7), int(neurons_of_hidden_layer_8), neurons_of_output_layer]
        elif neurons_of_hidden_layer_7:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), int(neurons_of_hidden_layer_4), int(neurons_of_hidden_layer_5), int(neurons_of_hidden_layer_6), int(neurons_of_hidden_layer_7), neurons_of_output_layer]
        elif neurons_of_hidden_layer_6:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), int(neurons_of_hidden_layer_4), int(neurons_of_hidden_layer_5), int(neurons_of_hidden_layer_6), neurons_of_output_layer]
        elif neurons_of_hidden_layer_5:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), int(neurons_of_hidden_layer_4), int(neurons_of_hidden_layer_5), neurons_of_output_layer]
        elif neurons_of_hidden_layer_4:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), int(neurons_of_hidden_layer_4), neurons_of_output_layer]
        elif neurons_of_hidden_layer_3:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), int(neurons_of_hidden_layer_3), neurons_of_output_layer]
        elif neurons_of_hidden_layer_2:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), int(neurons_of_hidden_layer_2), neurons_of_output_layer]
        elif neurons_of_hidden_layer_1:
            return [neurons_of_input_layer, int(neurons_of_hidden_layer_1), neurons_of_output_layer]
        else:
            return [neurons_of_input_layer, neurons_of_output_layer]

    def makeNetworkArchitecture(self):
        number_of_neurons_and_layers = self.number_of_neurons_in_layer()
        network_config = DrawNN(number_of_neurons_and_layers)
        widest_layer = network_config.getWidestLayer()
        network = nn(widest_layer)
        full_config = network_config.nn_config()
        
        for l in full_config:
            network.add_layer(l)
            
        fig = self.network_architect_canvas
        plt.clf()
        network.nn_draw()
        plt.axis('scaled')
        plt.axis('off')
        self.network_architect_canvas.draw()

    def viewLossWithRealTime(self):
        while True:
            try:
                history = self.history[0]
                
                fig = self.loss_record_figure.add_subplot(111)
                fig.clear()
                fig.plot(history.history['loss'])
                fig.plot(history.history['val_loss'])
                fig.set_xlabel('Epochs')
                fig.set_ylabel('Loss')
                fig.legend(['Loss', 'Validation loss'], loc='best')
                self.log_loss_canvas.draw()
                break
            except:
                self.popupmsg('Please train your network first', 'warning')
                break

    def zoomNetworkAchitecture(self):
        number_of_neurons_and_layers = self.number_of_neurons_in_layer()
        self.PW = PopupWindow(number_of_neurons_and_layers)
        self.PW.show()

    def showTrainingLog(self):
        while True:
            try:
                history = self.history[0]

                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = [i for i in range(len(train_loss))]

                df = pd.DataFrame({'Epochs': epochs, 'Train loss': train_loss, 'Validation loss': val_loss})

                self.train_log = ShowTrainLog(df)
                self.train_log.show()
                break
            except:
                self.popupmsg('Please train your network first', 'warning')
                break

    def save_model_after_train(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save model', '', 'Keras model (*.h5)', options=options)
        if fileName:
            self.model[0].save(fileName)

    
### THIRD TAB CONFIGURATION
    def load_model_to_predict(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "","Keras model (*.h5)", options=options)
        if fileName:
            model = load_model(fileName)
            if self.model:
                self.model = []
                self.model.append(model)
            else:
                self.model.append(model)

    def load_predict_data(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "","CSV Files (*.csv);;All Files (*.*)", options=options)
        if fileName:
            if self.file_link:
                self.file_link = []
                self.file_link.append(fileName)
            else:
                self.file_link.append(fileName)

    def getPredictData(self):
        while True:
            try:
                data = pd.read_csv(self.file_link[0]).values
                X_test = data[:, 1:-1]
                y_test = data[:, -1]
                depth = data[:, 0]
                return X_test, y_test, depth
            except:
                break

    def model_predict(self):
        while True:
            try:
                X_test, y_test, depth = self.getPredictData()
                y_pred = self.model[0].predict(X_test)
                if not self.predicted_data:
                    self.predicted_data.append(y_pred)
                else:
                    self.predicted_data = []
                    self.predicted_data.append(y_pred)
                self.popupmsg('Predict completed', 'info')
                break
            except:
                self.popupmsg('Something wrong. Load your test data or train model before predict', 'warning')
                break

    def plotEvaluation(self):
        while True:
            try:
                X_test, y_test, depth = self.getPredictData()
                y_pred = self.predicted_data[0].flatten()
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
                y_line = slope * y_test + intercept
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                fig = self.evaluation_figure.add_subplot(111)
                fig.clear()
                fig.scatter(y_test, y_pred, color='#2C89BE', zorder=2, label='$R^2$ = {:.6f}'.format(r2))
                fig.plot(y_test, y_line, zorder=1, label='$RMSE$ = {:.6f}'.format(rmse))
                fig.set_xlabel('Original')
                fig.set_ylabel('Prediction')
                fig.legend(markerscale=0, handlelength=0, loc='best')
                self.evaluation_figure_canvas.draw()
                break
            except:
                break

    def plotPredictResult(self):
        while True:
            try:
                X_test, y_test, depth = self.getPredictData()
                y_label = self.y_label_index.text()

                fig = self.result_figure.add_subplot(111)
                fig.clear()
                fig.scatter(depth, y_test, label='Original data')
                fig.scatter(depth, self.predicted_data[0], label='Predict data')
                fig.set_xlabel('Depth')
                fig.set_ylabel('{}'.format(y_label))
                fig.legend(loc='best')
                self.result_figure_canvas.draw()
                break
            except:
                self.popupmsg('Make prediction on your data first', 'warning')
                break

    def showDetailResult(self):
        while True:
            try:
                X_test, y_test, depth = self.getPredictData()
                y_pred = self.predicted_data[0]

                df = pd.DataFrame({'Depth': depth.flatten(), 'Original': y_test.flatten(), 'Predict': y_pred.flatten()})

                self.detail_result = ShowTrainLog(df)
                self.detail_result.show()
                break
            except:
                self.popupmsg('Make prediction on your data first', 'warning')
                break

    def SavePredictResult(self):
        X_test, y_test, depth = self.getPredictData()
        y_pred = self.predicted_data[0]

        df = pd.DataFrame({'Depth': depth.flatten(), 'Original': y_test.flatten(), 'Predict': y_pred.flatten()})

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save file', '', 'CSV file (*.csv)', options=options)
        if fileName:
            df.to_csv(fileName, index=False)


### BEYONCE CONFIGURATION
    @pyqtSlot()
    def popupmsg(self, msg, type_of_msg):
        if type_of_msg == 'info':
            QMessageBox.information(self, 'Information!!!', msg, QMessageBox.Ok, QMessageBox.Ok)
        elif type_of_msg == 'warning':
            QMessageBox.warning(self, 'Warning!!!', msg, QMessageBox.Ok, QMessageBox.Ok)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = cnmApp()
    mainWin.show()
    sys.exit(app.exec_())
