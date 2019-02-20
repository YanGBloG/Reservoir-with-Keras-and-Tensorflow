import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

from GasPropertiesCalculation import *
from WaterPropertiesCalculation import *

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

class Typical(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent = parent)

        self.setFont(norm_font)

        self.gas_line = QFrame(self)
        self.gas_line.setGeometry(QRect(10, 260, 291, 16))
        self.gas_line.setFrameShape(QFrame.HLine)
        self.gas_line.setFrameShadow(QFrame.Sunken)

        self.gas_oil_line = QFrame(self)
        self.gas_oil_line.setGeometry(QRect(300, 10, 20, 541))
        self.gas_oil_line.setFrameShape(QFrame.VLine)
        self.gas_oil_line.setFrameShadow(QFrame.Sunken)
        
        self.oil_line = QFrame(self)
        self.oil_line.setGeometry(QRect(320, 260, 291, 16))
        self.oil_line.setFrameShape(QFrame.HLine)
        self.oil_line.setFrameShadow(QFrame.Sunken)

        self.oil_water_line = QFrame(self)
        self.oil_water_line.setGeometry(QRect(610, 10, 20, 541))
        self.oil_water_line.setFrameShape(QFrame.VLine)
        self.oil_water_line.setFrameShadow(QFrame.Sunken)

        self.water_line = QFrame(self)
        self.water_line.setGeometry(QRect(630, 260, 291, 16))
        self.water_line.setFrameShape(QFrame.HLine)
        self.water_line.setFrameShadow(QFrame.Sunken)

        self.water_plots_line = QFrame(self)
        self.water_plots_line.setGeometry(QRect(920, 10, 20, 541))
        self.water_plots_line.setFrameShape(QFrame.VLine)
        self.water_plots_line.setFrameShadow(QFrame.Sunken)

    ### This part calculate PROPERTIES OF GAS        
        self.gas_input_label = QLabel('Input (Gas)', self)
        self.gas_input_label.setGeometry(QRect(10, 10, 291, 21))
        self.gas_input_label.setStyleSheet('background-color: rgb(0, 85, 255); border-radius: 10px; color: white;')
        self.gas_input_label.setAlignment(Qt.AlignCenter)
        self.gas_input_label.setFont(large_font)

        self.gas_input_tabs = QTabWidget(self)
        self.gas_input_tabs.setGeometry(QRect(10, 40, 291, 221))
        self.gas_params = QWidget()
        self.gas_corr = QWidget()
        self.gas_input_tabs.addTab(self.gas_params, "Parameter")
        self.gas_input_tabs.addTab(self.gas_corr, "Correlation")

        self.T_gas = QLabel('T', self.gas_params)
        self.T_gas.setGeometry(QRect(13, 10, 21, 21))
        self.T_gas.setFont(norm_font_bold)
        self._resT = QLabel('res', self.gas_params)
        self._resT.setGeometry(QRect(26, 14, 21, 21))

        self.T_gas_res = QLineEdit(self.gas_params)
        self.T_gas_res.setGeometry(QRect(60, 10, 101, 21))

        temp_unit_item = ['F', 'C', 'K', 'R']
        self.T_gas_unit = QComboBox(self.gas_params)
        self.T_gas_unit.setGeometry(QRect(170, 10, 111, 21))
        for item in temp_unit_item:
            self.T_gas_unit.addItem(item)

        self.P_gas = QLabel('P', self.gas_params)
        self.P_gas.setGeometry(QRect(13, 36, 21, 21))
        self.P_gas.setFont(norm_font_bold)
        self._resP = QLabel('res', self.gas_params)
        self._resP.setGeometry(QRect(26, 40, 21, 21))

        self.P_gas_res = QLineEdit(self.gas_params)
        self.P_gas_res.setGeometry(QRect(60, 40, 101, 21))

        press_unit_item = ['psia', 'psig', 'bara', 'barg', 'atma', 'atmg', 'kPa a', 'kPa g', 'MPa a', 'MPa g']
        self.P_gas_unit = QComboBox(self.gas_params)
        self.P_gas_unit.setGeometry(QRect(170, 40, 111, 21))
        for item in press_unit_item:
            self.P_gas_unit.addItem(item)

        self.sg_gas = QLabel('SG', self.gas_params)
        self.sg_gas.setGeometry(QRect(10, 66, 21, 21))
        self.sg_gas.setFont(norm_font_bold)
        self._gasSG = QLabel('gas', self.gas_params)
        self._gasSG.setGeometry(QRect(32, 70, 21, 21))

        self.SG_gas = QLineEdit(self.gas_params)
        self.SG_gas.setGeometry(QRect(60, 70, 101, 21))
        
        gravity_unit_item = ['lb/ft3', 'kg/m3', 'g/cc']
        self.SG_gas_unit = QComboBox(self.gas_params)
        self.SG_gas_unit.setGeometry(QRect(170, 70, 111, 21))
        for item in gravity_unit_item:
            self.SG_gas_unit.addItem(item)

        self.CO_2 = QLabel('CO2', self.gas_params)
        self.CO_2.setGeometry(QRect(15, 100, 31, 21))

        self.co2_input = QLineEdit(self.gas_params)
        self.co2_input.setGeometry(QRect(60, 100, 101, 21))

        self.CO_2_unit = QComboBox(self.gas_params)
        self.CO_2_unit.setGeometry(QRect(170, 100, 111, 21))
        self.CO_2_unit.addItem('mole (%)')

        self.H_2S = QLabel('H2S', self.gas_params)
        self.H_2S.setGeometry(QRect(15, 130, 31, 21))

        self.h2s_input = QLineEdit(self.gas_params)
        self.h2s_input.setGeometry(QRect(60, 130, 101, 21))

        self.H_2S_unit = QComboBox(self.gas_params)
        self.H_2S_unit.setGeometry(QRect(170, 130, 111, 21))
        self.H_2S_unit.addItem('mole (%)')

        self.N_2 = QLabel('N2', self.gas_params)
        self.N_2.setGeometry(QRect(19, 160, 31, 21))

        self.n2_input = QLineEdit(self.gas_params)
        self.n2_input.setGeometry(QRect(60, 160, 101, 21))

        self.N_2_unit = QComboBox(self.gas_params)
        self.N_2_unit.setGeometry(QRect(170, 160, 111, 21))
        self.N_2_unit.addItem('mole (%)')

        self.z_corr_label = QLabel('Z', self.gas_corr)
        self.z_corr_label.setGeometry(QRect(40, 15, 21, 21))
        self.z_corr_label.setFont(norm_font_bold)

        z_corrs_item = ['Hall-Yarborough', 'Dranchuk-Abu-Kassem', 'Dranchuk-Purvis-Robinson']
        self.z_corrs = QComboBox(self.gas_corr)
        self.z_corrs.setGeometry(QRect(90, 15, 191, 21))
        for item in z_corrs_item:
            self.z_corrs.addItem(item)

        viscosity_icon = QIcon()
        viscosity_icon.addPixmap(QPixmap("../pyqt5/fig/mu.png"), QIcon.Normal, QIcon.Off)
        self.viscosity_label = QPushButton(self.gas_corr)
        self.viscosity_label.setGeometry(QRect(25, 47, 21, 21))
        self.viscosity_label.setStyleSheet('border:none')
        self.viscosity_label.setIcon(viscosity_icon)
        self.viscosity_label.setIconSize(QSize(20, 20))
        self._corr_gas = QLabel('gas', self.gas_corr)
        self._corr_gas.setGeometry(QRect(45, 52, 21, 21))

        viscosity_corrs_list = ['Lee-Gonzalez-Eakin', 'Standing-Dempsey']
        self.viscosity_corrs = QComboBox(self.gas_corr)
        self.viscosity_corrs.setGeometry(QRect(90, 50, 191, 21))
        for item in viscosity_corrs_list:
            self.viscosity_corrs.addItem(item)
        
        self.gas_output_label = QLabel('Output (Gas)', self)
        self.gas_output_label.setGeometry(QRect(10, 280, 291, 21))
        self.gas_output_label.setStyleSheet('background-color: #DEF0D8; border-radius: 10px; color: #387144;')
        self.gas_output_label.setAlignment(Qt.AlignCenter)
        self.gas_output_label.setFont(large_font)

        self.z_output_label = QLabel('Z', self)
        self.z_output_label.setGeometry(QRect(12, 310, 21, 21))
        self.z_output_label.setFont(norm_font_bold)
        self._factor = QLabel('factor', self)
        self._factor.setGeometry(QRect(26, 314, 41, 21))

        self.z_display = QLineEdit(self)
        self.z_display.setGeometry(QRect(70, 314, 101, 21))
        self.z_display.setMaxLength(6)
        self.z_display.setReadOnly(True)
        self.z_display.setStyleSheet('background-color: #E1E1E1;')
        self.z_display.setAlignment(Qt.AlignRight)
        self.z_display.setText('0')

        self.z_output_unit = QLineEdit(self)
        self.z_output_unit.setGeometry(QRect(180, 314, 111, 21))
        self.z_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.z_output_unit.setReadOnly(True)

        self.viscosity_output_label = QPushButton(self)
        self.viscosity_output_label.setGeometry(QRect(17, 340, 21, 21))
        self.viscosity_output_label.setStyleSheet('border:none')
        self.viscosity_output_label.setIcon(viscosity_icon)
        self.viscosity_output_label.setIconSize(QSize(20, 20))
        self._gas_viscosity = QLabel('gas', self)
        self._gas_viscosity.setGeometry(QRect(36, 344, 21, 21))

        self.viscosity_display = QLineEdit(self)
        self.viscosity_display.setGeometry(QRect(70, 343, 101, 21))
        self.viscosity_display.setMaxLength(6)
        self.viscosity_display.setReadOnly(True)
        self.viscosity_display.setStyleSheet('background-color: #E1E1E1;')
        self.viscosity_display.setAlignment(Qt.AlignRight)
        self.viscosity_display.setText('0')

        self.viscosity_output_unit = QLineEdit(self)
        self.viscosity_output_unit.setGeometry(QRect(180, 343, 111, 21))
        self.viscosity_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.viscosity_output_unit.setText('cP')
        self.viscosity_output_unit.setReadOnly(True)

        rho_icon = QIcon()
        rho_icon.addPixmap(QPixmap("../pyqt5/fig/rho.png"), QIcon.Normal, QIcon.Off)
        self.density_label = QPushButton(self)
        self.density_label.setGeometry(QRect(17, 366, 21, 21))
        self.density_label.setStyleSheet('border:none')
        self.density_label.setIcon(rho_icon)
        self.density_label.setIconSize(QSize(20, 20))
        self._gas_density = QLabel('gas', self)
        self._gas_density.setGeometry(QRect(36, 373, 61, 21))

        self.density_display = QLineEdit(self)
        self.density_display.setGeometry(QRect(70, 371, 101, 21))
        self.density_display.setMaxLength(6)
        self.density_display.setReadOnly(True)
        self.density_display.setStyleSheet('background-color: #E1E1E1;')
        self.density_display.setAlignment(Qt.AlignRight)
        self.density_display.setText('0')

        self.density_output_unit = QLineEdit(self)
        self.density_output_unit.setGeometry(QRect(180, 371, 111, 21))
        self.density_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.density_output_unit.setText('lbm/ft3')
        self.density_output_unit.setReadOnly(True)

        self.B_gas = QLabel('B', self)
        self.B_gas.setGeometry(QRect(22, 396, 21, 21))
        self.B_gas.setFont(norm_font_bold)
        self._gas_B = QLabel('gas', self)
        self._gas_B.setGeometry(QRect(36, 401, 41, 21))

        self.B_display = QLineEdit(self)
        self.B_display.setGeometry(QRect(70, 400, 101, 21))
        self.B_display.setMaxLength(8)
        self.B_display.setReadOnly(True)
        self.B_display.setStyleSheet('background-color: #E1E1E1;')
        self.B_display.setAlignment(Qt.AlignRight)
        self.B_display.setText('0')

        self.B_output_unit = QLineEdit(self)
        self.B_output_unit.setGeometry(QRect(180, 400, 111, 21))
        self.B_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.B_output_unit.setText('scf/scf')
        self.B_output_unit.setReadOnly(True)

        self.C_gas = QLabel('C', self)
        self.C_gas.setGeometry(QRect(22, 426, 21, 21))
        self.C_gas.setFont(norm_font_bold)
        self._gas_C = QLabel('gas', self)
        self._gas_C.setGeometry(QRect(36, 430, 41, 21))

        self.C_display = QLineEdit(self)
        self.C_display.setGeometry(QRect(70, 430, 101, 21))
        self.C_display.setMaxLength(8)
        self.C_display.setReadOnly(True)
        self.C_display.setStyleSheet('background-color: #E1E1E1;')
        self.C_display.setAlignment(Qt.AlignRight)
        self.C_display.setText('0')

        self.C_output_unit = QLineEdit(self)
        self.C_output_unit.setGeometry(QRect(180, 430, 111, 21))
        self.C_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.C_output_unit.setText('1/psia')
        self.C_output_unit.setReadOnly(True)

        calculate_icon = QIcon()
        calculate_icon.addPixmap(QPixmap("../pyqt5/fig/calculate.png"), QIcon.Normal, QIcon.Off)
        
        self.gas_calculate_button = QPushButton('Calculate', self)
        self.gas_calculate_button.setGeometry(QRect(190, 520, 101, 31))
        self.gas_calculate_button.setIcon(calculate_icon)
        self.gas_calculate_button.setIconSize(QSize(30, 30))
        self.gas_calculate_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.gas_calculate_button.clicked.connect(self.printGasResult)
        self.gas_calculate_button.clicked.connect(self.printWaterResult)

    ### This part calculate PROPERTIES OF OIL
        self.oil_input_label = QLabel('Input (Oil)', self)
        self.oil_input_label.setGeometry(QRect(320, 10, 291, 21))
        self.oil_input_label.setStyleSheet('background-color: rgb(0, 85, 255); border-radius: 10px; color: white;')
        self.oil_input_label.setAlignment(Qt.AlignCenter)
        self.oil_input_label.setFont(large_font)

        self.oil_input_tabs = QTabWidget(self)
        self.oil_input_tabs.setGeometry(QRect(320, 40, 291, 221))
        self.oil_params = QWidget()
        self.oil_corr = QWidget()

        self.oil_input_tabs.addTab(self.oil_params, "Parameter")
        self.oil_input_tabs.addTab(self.oil_corr, "Correlation")

        self.T_oil = QLabel('T', self.oil_params)
        self.T_oil.setGeometry(QRect(13, 10, 21, 21))
        self.T_oil.setFont(norm_font_bold)
        self._resT_oil = QLabel('res', self.oil_params)
        self._resT_oil.setGeometry(QRect(26, 14, 21, 21))

        self.T_oil_res = QLineEdit(self.oil_params)
        self.T_oil_res.setGeometry(QRect(60, 10, 101, 21))

        temp_unit_item = ['F', 'C', 'K', 'R']
        self.T_oil_unit = QComboBox(self.oil_params)
        self.T_oil_unit.setGeometry(QRect(170, 10, 111, 21))
        for item in temp_unit_item:
            self.T_oil_unit.addItem(item)

        self.P_oil = QLabel('P', self.oil_params)
        self.P_oil.setGeometry(QRect(13, 36, 21, 21))
        self.P_oil.setFont(norm_font_bold)
        self._resP_oil = QLabel('res', self.oil_params)
        self._resP_oil.setGeometry(QRect(26, 40, 21, 21))

        self.P_oil_res = QLineEdit(self.oil_params)
        self.P_oil_res.setGeometry(QRect(60, 40, 101, 21))

        press_unit_item = ['psia', 'psig', 'bara', 'barg', 'atma', 'atmg', 'kPa a', 'kPa g', 'MPa a', 'MPa g']
        self.P_oil_unit = QComboBox(self.oil_params)
        self.P_oil_unit.setGeometry(QRect(170, 40, 111, 21))
        for item in press_unit_item:
            self.P_oil_unit.addItem(item)

        self.sg_oil_label = QLabel('SG', self.oil_params)
        self.sg_oil_label.setGeometry(QRect(10, 66, 21, 21))
        self.sg_oil_label.setFont(norm_font_bold)
        self._oilSG = QLabel('oil', self.oil_params)
        self._oilSG.setGeometry(QRect(32, 70, 21, 21))

        self.SG_oil = QLineEdit(self.oil_params)
        self.SG_oil.setGeometry(QRect(60, 70, 101, 21))

        self.SG_oil_unit = QComboBox(self.oil_params)
        self.SG_oil_unit.setGeometry(QRect(170, 70, 111, 21))
        self.SG_oil_unit.addItem('API')

        self.sg_oil_gas_label = QLabel('SG', self.oil_params)
        self.sg_oil_gas_label.setGeometry(QRect(10, 97, 21, 21))
        self.sg_oil_gas_label.setFont(norm_font_bold)
        self._oilgasSG = QLabel('gas', self.oil_params)
        self._oilgasSG.setGeometry(QRect(32, 100, 21, 21))

        self.SG_oil_gas = QLineEdit(self.oil_params)
        self.SG_oil_gas.setGeometry(QRect(60, 100, 101, 21))

        self.SG_oil_gas_unit = QComboBox(self.oil_params)
        self.SG_oil_gas_unit.setGeometry(QRect(170, 100, 111, 21))
        for item in gravity_unit_item:
            self.SG_oil_gas_unit.addItem(item)

        self.oil_type_label = QLabel('Oil type:', self.oil_params)
        self.oil_type_label.setGeometry(QRect(5, 125, 51, 21))

        self.dead_oil = QRadioButton('Dead oil', self.oil_params)
        self.dead_oil.setGeometry(QRect(68, 127, 82, 21))

        self.saturated_oil = QRadioButton('Saturated oil', self.oil_params)
        self.saturated_oil.setGeometry(QRect(68, 147, 111, 21))

        self.under_saturated_oil = QRadioButton('Under-Saturated oil', self.oil_params)
        self.under_saturated_oil.setGeometry(QRect(68, 167, 141, 21))

        self.Pb_corrs_label = QLabel('P', self.oil_corr)
        self.Pb_corrs_label.setGeometry(QRect(30, 15, 21, 21))
        self.Pb_corrs_label.setFont(norm_font_bold)
        self._b = QLabel('b', self.oil_corr)
        self._b.setGeometry(QRect(44, 19, 21, 21))

        pb_correlation_list = ['Standing', 'Glaso', 'Marhoun', 'Petrosky-Farshad']
        self.pb_corrs = QComboBox(self.oil_corr)
        self.pb_corrs.setGeometry(QRect(90, 15, 191, 21))
        for item in pb_correlation_list:
            self.pb_corrs.addItem(item)

        self.Rsb_corrs_label = QLabel('R', self.oil_corr)
        self.Rsb_corrs_label.setGeometry(QRect(30, 47, 21, 21))
        self.Rsb_corrs_label.setFont(norm_font_bold)
        self._sb = QLabel('sb', self.oil_corr)
        self._sb.setGeometry(QRect(45, 52, 21, 21))

        Rsb_correlation_list = ['Standing', 'Glaso', 'Marhoun', 'Petrosky-Farshad']
        self.rsb_corrs = QComboBox(self.oil_corr)
        self.rsb_corrs.setGeometry(QRect(90, 50, 191, 21))
        for item in Rsb_correlation_list:
            self.rsb_corrs.addItem(item)

        self.oil_viscosity_corrs_label = QPushButton(self.oil_corr)
        self.oil_viscosity_corrs_label.setGeometry(QRect(25, 82, 21, 21))
        self.oil_viscosity_corrs_label.setIcon(viscosity_icon)
        self.oil_viscosity_corrs_label.setStyleSheet('border:none')
        self.oil_viscosity_corrs_label.setIconSize(QSize(20, 20))
        self._oil_vis_corr = QLabel('oil', self.oil_corr)
        self._oil_vis_corr.setGeometry(QRect(45, 87, 21, 21))

        self.oil_viscosity_corrs = QComboBox(self.oil_corr)
        self.oil_viscosity_corrs.setGeometry(QRect(90, 85, 191, 21))

        self.B_oil_corrs_label = QLabel('B', self.oil_corr)
        self.B_oil_corrs_label.setGeometry(QRect(30, 116, 21, 21))
        self.B_oil_corrs_label.setFont(norm_font_bold)
        self._b_oil = QLabel('oil', self.oil_corr)
        self._b_oil.setGeometry(QRect(45, 123, 21, 21))

        Bo_correlation_list = ['Standing', 'Glaso', 'Marhoun', 'Petrosky-Farshad']
        self.b_oil_corrs = QComboBox(self.oil_corr)
        self.b_oil_corrs.setGeometry(QRect(90, 120, 191, 21))
        for item in Bo_correlation_list:
            self.b_oil_corrs.addItem(item)

        self.C_oil_corrs_label = QLabel('C', self.oil_corr)
        self.C_oil_corrs_label.setGeometry(QRect(30, 151, 21, 21))
        self.C_oil_corrs_label.setFont(norm_font_bold)
        self._c_oil = QLabel('oil', self.oil_corr)
        self._c_oil.setGeometry(QRect(45, 157, 21, 21))

        self.c_oil_corrs = QComboBox(self.oil_corr)
        self.c_oil_corrs.setGeometry(QRect(90, 155, 191, 21))
        
        self.oil_output_label = QLabel('Output (Oil)', self)
        self.oil_output_label.setGeometry(QRect(320, 280, 291, 21))
        self.oil_output_label.setStyleSheet('background-color: #DEF0D8; border-radius: 10px; color: #387144;')
        self.oil_output_label.setAlignment(Qt.AlignCenter)
        self.oil_output_label.setFont(large_font)

        self.bubblepoint_pressure_output_label = QLabel('P', self)
        self.bubblepoint_pressure_output_label.setGeometry(QRect(335, 310, 21, 21))
        self.bubblepoint_pressure_output_label.setFont(norm_font_bold)
        self._bubblepoint = QLabel('b', self)
        self._bubblepoint.setGeometry(QRect(348, 314, 41, 21))

        self.bubblepoint_pressure_display = QLineEdit(self)
        self.bubblepoint_pressure_display.setGeometry(QRect(380, 314, 101, 21))
        self.bubblepoint_pressure_display.setMaxLength(6)
        self.bubblepoint_pressure_display.setReadOnly(True)
        self.bubblepoint_pressure_display.setStyleSheet('background-color: #E1E1E1;')
        self.bubblepoint_pressure_display.setAlignment(Qt.AlignRight)
        self.bubblepoint_pressure_display.setText('0')

        self.bubblepoint_pressure_output_unit = QLineEdit(self)
        self.bubblepoint_pressure_output_unit.setGeometry(QRect(490, 314, 111, 21))
        self.bubblepoint_pressure_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.bubblepoint_pressure_output_unit.setText('psia')
        self.bubblepoint_pressure_output_unit.setReadOnly(True)

        self.solubility_output_label = QLabel('R', self)
        self.solubility_output_label.setGeometry(QRect(335, 340, 21, 21))
        self.solubility_output_label.setFont(norm_font_bold)
        self._solubility = QLabel('sb', self)
        self._solubility.setGeometry(QRect(349, 344, 41, 21))

        self.solubility_display = QLineEdit(self)
        self.solubility_display.setGeometry(QRect(380, 343, 101, 21))
        self.solubility_display.setMaxLength(6)
        self.solubility_display.setReadOnly(True)
        self.solubility_display.setStyleSheet('background-color: #E1E1E1;')
        self.solubility_display.setAlignment(Qt.AlignRight)
        self.solubility_display.setText('0')

        self.solubility_output_unit = QLineEdit(self)
        self.solubility_output_unit.setGeometry(QRect(490, 343, 111, 21))
        self.solubility_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.solubility_output_unit.setReadOnly(True)

        self.oil_viscosity_output_label = QPushButton(self)
        self.oil_viscosity_output_label.setGeometry(QRect(330, 370, 21, 21))
        self.oil_viscosity_output_label.setStyleSheet('border:none')
        self.oil_viscosity_output_label.setIcon(viscosity_icon)
        self.oil_viscosity_output_label.setIconSize(QSize(20, 20))
        self._oil_viscosity = QLabel('oil', self)
        self._oil_viscosity.setGeometry(QRect(350, 374, 21, 21))

        self.oil_viscosity_display = QLineEdit(self)
        self.oil_viscosity_display.setGeometry(QRect(380, 371, 101, 21))
        self.oil_viscosity_display.setMaxLength(6)
        self.oil_viscosity_display.setReadOnly(True)
        self.oil_viscosity_display.setStyleSheet('background-color: #E1E1E1;')
        self.oil_viscosity_display.setAlignment(Qt.AlignRight)
        self.oil_viscosity_display.setText('0')

        self.oil_viscosity_output_unit = QLineEdit(self)
        self.oil_viscosity_output_unit.setGeometry(QRect(490, 371, 111, 21))
        self.oil_viscosity_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.oil_viscosity_output_unit.setText('cP')
        self.oil_viscosity_output_unit.setReadOnly(True)

        self.oil_density_label = QPushButton(self)
        self.oil_density_label.setGeometry(QRect(330, 400, 21, 21))
        self.oil_density_label.setStyleSheet('border:none')
        self.oil_density_label.setIcon(rho_icon)
        self.oil_density_label.setIconSize(QSize(20, 20))
        self._oil_density = QLabel('oil', self)
        self._oil_density.setGeometry(QRect(350, 404, 61, 21))

        self.oil_density_display = QLineEdit(self)
        self.oil_density_display.setGeometry(QRect(380, 400, 101, 21))
        self.oil_density_display.setMaxLength(6)
        self.oil_density_display.setReadOnly(True)
        self.oil_density_display.setStyleSheet('background-color: #E1E1E1;')
        self.oil_density_display.setAlignment(Qt.AlignRight)
        self.oil_density_display.setText('0')

        self.oil_density_output_unit = QLineEdit(self)
        self.oil_density_output_unit.setGeometry(QRect(490, 400, 111, 21))
        self.oil_density_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.oil_density_output_unit.setText('lbm/ft3')
        self.oil_density_output_unit.setReadOnly(True)

        sigma_icon = QIcon()
        sigma_icon.addPixmap(QPixmap("../pyqt5/fig/sigma.png"), QIcon.Normal, QIcon.Off)
        self.oil_surface_tension_label = QPushButton(self)
        self.oil_surface_tension_label.setGeometry(QRect(330, 426, 21, 21))
        self.oil_surface_tension_label.setStyleSheet('border:none')
        self.oil_surface_tension_label.setIcon(sigma_icon)
        self.oil_surface_tension_label.setIconSize(QSize(25, 25))
        self._oil_surface_tension = QLabel('oil', self)
        self._oil_surface_tension.setGeometry(QRect(350, 433, 21, 21))

        self.oil_surface_tension_display = QLineEdit(self)
        self.oil_surface_tension_display.setGeometry(QRect(380, 430, 101, 21))
        self.oil_surface_tension_display.setMaxLength(6)
        self.oil_surface_tension_display.setReadOnly(True)
        self.oil_surface_tension_display.setStyleSheet('background-color: #E1E1E1;')
        self.oil_surface_tension_display.setAlignment(Qt.AlignRight)
        self.oil_surface_tension_display.setText('0')

        self.oil_surface_tension_unit = QLineEdit(self)
        self.oil_surface_tension_unit.setGeometry(QRect(490, 430, 111, 21))
        self.oil_surface_tension_unit.setStyleSheet('background-color: #E1E1E1;')
        self.oil_surface_tension_unit.setText('dyn/cm')
        self.oil_surface_tension_unit.setReadOnly(True)

        self.B_oil = QLabel('B', self)
        self.B_oil.setGeometry(QRect(335, 456, 21, 21))
        self.B_oil.setFont(norm_font_bold)
        self._oil_B = QLabel('oil', self)
        self._oil_B.setGeometry(QRect(350, 462, 41, 21))

        self.B_oil_display = QLineEdit(self)
        self.B_oil_display.setGeometry(QRect(380, 460, 101, 21))
        self.B_oil_display.setMaxLength(8)
        self.B_oil_display.setReadOnly(True)
        self.B_oil_display.setStyleSheet('background-color: #E1E1E1;')
        self.B_oil_display.setAlignment(Qt.AlignRight)
        self.B_oil_display.setText('0')

        self.B_oil_output_unit = QLineEdit(self)
        self.B_oil_output_unit.setGeometry(QRect(490, 460, 111, 21))
        self.B_oil_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.B_oil_output_unit.setText('ft3/scf')
        self.B_oil_output_unit.setReadOnly(True)

        self.C_oil = QLabel('C', self)
        self.C_oil.setGeometry(QRect(335, 487, 21, 21))
        self.C_oil.setFont(norm_font_bold)
        self._oil_C = QLabel('oil', self)
        self._oil_C.setGeometry(QRect(350, 492, 21, 21))

        self.C_oil_display = QLineEdit(self)
        self.C_oil_display.setGeometry(QRect(380, 490, 101, 21))
        self.C_oil_display.setMaxLength(8)
        self.C_oil_display.setReadOnly(True)
        self.C_oil_display.setStyleSheet('background-color: #E1E1E1;')
        self.C_oil_display.setAlignment(Qt.AlignRight)
        self.C_oil_display.setText('0')

        self.C_oil_output_unit = QLineEdit(self)
        self.C_oil_output_unit.setGeometry(QRect(490, 490, 111, 21))
        self.C_oil_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.C_oil_output_unit.setText('1/psia')
        self.C_oil_output_unit.setReadOnly(True)

        self.oil_calculate_button = QPushButton('Calculate', self)
        self.oil_calculate_button.setGeometry(QRect(500, 520, 101, 31))
        self.oil_calculate_button.setIcon(calculate_icon)
        self.oil_calculate_button.setIconSize(QSize(30, 30))
        self.oil_calculate_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.oil_calculate_button.clicked.connect(self.printOilResult)

    ### This part calculate PROPERTIES OF WATER
        self.water_input_label = QLabel('Input (Water)', self)
        self.water_input_label.setGeometry(QRect(630, 10, 291, 21))
        self.water_input_label.setStyleSheet('background-color: rgb(0, 85, 255); border-radius: 10px; color: white;')
        self.water_input_label.setAlignment(Qt.AlignCenter)
        self.water_input_label.setFont(large_font)

        self.water_input_tabs = QTabWidget(self)
        self.water_input_tabs.setGeometry(QRect(630, 40, 291, 221))
        self.water_params = QWidget()
        self.water_corr = QWidget()

        self.water_input_tabs.addTab(self.water_params, "Parameter")
        self.water_input_tabs.addTab(self.water_corr, "Correlation")

        self.sg_water = QLabel('SG', self.water_params)
        self.sg_water.setGeometry(QRect(2, 8, 21, 21))
        self.sg_water.setFont(norm_font_bold)
        self._waterSG = QLabel('water', self.water_params)
        self._waterSG.setGeometry(QRect(23, 12, 41, 21))

        self.SG_water = QLineEdit(self.water_params)
        self.SG_water.setGeometry(QRect(60, 10, 101, 21))
        
        self.SG_water_unit = QComboBox(self.water_params)
        self.SG_water_unit.setGeometry(QRect(170, 10, 111, 21))
        for item in gravity_unit_item:
            self.SG_water_unit.addItem(item)

        self.NaCl_content_label = QLabel('Salinity', self.water_params)
        self.NaCl_content_label.setGeometry(QRect(8, 43, 45, 20))

        self.NaCl_content = QLineEdit(self.water_params)
        self.NaCl_content.setGeometry(QRect(60, 40, 101, 21))
        
        self.NaCl_content_unit = QComboBox(self.water_params)
        self.NaCl_content_unit.setGeometry(QRect(170, 40, 111, 21))
        self.NaCl_content_unit.addItem('wt %')

        self.gas_type = QLabel('Gas type:', self.water_params)
        self.gas_type.setGeometry(QRect(5, 70, 55, 16))

        self.free_gas = QRadioButton('Gas-Free', self.water_params)
        self.free_gas.setGeometry(QRect(80, 70, 82, 17))

        self.saturated_gas = QRadioButton('Gas-Saturated', self.water_params)
        self.saturated_gas.setGeometry(QRect(80, 100, 111, 17))

        self.water_viscosity_correlation_label = QPushButton(self.water_corr)
        self.water_viscosity_correlation_label.setGeometry(QRect(17, 15, 21, 21))
        self.water_viscosity_correlation_label.setStyleSheet('border:none')
        self.water_viscosity_correlation_label.setIcon(viscosity_icon)
        self.water_viscosity_correlation_label.setIconSize(QSize(20, 20))
        self.__water_corr_label = QLabel('water', self.water_corr)
        self.__water_corr_label.setGeometry(QRect(35, 20, 41, 21))

        self.water_viscosity_corrs = QComboBox(self.water_corr)
        self.water_viscosity_corrs.setGeometry(QRect(90, 15, 191, 21))

        water_viscosity_corrs_lst = ['Beggs-Brill', 'McCain']
        for item in water_viscosity_corrs_lst:
            self.water_viscosity_corrs.addItem(item)
        
        self.water_output_label = QLabel('Output (Water)', self)
        self.water_output_label.setGeometry(QRect(630, 280, 291, 21))
        self.water_output_label.setStyleSheet('background-color: #DEF0D8; border-radius: 10px; color: #387144;')
        self.water_output_label.setAlignment(Qt.AlignCenter)
        self.water_output_label.setFont(large_font)

        self.Rsb_water_label = QLabel('R', self)
        self.Rsb_water_label.setGeometry(QRect(650, 311, 21, 21))
        self.Rsb_water_label.setFont(norm_font_bold)
        self._sb_water = QLabel('sb', self)
        self._sb_water.setGeometry(QRect(664, 316, 21, 21))

        self.Rsb_water_display = QLineEdit(self)
        self.Rsb_water_display.setGeometry(QRect(700, 314, 101, 21))
        self.Rsb_water_display.setMaxLength(6)
        self.Rsb_water_display.setReadOnly(True)
        self.Rsb_water_display.setStyleSheet('background-color: #E1E1E1;')
        self.Rsb_water_display.setAlignment(Qt.AlignRight)
        self.Rsb_water_display.setText('0')

        self.Rsb_water_unit = QLineEdit(self)
        self.Rsb_water_unit.setGeometry(QRect(810, 314, 111, 21))
        self.Rsb_water_unit.setStyleSheet('background-color: #E1E1E1;')
        self.Rsb_water_unit.setReadOnly(True)

        self.water_viscosity_output_label = QPushButton(self)
        self.water_viscosity_output_label.setGeometry(QRect(635, 340, 21, 21))
        self.water_viscosity_output_label.setStyleSheet('border:none')
        self.water_viscosity_output_label.setIcon(viscosity_icon)
        self.water_viscosity_output_label.setIconSize(QSize(20, 20))
        self._water_viscosity = QLabel('water', self)
        self._water_viscosity.setGeometry(QRect(655, 346, 41, 21))

        self.water_viscosity_display = QLineEdit(self)
        self.water_viscosity_display.setGeometry(QRect(700, 343, 101, 21))
        self.water_viscosity_display.setMaxLength(6)
        self.water_viscosity_display.setReadOnly(True)
        self.water_viscosity_display.setStyleSheet('background-color: #E1E1E1;')
        self.water_viscosity_display.setAlignment(Qt.AlignRight)
        self.water_viscosity_display.setText('0')

        self.water_viscosity_output_unit = QLineEdit(self)
        self.water_viscosity_output_unit.setGeometry(QRect(810, 343, 111, 21))
        self.water_viscosity_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.water_viscosity_output_unit.setText('cP')
        self.water_viscosity_output_unit.setReadOnly(True)

        self.water_density_label = QPushButton(self)
        self.water_density_label.setGeometry(QRect(635, 370, 21, 21))
        self.water_density_label.setStyleSheet('border:none')
        self.water_density_label.setIcon(rho_icon)
        self.water_density_label.setIconSize(QSize(20, 20))
        self._water_density = QLabel('water', self)
        self._water_density.setGeometry(QRect(655, 374, 41, 21))

        self.water_density_display = QLineEdit(self)
        self.water_density_display.setGeometry(QRect(700, 371, 101, 21))
        self.water_density_display.setMaxLength(6)
        self.water_density_display.setReadOnly(True)
        self.water_density_display.setStyleSheet('background-color: #E1E1E1;')
        self.water_density_display.setAlignment(Qt.AlignRight)
        self.water_density_display.setText('0')

        self.water_density_output_unit = QLineEdit(self)
        self.water_density_output_unit.setGeometry(QRect(810, 371, 111, 21))
        self.water_density_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.water_density_output_unit.setText('lbm/ft3')
        self.water_density_output_unit.setReadOnly(True)

        self.B_water = QLabel('B', self)
        self.B_water.setGeometry(QRect(640, 400, 21, 21))
        self.B_water.setFont(norm_font_bold)
        self._water_B = QLabel('water', self)
        self._water_B.setGeometry(QRect(655, 404, 41, 21))

        self.B_water_display = QLineEdit(self)
        self.B_water_display.setGeometry(QRect(700, 400, 101, 21))
        self.B_water_display.setMaxLength(8)
        self.B_water_display.setReadOnly(True)
        self.B_water_display.setStyleSheet('background-color: #E1E1E1;')
        self.B_water_display.setAlignment(Qt.AlignRight)
        self.B_water_display.setText('0')

        self.B_water_output_unit = QLineEdit(self)
        self.B_water_output_unit.setGeometry(QRect(810, 400, 111, 21))
        self.B_water_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.B_water_output_unit.setText('ft3/scf')
        self.B_water_output_unit.setReadOnly(True)

        self.C_water = QLabel('C', self)
        self.C_water.setGeometry(QRect(640, 426, 21, 21))
        self.C_water.setFont(norm_font_bold)
        self._water_C = QLabel('water', self)
        self._water_C.setGeometry(QRect(655, 433, 41, 21))

        self.C_water_display = QLineEdit(self)
        self.C_water_display.setGeometry(QRect(700, 430, 101, 21))
        self.C_water_display.setReadOnly(True)
        self.C_water_display.setStyleSheet('background-color: #E1E1E1;')
        self.C_water_display.setAlignment(Qt.AlignRight)
        self.C_water_display.setText('0')

        self.C_water_output_unit = QLineEdit(self)
        self.C_water_output_unit.setGeometry(QRect(810, 430, 111, 21))
        self.C_water_output_unit.setStyleSheet('background-color: #E1E1E1;')
        self.C_water_output_unit.setText('1/psia')
        self.C_water_output_unit.setReadOnly(True)

    ### This part PLOTS FIGURE FOR ALL OIL AND GAS PROPERTIES
        self.plots_label = QLabel('Plots', self)
        self.plots_label.setGeometry(QRect(940, 10, 321, 21))
        self.plots_label.setStyleSheet('background-color: #DEF0D8; border-radius: 10px; color: #387144;')
        self.plots_label.setFont(large_font)
        self.plots_label.setAlignment(Qt.AlignCenter)

    def getGravity(self, type_of_hc):
        unit = str(self.SG_gas_unit.currentText())
        try:
            gamma_gas_init = float(self.SG_gas.text())
        except ValueError:
            gamma_gas_init = 1
        try:
            gamma_water_init = float(self.SG_water.text())
        except ValueError:
            gamma_water_init = 1

        if unit == 'kg/m3':
            gamma_gas = gamma_gas_init / 16.018
            gamma_water = gamma_water_init / 16.018
        elif unit == 'g/cc':
            gamma_gas = gamma_gas_init * 62.428
            gamma_water = gamma_water_init * 62.428
        elif unit == 'lb/ft3':
            gamma_gas = gamma_gas_init
            gamma_water = gamma_water_init
        elif gamma_gas_init == 1:
            gamma_gas = 1
            gamma_water = 1

        try:
            gamma_oil_init = float(self.SG_oil.text())
        except ValueError:
            gamma_oil_init = 1
        if gamma_oil_init != 1:
            gamma_oil = 141.5 / (gamma_oil_init + 131.5)
        elif gamma_oil_init == 1:
            gamma_oil = 1

        if type_of_hc == 'oil':
            return gamma_gas, gamma_oil, gamma_water
        elif type_of_hc == 'gas':
            return gamma_gas, gamma_water

    def getTemperature(self, type_of_hc):
        unit = str(self.T_gas_unit.currentText())
        if type_of_hc == 'gas':
            try:
                T_init = float(self.T_gas_res.text())
            except ValueError:
                T_init = 1
        elif type_of_hc == 'oil':
            try:
                T_init = float(self.T_oil_res.text())
            except ValueError:
                T_init = 1

        if unit == 'C':
            T = T_init + 9/5 + 491.67
        elif unit == 'F':
            T = T_init + 460
        elif unit == 'K':
            T = T_init*1.8
        elif unit == 'R':
            T = T_init
        elif T_init == 1:
            T = 1
        return T

    def getPressure(self, type_of_hc):
        unit = str(self.P_gas_unit.currentText())
        if type_of_hc == 'gas':
            try:
                P_init = float(self.P_gas_res.text())
            except ValueError:
                P_init = 1
        elif type_of_hc == 'oil':
            try:
                P_init = float(self.P_oil_res.text())
            except ValueError:
                P_init = 1

        if unit == 'psia':
            P = P_init
        elif unit == 'psig':
            P = P_init + 14.7
        elif unit == 'bara' or unit == 'barg':
            P = P_init * 14.504
        elif unit == 'atma' or unit == 'atmg':
            P = P_init * 14.7
        elif unit == 'kPa a' or unit == 'kPa g':
            P = P_init / 6.9
        elif unit == 'MPa a' or unit == 'MPa g':
            P = P_init * 145.038
        elif P_init == 1:
            P = 1
        return P

    def nonHCcomponents(self):
        try:
            co2 = float(self.co2_input.text())
        except ValueError:
            co2 = 0
        try:
            h2s = float(self.h2s_input.text())
        except ValueError:
            h2s = 0
        try:
            n2 = float(self.n2_input.text())
        except ValueError:
            n2 = 0
        return co2/100, h2s/100, n2/100

    def exeGasProperties(self):
        P = self.getPressure('gas')
        T = self.getTemperature('gas')
        sg_gas, sg_water = self.getGravity('gas')
        co2, h2s, n2 = self.nonHCcomponents()
        z_corr = str(self.z_corrs.currentText())
        vis_corr = str(self.viscosity_corrs.currentText())

        pseudo_z = zFactor(P, T, sg_gas, co2, h2s, n2, z_corr)
        z = pseudo_z.calculate_z()

        pseudo_viscosity = gasViscosity(P, T, sg_gas, z, co2, h2s, n2, vis_corr)
        viscosity = pseudo_viscosity.calculate_viscosity()

        density_of_gas = pseudo_viscosity.gasDensity()

        pseudo_Bg = gasFVF(P, T, sg_gas, co2, h2s, n2, z_corr)
        Bg = pseudo_Bg.Bg()

        pseudo_Cp = gasCompressibility(P, T, sg_gas, co2, h2s, n2, z_corr)
        Cp = pseudo_Cp.Cp()

        return z, viscosity, density_of_gas, Bg, Cp

    def exeOilProperties(self):
        pass

    def exeWaterGasProperties(self):
        P = self.getPressure('gas')
        T = self.getTemperature('gas')
        sg_gas, sg_water = self.getGravity('gas')

        water_viscosity_corr = str(self.water_viscosity_corrs.currentText())
        try:
            nacl = float(self.NaCl_content.text())
        except ValueError:
            nacl = 0

        if self.saturated_gas.isChecked():
            gas_type = 'Gas-Saturated'
        else:
            gas_type = 'Gas-Free'

        density_pr = WaterDensity(P, T, sg_water, nacl, gas_type)
        water_density = density_pr.reservoir_condition()

        Bwater_pr = WaterFVF(P, T, sg_water, gas_type)
        Bw = Bwater_pr.Bwater()

        viscosity_pr = WaterViscosity(P, T, sg_water, nacl, water_viscosity_corr)
        water_viscosity = viscosity_pr.waterViscosity()

        Rsb_water_pr = GasSolubilityinWater(P, T, sg_water)
        Rsb = Rsb_water_pr.Rsw()

        compressibility_pr = WaterIsothermalCompressibity(P, T, sg_water)
        Cw = compressibility_pr.Cw()

        return water_density, Bw, water_viscosity, Rsb, Cw

    def exeWaterOilProperties(self):
        pass

    def printGasResult(self):
        z, viscosity, density_of_gas, Bg, Cp = self.exeGasProperties()
        if all([self.getGravity('gas') != 1, self.getTemperature('gas') != 1, self.getPressure('gas') != 1]):
            self.z_display.setText(str(z))
            self.viscosity_display.setText(str(viscosity))
            self.density_display.setText(str(density_of_gas))
            self.B_display.setText(str(Bg))
            self.C_display.setText(str(Cp))

    def printOilResult(self):
        pass

    def printWaterResult(self):
        water_density, Bw, water_viscosity, Rsb, Cw = self.exeWaterGasProperties()
        setCw = str(Cw)[:6] + str(Cw)[-4:]
        self.Rsb_water_display.setText(str(Rsb))
        self.water_viscosity_display.setText(str(water_viscosity))
        self.water_density_display.setText(str(water_density))
        self.B_water_display.setText(str(Bw))
        self.C_water_display.setText(setCw)
        