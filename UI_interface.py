# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QComboBox
import sys
from pyqtgraph import GraphicsLayoutWidget
import globalvar as gl
import pyqtgraph as pg
import os
import matplotlib
import matplotlib.cm
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

gl._init()
gl.set_value('usr_gesture', False)

radarRFconfigpathfile = 'config'
modelpathfile = 'save_model'


def _list_cmap_names():
    try:
        return list(matplotlib.colormaps)
    except Exception:
        return list(getattr(matplotlib.cm, "cmap_d", {}).keys())


class ClickedComboBox(QComboBox):
    arrowClicked = pyqtSignal()

    def showPopup(self):
        super(ClickedComboBox, self).showPopup()
        self.arrowClicked.emit()


class Qt_pet(QWidget):
    def __init__(self, MainWindow):
        super(Qt_pet, self).__init__()
        self.MainWindow = MainWindow
        self.dis_file = "gesture_icons/"
        self.windowinit()
        self.pos_first = self.pos()

    def img_update(self, img_path):
        self.img_path = img_path
        self.qpixmap = QPixmap(self.img_path).scaled(256, 256)
        self.lab.setPixmap(self.qpixmap)

    def windowinit(self):
        self.x = 800
        self.y = 600
        self.setGeometry(self.x, self.y, 256, 256)
        self.img_path = 'gesture_icons/7.jpg'
        self.lab = QLabel(self)
        self.qpixmap = QPixmap(self.img_path).scaled(256, 256)
        self.lab.setPixmap(self.qpixmap)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow
        )
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.pos_first = QMouseEvent.globalPos() - self.pos()
            QMouseEvent.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
        if QMouseEvent.button() == Qt.RightButton:
            self.MainWindow.show()
            self.hide()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton:
            self.move(QMouseEvent.globalPos() - self.pos_first)
            self.x, self.y = self.pos().x, self.pos().y
            QMouseEvent.accept()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', 'd')
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            MainWindow.sizePolicy().hasHeightForWidth()
        )
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        # -------- tab1: 实时界面 --------
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.splitter = QtWidgets.QSplitter(self.tab)
        self.splitter.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.splitter.sizePolicy().hasHeightForWidth()
        )
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setAutoFillBackground(False)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        # 右侧配置区
        self.groupBox_11 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(15)
        sizePolicy.setHeightForWidth(
            self.groupBox_11.sizePolicy().hasHeightForWidth()
        )
        self.groupBox_11.setSizePolicy(sizePolicy)
        self.groupBox_11.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_11.setMaximumSize(QtCore.QSize(2400, 8000))
        self.groupBox_11.setObjectName("groupBox_11")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_11)
        self.verticalLayout_4.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        # ---- Radar setup ----
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_11)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout.setObjectName("verticalLayout")

        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 3, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_3.addWidget(self.label_16, 3, 1, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.groupBox_2)
        self.label_36.setObjectName("label_36")
        self.gridLayout_3.addWidget(self.label_36, 4, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 1, 1, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.groupBox_2)
        self.label_43.setObjectName("label_43")
        self.gridLayout_3.addWidget(self.label_43, 6, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 1, 0, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.groupBox_2)
        self.label_35.setObjectName("label_35")
        self.gridLayout_3.addWidget(self.label_35, 2, 1, 1, 1)
        self.comboBox_7 = ClickedComboBox(self.groupBox_2)
        self.comboBox_7.setObjectName("comboBox_7")
        self.gridLayout_3.addWidget(self.comboBox_7, 0, 1, 1, 1)
        self.label_45 = QtWidgets.QLabel(self.groupBox_2)
        self.label_45.setAlignment(QtCore.Qt.AlignCenter)
        self.label_45.setObjectName("label_45")
        self.gridLayout_3.addWidget(self.label_45, 5, 0, 1, 2)
        self.comboBox_9 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_9.setEnabled(False)
        self.comboBox_9.setEditable(False)
        self.comboBox_9.setObjectName("comboBox_9")
        self.gridLayout_3.addWidget(self.comboBox_9, 7, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 1)
        self.label_44 = QtWidgets.QLabel(self.groupBox_2)
        self.label_44.setEnabled(False)
        self.label_44.setObjectName("label_44")
        self.gridLayout_3.addWidget(self.label_44, 7, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setObjectName("label_17")
        self.gridLayout_3.addWidget(self.label_17, 2, 0, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.groupBox_2)
        self.label_37.setObjectName("label_37")
        self.gridLayout_3.addWidget(self.label_37, 4, 1, 1, 1)
        self.comboBox_8 = ClickedComboBox(self.groupBox_2)
        self.comboBox_8.setObjectName("comboBox_8")
        self.gridLayout_3.addWidget(self.comboBox_8, 6, 1, 1, 1)
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_12.setObjectName("pushButton_12")
        self.gridLayout_3.addWidget(self.pushButton_11, 8, 1, 1, 1)
        self.gridLayout_3.addWidget(self.pushButton_12, 8, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.verticalLayout_4.addWidget(self.groupBox_2)

        # ---- Collect / Recognize ----
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_11)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(
            self.groupBox_7.sizePolicy().hasHeightForWidth()
        )
        self.groupBox_7.setSizePolicy(sizePolicy)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_7.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_5 = QtWidgets.QLabel(self.groupBox_7)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.comboBox = ClickedComboBox(self.groupBox_7)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_3.setFlat(True)
        self.groupBox_3.setCheckable(True)
        self.groupBox_3.setChecked(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setContentsMargins(2, 6, 2, 6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.comboBox_2 = ClickedComboBox(self.groupBox_3)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_3.addWidget(self.comboBox_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.setCheckable(True)
        self.verticalLayout_5.addWidget(self.pushButton_15)
        self.verticalLayout_7.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_4.setFlat(True)
        self.groupBox_4.setCheckable(True)
        self.groupBox_4.setChecked(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setContentsMargins(2, 6, 2, 6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setObjectName("label_3")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.horizontalLayout_5.addWidget(self.lineEdit_6)
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_4)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton.setCheckable(True)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_6.addWidget(self.pushButton)
        self.verticalLayout_7.addWidget(self.groupBox_4)
        self.verticalLayout_4.addWidget(self.groupBox_7)

        # ---- printlog ----
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_11)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint
        )
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textEdit.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.textEdit.setTabStopWidth(80)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.verticalLayout_4.addWidget(self.groupBox)

        # 左侧 2×3 视图区
        self.groupBox_9 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(15)
        sizePolicy.setHeightForWidth(
            self.groupBox_9.sizePolicy().hasHeightForWidth()
        )
        self.groupBox_9.setSizePolicy(sizePolicy)
        self.groupBox_9.setObjectName("groupBox_9")

        # ====== 这里是修改后的 2×3 布局 ======
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setHorizontalSpacing(7)
        self.gridLayout.setObjectName("gridLayout")

        # 顶部标题：RTI, DTI, RDI
        self.label = QtWidgets.QLabel(self.groupBox_9)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_7 = QtWidgets.QLabel(self.groupBox_9)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 1, 1, 1)

        self.label_11 = QtWidgets.QLabel(self.groupBox_9)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 2, 1, 1)

        # 顶部图像：RTI, DTI, RDI
        self.graphicsView = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)

        self.graphicsView_2 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_2.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)

        self.graphicsView_6 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_6.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_6.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.gridLayout.addWidget(self.graphicsView_6, 1, 2, 1, 1)

        # 底部标题：REI, RAI, Camera
        self.label_8 = QtWidgets.QLabel(self.groupBox_9)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 2, 0, 1, 1)

        self.label_9 = QtWidgets.QLabel(self.groupBox_9)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 1, 1, 1)

        self.label_10 = QtWidgets.QLabel(self.groupBox_9)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)

        # 底部图像：REI, RAI, Camera
        self.graphicsView_3 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_3.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout.addWidget(self.graphicsView_3, 3, 0, 1, 1)

        self.graphicsView_4 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_4.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_4.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.gridLayout.addWidget(self.graphicsView_4, 3, 1, 1, 1)

        self.graphicsView_5 = QtWidgets.QLabel(self.groupBox_9)
        self.graphicsView_5.setStyleSheet(
            'border-width: 2px;border-style: solid;border-color: rgb(255, 170, 0);'
            'background-color: rgb(180,180,180);'
        )
        self.graphicsView_5.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_5.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.gridLayout.addWidget(self.graphicsView_5, 3, 2, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 4)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 4)

        # 拼好 splitter
        self.verticalLayout_3.addWidget(self.splitter)
        self.splitter.addWidget(self.groupBox_9)
        self.splitter.addWidget(self.groupBox_11)
        self.splitter.setSizes([960, 320])

        self.tabWidget.addTab(self.tab, "")

        # -------- tab2: waiting... --------
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_8.addWidget(self.label_4)
        self.tabWidget.addTab(self.tab_2, "")

        self.horizontalLayout_4.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1079, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionload = QtWidgets.QAction(MainWindow)
        self.actionload.setObjectName("actionload")
        self.menu.addSeparator()
        self.menu.addAction(self.actionload)
        self.menu.addSeparator()
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.slot_init()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 统一右栏字号
        MainWindow.setStyleSheet("""
        QGroupBox::title { font-size: 12pt; font-weight: 600; }
        QGroupBox, QLabel, QComboBox, QPushButton, QLineEdit { font-size: 11pt; }
        QTextEdit { font-family: Consolas, "Courier New", monospace; font-size: 11pt; }
        """)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RadarStream"))
        self.groupBox_11.setTitle(_translate("MainWindow", "Configuration"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Radar Setup"))
        self.label_15.setText(_translate("MainWindow", "Max Range:"))
        self.label_16.setText(_translate("MainWindow", "80 m"))
        self.label_36.setText(_translate("MainWindow", "Max Velocity:"))
        self.label_14.setText(_translate("MainWindow", "3.75 cm"))
        self.label_43.setText(_translate("MainWindow", "CLI Port:"))
        self.label_13.setText(_translate("MainWindow", "Range Resolution:"))
        self.label_35.setText(_translate("MainWindow", "2 m/s"))
        self.comboBox_7.setToolTip(
            _translate("MainWindow", "Put config files under /config")
        )
        self.label_45.setText(_translate("MainWindow", "More settings—"))
        self.label_12.setText(_translate("MainWindow", "Config file:"))
        self.label_44.setText(_translate("MainWindow", "Data Port:"))
        self.label_17.setText(_translate("MainWindow", "Velocity Resolution:"))
        self.label_37.setText(_translate("MainWindow", "20 m/s"))
        self.pushButton_11.setText(_translate("MainWindow", "send"))
        self.pushButton_12.setText(_translate("MainWindow", "Exit"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Collect / Recognize"))
        self.label_5.setText(_translate("MainWindow", "colormap:"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Recognition"))
        self.label_2.setText(_translate("MainWindow", "model:"))
        self.pushButton_15.setText(_translate("MainWindow", "recognize"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Capture"))
        self.label_3.setText(_translate("MainWindow", "scene:"))
        self.lineEdit_6.setText(_translate("MainWindow", "chaotic_dataset"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Back"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Dblclick"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Down"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Front"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "Left"))
        self.comboBox_3.setItemText(5, _translate("MainWindow", "Right"))
        self.comboBox_3.setItemText(6, _translate("MainWindow", "Up"))
        self.pushButton.setText(_translate("MainWindow", "capture"))
        self.groupBox.setTitle(_translate("MainWindow", "printlog"))
        self.groupBox_9.setTitle(
            _translate("MainWindow", "Real-time Radar Views")
        )
        self.label.setText(_translate("MainWindow", "Range–Time (RTI)"))
        self.label_7.setText(_translate("MainWindow", "Doppler–Time (DTI)"))
        self.label_11.setText(_translate("MainWindow", "Range–Doppler (RDI)"))
        self.label_8.setText(_translate("MainWindow", "Range–Elevation (REI)"))
        self.label_9.setText(_translate("MainWindow", "Range–Azimuth (RAI)"))
        self.label_10.setText(_translate("MainWindow", "Realtime Camera"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab),
            _translate("MainWindow", "real-time system")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_2),
            _translate("MainWindow", "waiting...")
        )
        self.menu.setTitle(_translate("MainWindow", "Menu"))
        self.actionload.setText(_translate("MainWindow", "mini"))
        self.actionload.setToolTip(_translate("MainWindow", "mini window"))

    # ----------------- 下面是右侧逻辑，与原来一样 -----------------
    def printlog(self, textEdit, string, fontcolor='green'):
        textEdit.moveCursor(QtGui.QTextCursor.End)
        gettime = time.strftime("%H:%M:%S", time.localtime())
        textEdit.append(
            "<font color=" + fontcolor + ">" +
            str(gettime) + "-->" + string + "</font>"
        )

    def slot_init(self):
        self.printlog(self.textEdit, 'Welcome!', fontcolor='green')
        self.getcolorlist()
        self.comboBox.currentIndexChanged.connect(self.setcolor)
        self.comboBox_7.arrowClicked.connect(self.configpath)
        self.comboBox_2.arrowClicked.connect(self.modelpath)
        self.groupBox_3.clicked.connect(
            lambda: self.IsRecognizeorCapture(box_name='box_3')
        )
        self.groupBox_4.clicked.connect(
            lambda: self.IsRecognizeorCapture(box_name='box_4')
        )
        self.pushButton_15.clicked.connect(
            lambda: self.Iscapture(
                self.pushButton_15, 'recognizing', 'recognize'
            )
        )
        self.pushButton.clicked.connect(
            lambda: self.Iscapture(self.pushButton, 'capturing', 'capture')
        )

    def IsRecognizeorCapture(self, box_name):
        if box_name == 'box_3' and self.groupBox_3.isChecked():
            self.groupBox_4.setChecked(False)
            self.pushButton.setChecked(False)
            self.Iscapture(self.pushButton, 'capturing', 'capture')
        elif box_name == 'box_4' and self.groupBox_4.isChecked():
            self.groupBox_3.setChecked(False)
            self.pushButton_15.setChecked(False)
            self.Iscapture(self.pushButton_15, 'recognizing', 'recognize')
        elif box_name == 'box_3' and not self.groupBox_3.isChecked():
            gl.set_value('IsRecognizeorCapture', False)
            self.graphicsView_5.setPixmap(
                QtGui.QPixmap("gesture_icons/7.jpg")
            )
        elif box_name == 'box_4' and not self.groupBox_4.isChecked():
            gl.set_value('IsRecognizeorCapture', False)
            self.graphicsView_5.setPixmap(
                QtGui.QPixmap("gesture_icons/7.jpg")
            )

    def Iscapture(self, btn, text1, text2):
        if btn.isChecked():
            gl.set_value('IsRecognizeorCapture', True)
            btn.setText(text1)
        else:
            gl.set_value('IsRecognizeorCapture', False)
            self.printlog(
                self.textEdit, 'IsRecognizeorCapture:False', fontcolor='green'
            )
            self.graphicsView_5.setPixmap(
                QtGui.QPixmap("gesture_icons/7.jpg")
            )
            btn.setText(text2)

    def modelpath(self):
        self.comboBox_2.clear()
        self.comboBox_2.addItem("--select--")
        lst = []
        if os.path.exists(modelpathfile):
            files = os.listdir(modelpathfile)
            for file in files:
                lst.append(modelpathfile + '/' + file)
        self.comboBox_2.addItems(lst)

    def configpath(self):
        self.comboBox_7.clear()
        self.comboBox_7.addItem("--select--")
        lst = []
        if os.path.exists(radarRFconfigpathfile):
            files = os.listdir(radarRFconfigpathfile)
            for file in files:
                lst.append(radarRFconfigpathfile + '/' + file)
        self.comboBox_7.addItems(lst)

    def getcolorlist(self):
        values = _list_cmap_names()
        self.comboBox.addItem("--select--")
        self.comboBox.addItem("customize")
        for value in values:
            self.comboBox.addItem(value)

    def setcolor(self):
        if (self.comboBox.currentText() != '--select--' and
                self.comboBox.currentText() != ''):
            self.printlog(
                self.textEdit,
                'selected color:' + self.comboBox.currentText(),
                fontcolor='blue'
            )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    sys.exit(app.exec_())
