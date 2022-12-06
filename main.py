# from PyQt5 import uic, QtWidgets
# from PyQt5.QtGui import QPixmap, QImage

# from PySide6.QtCore import *  # type: ignore
# from PySide6.QtGui import *  # type: ignore
# from PySide6.QtWidgets import *  # type: ignore

import multiprocessing
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import *


from PyQt5.QtCore import * #(QRect, QSize, QCoreApplication, QMetaObject, Qt)

from PyQt5.QtWidgets import (QWidget, QPushButton, QFrame, QSlider,
                             QHBoxLayout, QVBoxLayout, QLineEdit, QGraphicsView,
                             QLayout, QSpinBox, QLabel, QProgressBar, QMenuBar,
                             QMenu, QStatusBar, QAction)


import sys
import numpy as np
# from PIL import Image
import glob
import os

# from threading import Thread
from multiprocessing import Process, Array


import cv2


def bwareaopen(img, min_size=7000000, connectivity=8):
    """Remove small objects from binary image (approximation of 
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)

    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]

        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0

    return img

def segment_mask(image, object_size, low_hsv, high_hsv):

    hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(low_hsv) # [35,50,20]
    upper_green = np.array(high_hsv) # [190,255,255]

    segmented = cv2.inRange(hsv, lower_green, upper_green)
    inverted = cv2.bitwise_not(segmented)

    kernel = np.ones((30,30),np.uint8)
    closing = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    mask = bwareaopen(img=closing, min_size=object_size, connectivity=8)

    return mask


def parallel_function(image_address, input_path, output_path, object_size, low_hsv, high_hsv):

    image = cv2.imread(input_path+image_address)

    mask = segment_mask(image, object_size, low_hsv, high_hsv)


    cv2.imwrite(output_path+image_address[:-4]+"_mask.png", mask)

    return


# class Ui(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(Ui, self).__init__()
#         # uic.loadUi('view.ui', self)

#         # Variables initial values
#         self.input_path = ''
#         self.output_path = ''
#         self.object_size = 700000
#         self.low_hsv = [35,50,20]
#         self.high_hsv = [190,255,255]
#         self.image_list = None
#         self.image_position = 0
#         self.opacity = 0.5

#         self.preview = None
#         self.preview_mask = None

#         self.init_Ui()
#         self.show()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(924, 641)
        icon = QIcon()
        icon.addFile(u"vizlab.ico", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayoutWidget =  QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 901, 598))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.centralwidget.setLayout(self.verticalLayout)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.labelInputFolder = QLineEdit(self.verticalLayoutWidget)
        self.labelInputFolder.setObjectName(u"labelInputFolder")

        self.horizontalLayout_3.addWidget(self.labelInputFolder)

        self.openInputFolder = QPushButton(self.verticalLayoutWidget)
        self.openInputFolder.setObjectName(u"openInputFolder")
        self.openInputFolder.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_3.addWidget(self.openInputFolder)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.labelOutputFolder = QLineEdit(self.verticalLayoutWidget)
        self.labelOutputFolder.setObjectName(u"labelOutputFolder")

        self.horizontalLayout_2.addWidget(self.labelOutputFolder)

        self.selectOutputFolder = QPushButton(self.verticalLayoutWidget)
        self.selectOutputFolder.setObjectName(u"selectOutputFolder")
        self.selectOutputFolder.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_2.addWidget(self.selectOutputFolder)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetNoConstraint)
        self.horizontalLayout.setContentsMargins(0, -1, -1, -1)
        self.scrollLeftButton = QPushButton(self.verticalLayoutWidget)
        self.scrollLeftButton.setObjectName(u"scrollLeftButton")

        self.horizontalLayout.addWidget(self.scrollLeftButton)

        self.graphicsView = QGraphicsView(self.verticalLayoutWidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setMinimumSize(QSize(400, 400))
        self.graphicsView.viewport().setProperty("cursor", QCursor(Qt.CrossCursor))
        self.graphicsView.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.graphicsView)

        self.scrollRightButton = QPushButton(self.verticalLayoutWidget)
        self.scrollRightButton.setObjectName(u"scrollRightButton")

        self.horizontalLayout.addWidget(self.scrollRightButton)

        self.line = QFrame(self.verticalLayoutWidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line)

        self.frame = QFrame(self.verticalLayoutWidget)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(180, 500))
        self.frame.setMaximumSize(QSize(180, 500))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.spinSL = QSpinBox(self.frame)
        self.spinSL.setObjectName(u"spinSL")
        self.spinSL.setGeometry(QRect(60, 30, 42, 22))
        self.spinSL.setMaximum(255)
        self.spinSL.setValue(50)
        self.processButton = QPushButton(self.frame)
        self.processButton.setObjectName(u"processButton")
        self.processButton.setGeometry(QRect(10, 380, 141, 71))
        self.spinObjectSize = QSpinBox(self.frame)
        self.spinObjectSize.setObjectName(u"spinObjectSize")
        self.spinObjectSize.setGeometry(QRect(10, 160, 141, 22))
        self.spinObjectSize.setMaximum(3000000)
        self.spinObjectSize.setValue(700000)
        self.spinVL = QSpinBox(self.frame)
        self.spinVL.setObjectName(u"spinVL")
        self.spinVL.setGeometry(QRect(110, 30, 42, 22))
        self.spinVL.setMaximum(255)
        self.spinVL.setValue(20)
        self.previewButton = QPushButton(self.frame)
        self.previewButton.setObjectName(u"previewButton")
        self.previewButton.setGeometry(QRect(10, 200, 141, 71))
        self.sliderOpacity = QSlider(self.frame)
        self.sliderOpacity.setObjectName(u"sliderOpacity")
        self.sliderOpacity.setGeometry(QRect(0, 330, 160, 22))
        self.sliderOpacity.setMaximum(100)
        self.sliderOpacity.setValue(50)
        self.sliderOpacity.setOrientation(Qt.Horizontal)
        self.opacityValueText = QLabel(self.frame)
        self.opacityValueText.setObjectName(u"opacityValueText")
        self.opacityValueText.setGeometry(QRect(100, 300, 49, 16))
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 130, 101, 21))
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 10, 91, 16))
        self.spinSH = QSpinBox(self.frame)
        self.spinSH.setObjectName(u"spinSH")
        self.spinSH.setGeometry(QRect(60, 90, 42, 22))
        self.spinSH.setMaximum(255)
        self.spinSH.setValue(255)
        self.spinVH = QSpinBox(self.frame)
        self.spinVH.setObjectName(u"spinVH")
        self.spinVH.setGeometry(QRect(110, 90, 42, 22))
        self.spinVH.setMaximum(255)
        self.spinVH.setValue(255)
        self.label_4 = QLabel(self.frame)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 300, 49, 16))
        self.spinHL = QSpinBox(self.frame)
        self.spinHL.setObjectName(u"spinHL")
        self.spinHL.setGeometry(QRect(10, 30, 42, 22))
        self.spinHL.setMaximum(255)
        self.spinHL.setValue(35)
        self.spinHH = QSpinBox(self.frame)
        self.spinHH.setObjectName(u"spinHH")
        self.spinHH.setGeometry(QRect(10, 90, 42, 22))
        self.spinHH.setMaximum(255)
        self.spinHH.setValue(190)
        self.label_3 = QLabel(self.frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 60, 91, 16))

        self.horizontalLayout.addWidget(self.frame)

        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.progressBar = QProgressBar(self.verticalLayoutWidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(15)
        self.progressBar.setTextVisible(False)

        self.verticalLayout.addWidget(self.progressBar)

        self.verticalLayout.setStretch(2, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 924, 22))
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuHelp.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

        # Variables initial values
        self.input_path = ''
        self.output_path = ''
        self.object_size = 700000
        self.low_hsv = [35,50,20]
        self.high_hsv = [190,255,255]
        self.image_list = None
        self.image_position = 0
        self.opacity = 0.5

        self.preview = None
        self.preview_mask = None
        self.init_Ui()
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Vizlab Mask/object segmentation", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.openInputFolder.setText(QCoreApplication.translate("MainWindow", u"Input folder", None))
        self.selectOutputFolder.setText(QCoreApplication.translate("MainWindow", u"Output folder", None))
        self.scrollLeftButton.setText(QCoreApplication.translate("MainWindow", u"<", None))
        self.scrollRightButton.setText(QCoreApplication.translate("MainWindow", u">", None))
        self.processButton.setText(QCoreApplication.translate("MainWindow", u"Process masks", None))
        self.previewButton.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.opacityValueText.setText(QCoreApplication.translate("MainWindow", u"50", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Min size of object", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Lower HSV limit", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Opacity", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Higher HSV limit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi


    def init_Ui(self):
        self.openInputFolder.clicked.connect(lambda: self.open_folder(0))
        self.selectOutputFolder.clicked.connect(lambda: self.open_folder(1))

        self.spinHL.valueChanged.connect(self.preview_segmentation)
        self.spinSL.valueChanged.connect(self.preview_segmentation)
        self.spinVL.valueChanged.connect(self.preview_segmentation)
        self.spinHH.valueChanged.connect(self.preview_segmentation)
        self.spinSH.valueChanged.connect(self.preview_segmentation)
        self.spinVH.valueChanged.connect(self.preview_segmentation)
        self.spinObjectSize.valueChanged.connect(self.preview_segmentation)

        self.previewButton.clicked.connect(self.preview_segmentation)
        # self.originalButton.clicked.connect(self.preview_original)
        self.scrollLeftButton.clicked.connect(lambda: self.preview_original(-1))
        self.scrollRightButton.clicked.connect(lambda: self.preview_original(1))

        self.sliderOpacity.valueChanged.connect(self.set_opacity)

        self.processButton.clicked.connect(self.process_images)
        self.progressBar.setValue(0)
        self.statusbar.showMessage("Idle")


        # Scene definitions
        self.pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene = QtWidgets.QGraphicsScene(self.centralwidget)
        self.scene.addItem(self.pixmap)
        self.graphicsView.setScene(self.scene)


    def open_folder(self, mode):
        try:
            path = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Open a folder")
            if path != ('', ''):
                if mode == 0:
                    self.input_path = path + '/'
                    self.labelInputFolder.setText(self.input_path)

                    os.chdir(self.input_path)
                    self.image_list = glob.glob('*.JPG')
                    self.preview = cv2.imread(self.image_list[0])

                    # self.preview = cv2.resize(self.preview, (500, 500))

                    self.update_canvas()


                if mode == 1:
                    self.output_path = path + '/'
                    self.labelOutputFolder.setText(self.output_path)
        except Exception as e:
            print(e)


    def array_to_QPixmap(self, image):
        if np.size(image.shape)==3:
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine,
                        QImage.Format_RGB888).rgbSwapped()
        if np.size(image.shape)==2:
            height, width = image.shape
            bytesPerLine = width
            qImg = QImage(image.data, width, height, bytesPerLine,
                        QImage.Format_Grayscale8)


        return qImg


    def update_canvas(self):
        if type(self.preview) != type(None) and type(self.preview_mask) == type(None):
            image = self.preview
        if type(self.preview_mask) != type(None) and type(self.preview_mask) != type(None):
            # print(np.shape(self.preview))
            # print(np.shape(self.preview_mask))
            image = cv2.addWeighted(self.preview, 0.5, self.preview_mask, self.opacity, 0.0)


        image = cv2.resize(image, (500, 500))
        q_img = self.array_to_QPixmap(image)
        pixmap_image = QPixmap.fromImage(q_img)
        self.pixmap.setPixmap(pixmap_image)


    def set_opacity(self):
        self.opacity = self.sliderOpacity.value()/100
        self.opacityValueText.setText(str(self.sliderOpacity.value()))
        self.update_canvas()


    def preview_segmentation(self):

        self.low_hsv = [self.spinHL.value(), self.spinSL.value(), self.spinVL.value()]
        self.high_hsv = [self.spinHH.value(), self.spinSH.value(), self.spinVH.value()]

        self.object_size = self.spinObjectSize.value()

        # image = cv2.resize(self.preview, (int(self.preview.shape[1]/2), int(self.preview.shape[0]/2)))

        self.preview_mask = segment_mask(self.preview, self.object_size, self.low_hsv, self.high_hsv)

        self.preview_mask = cv2.cvtColor(self.preview_mask, cv2.COLOR_GRAY2BGR)

        # self.update_canvas(cv2.cvtColor(np.uint8(mask), cv2.COLOR_GRAY2BGR))
        self.update_canvas()

        return

    def preview_original(self, direction=None):

        if direction== -1:
            self.image_position = self.image_position-1

        if direction== 1:
            self.image_position = self.image_position+1

        if self.image_position < 0:
            self.image_position = 0
        if self.image_position > np.size(self.image_list) - 1:
            self.image_position = np.size(self.image_list) - 1


        self.preview = cv2.imread(self.image_list[self.image_position])

        self.update_canvas()

        return


    def process_images(self):

        self.low_hsv = [self.spinHL.value(), self.spinSL.value(), self.spinVL.value()]
        self.high_hsv = [self.spinHH.value(), self.spinSH.value(), self.spinVH.value()]
        self.object_size = self.spinObjectSize.value()


        os.chdir(self.input_path)

        image_list = glob.glob('*.JPG')
        

        progress_bar_max = np.size(image_list)
        progress_bar_count = 0
        self.progressBar.setValue(0)
        self.statusbar.showMessage("Processing")

        processes = list()
        for index in range(np.size(image_list)):
            p = Process(target=parallel_function, args=(str(image_list[index]), str(self.input_path), 
            str(self.output_path), int(self.object_size), Array('i', self.low_hsv), Array('i', self.high_hsv)))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            progress_bar_count += 1
            self.progressBar.setValue(int((progress_bar_count/progress_bar_max)*100))
            QtWidgets.QApplication.processEvents()


        # for image_address in image_list:            
            
        #     self.statusbar().showMessage("Processing " + image_address+"")

        #     image = cv2.imread(image_address)


        #     mask = segment_mask(image, self.object_size, self.low_hsv, self.high_hsv)


        #     cv2.imwrite(self.output_path+image_address[:-4]+"_mask.png", mask)

        #     self.statusBar().showMessage("Saved "+image_address[:-4]+"_mask.png"+" into the output folder.")

        #     progress_bar_count += 1
        #     self.progressBar.setValue(int((progress_bar_count/progress_bar_max)*100))
        #     QtWidgets.QApplication.processEvents()
            
            
        self.statusbar.showMessage("Done!")


class MainWindow(QtWidgets.QMainWindow):
    def resizeEvent(self, event):
        print("Window has been resized")
        QtWidgets.QMainWindow.resizeEvent(self, event)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow() #QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    app.exec_()