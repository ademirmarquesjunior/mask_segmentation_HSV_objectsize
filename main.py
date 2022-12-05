from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import sys
import numpy as np
from PIL import Image
import glob
import os

from threading import Thread
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


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('view.ui', self)

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
        self.show()

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
        self.statusBar().showMessage("Idle")


        # Scene definitions
        self.pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.addItem(self.pixmap)
        self.graphicsView.setScene(self.scene)


    def open_folder(self, mode):
        try:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a folder")
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


        image = cv2.resize(image, (400, 400))
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
        self.statusBar().showMessage("Processing")

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
            
        #     self.statusBar().showMessage("Processing " + image_address+"")

        #     image = cv2.imread(image_address)


        #     mask = segment_mask(image, self.object_size, self.low_hsv, self.high_hsv)


        #     cv2.imwrite(self.output_path+image_address[:-4]+"_mask.png", mask)

        #     self.statusBar().showMessage("Saved "+image_address[:-4]+"_mask.png"+" into the output folder.")

        #     progress_bar_count += 1
        #     self.progressBar.setValue(int((progress_bar_count/progress_bar_max)*100))
        #     QtWidgets.QApplication.processEvents()
            
            
        self.statusBar().showMessage("Done!")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()