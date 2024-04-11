from src.plugin_interface import PluginInterface
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtGui
from .ui_main import Ui_Form
from skimage.segmentation import watershed
from scipy import ndimage
import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

class Controller(QWidget):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = model
        self.image = None
        self.render_image = False   # variabel agar tidak bisa load img ketika sudah ada img
        self.path_img_save = "./plugins/moilapp-plugin-histologi-bat/src"
        self.set_stylesheet()

    def set_stylesheet(self):
        self.ui.label.setStyleSheet(self.model.style_label())
        self.ui.label_2.setStyleSheet(self.model.style_label())
        self.ui.label_4.setStyleSheet(self.model.style_label())
        self.ui.label_5.setStyleSheet(self.model.style_label())
        self.ui.label_6.setStyleSheet(self.model.style_label())
        self.ui.label_7.setStyleSheet(self.model.style_label())
        self.ui.label_8.setStyleSheet(self.model.style_font_12())
        self.ui.lbl_cell.setStyleSheet(self.model.style_font_12())

        self.ui.img_grafik.setStyleSheet(self.model.style_label())
        self.ui.img_ori.setStyleSheet(self.model.style_label())
        self.ui.img_dist.setStyleSheet(self.model.style_label())
        self.ui.img_morph.setStyleSheet(self.model.style_label())
        self.ui.img_canny.setStyleSheet(self.model.style_label())
        self.ui.img_label.setStyleSheet(self.model.style_label())

        self.ui.btn_load.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_clear.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_crop.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_save.setStyleSheet(self.model.style_pushbutton())

        self.ui.btn_load.clicked.connect(self.load_image_1)
        self.ui.btn_crop.clicked.connect(self.load_image_crop)

        self.checkDir(f"{self.path_img_save}")

    # def clearImg(self):


    def load_image_1(self):
        if self.render_image: return    # kalo true bakal kembali
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            if file.partition('.')[-1].upper() in ['APNG', 'AVIF', 'GIF', 'JPEG', 'JPG', 'PNG', 'SVG', 'TIFF', 'WEBP']:
                self.image_original = cv2.imread(file)
                self.image = self.image_original.copy()
                self.render_image = True
                self.show_to_ui_img_1(file)

    def load_image_crop(self):
        self.checkDir(f"{self.path_img_save}/crop")
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            self.image_original = cv2.imread(file)
            self.image = self.image_original.copy()
            self.show_to_ui_img_crop(file)

    def show_to_ui_img_1(self, img_path):
        self.checkDir(f"{self.path_img_save}/tmp")
        img = cv2.imread(img_path)
        size = 400

        self.morp_opr(img)

        switch_obj = cv2.imread(f"{self.path_img_save}/tmp/switch-obj.png")

        self.labelling(switch_obj)
        self.count_cell()

        canny = cv2.imread(f"{self.path_img_save}/tmp/canny.png")
        distace = cv2.imread(f"{self.path_img_save}/tmp/distance.png")

        self.model.show_image_to_label(self.ui.img_ori, img, size)
        self.model.show_image_to_label(self.ui.img_morph, switch_obj, size)
        self.model.show_image_to_label(self.ui.img_canny, canny, size)
        self.model.show_image_to_label(self.ui.img_dist, distace, size)
        self.model.show_image_to_label(self.ui.img_label, self.image_original, size)

        self.graph()
        graph = cv2.imread(f"{self.path_img_save}/tmp/graph.png")
        self.model.show_image_to_label(self.ui.img_grafik, graph, size)
        plt.close("all")

    def morp_opr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img_mop = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
        img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
        cv2.imwrite(f"{self.path_img_save}/tmp/morp.png", img_mop)

        # switch objek
        img_morp = img_mop.copy()
        for i in range(0, img_morp.shape[0]):
            for j in range(0, img_morp.shape[1]):
                px = 255 if img_morp[i][j] == 0 else 0
                img_morp[i][j] = px
        cv2.imwrite(f"{self.path_img_save}/tmp/switch-obj.png", img_morp)

        # return img_mop

    def switch_obj(self):
        img_morp = cv2.imread(f"{self.path_img_save}/tmp/morp.png", 0)
        for i in range(0, img_morp.shape[0]):
            for j in range(0, img_morp.shape[1]):
                px = 255 if img_morp[i][j] == 0 else 0
                img_morp[i][j] = px
        cv2.imwrite(f"{self.path_img_save}/tmp/switch-obj.png", img_morp)
        # return  img_morp


class HistologiBat(PluginInterface):
    def __init__(self):
        super().__init__()
        self.widget = None
        self.description = "This is a plugins application"

    def set_plugin_widget(self, model):
        self.widget = Controller(model)
        return self.widget

    def set_icon_apps(self):
        return "icon.png"

    def change_stylesheet(self):
        self.widget.set_stylesheet()
