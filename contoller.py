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
        self.set_stylesheet()

    def set_stylesheet(self):
        self.ui.img_ori.setStyleSheet(self.model.style_frame_object())
        self.ui.img_thres.setStyleSheet(self.model.style_label())
        self.ui.img_morph.setStyleSheet(self.model.style_label())
        self.ui.img_canny.setStyleSheet(self.model.style_label())
        self.ui.img_label.setStyleSheet(self.model.style_label())
        self.ui.img_grafik.setStyleSheet(self.model.style_label())


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
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            self.image_original = cv2.imread(file)
            self.image = self.image_original.copy()
            self.show_to_ui_img_crop(file)

    def show_to_ui_img_1(self, img_path):
        # self.con
        img = cv2.imread(img_path)

        self.graph()

    def show_to_ui_img_crop(self, img_path):
        dir_img_save_path = "./plugins/moilapp-plugin-histologi-bat-git/saved_img/crop"

        self.model.show_image_to_label(self.ui.label_ori_1, self.image_original, 620)
        self.crop_img(dir_img_save_path, img_path)

    def crop_img(self, dir_path, img_path):
        # jumlah potongan gambar
        jmh_crop = 4

        # gambar yg sudah di labeling
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # menghitung ukuran gambar untuk dipotong
        row_start, row_end = self.count_crop_img(jmh_crop, height)
        col_start, col_end = self.count_crop_img(jmh_crop, width)

        # img_save_path = f"{dir_path}/count_cell.png"

        if (os.path.exists(f"{dir_path}")):
            if (os.path.isdir(f"{dir_path}")):
                os.system(f"rm -R {dir_path}")
                os.mkdir(f"{dir_path}")
                # cv2.imwrite(img_save_path, image)
        else:
            os.mkdir(f"{dir_path}")

        for i in range(0, jmh_crop):
            for j in range(0, jmh_crop):
                # memotong gambar
                cropped = img[row_start[i]:row_end[i], col_start[j]:col_end[j]]

                # menyimpan gambar
                cv2.imwrite(f"{dir_path}/img_crop_{i + 1}_{j + 1}.png", cropped)

    def count_crop_img(self, jmh_crop, size_img):
        start_crop = []
        end_crop = []

        # perhitungan ukuran potongan gambar
        size_crop = int(size_img / jmh_crop)
        for i in range(0, jmh_crop + 1):
            # fungsi append untuk menambahkan nilai list
            end_crop.append(i * size_crop)
            start_crop.append(end_crop[i] - size_crop)

        # fungsi pop untuk menhapus list index 0
        start_crop.pop(0)
        end_crop.pop(0)
        return start_crop, end_crop

    def graph(self):    # untuk grafik

        species = ("Adelie", "Chinstrap", "Gentoo")
        penguin_means = {
            '0,02': (18.35, 18.43, 14.98),
            '0,01': (38.79, 48.83, 47.50),
            '0,03': (189.95, 195.82, 217.19),
        }

        x = np.arange(len(species))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Length (mm)')
        ax.set_title('Penguin attributes by species')
        ax.set_xticks(x + width, species)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 250)

        plt.show()


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
