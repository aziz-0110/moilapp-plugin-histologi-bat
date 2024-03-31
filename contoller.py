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
        # This is set up style label on bonding box ui
        self.ui.label_ori_1.setStyleSheet(self.model.style_label())
        # self.ui.label_ori_1.setFixedHeight(484)
        self.ui.label_1_1.setStyleSheet(self.model.style_label())
        self.ui.label_1_2.setStyleSheet(self.model.style_label())
        self.ui.label_1_3.setStyleSheet(self.model.style_label())
        self.ui.label_1_4.setStyleSheet(self.model.style_label())
        # self.ui.label_ori_2.setStyleSheet(self.model.style_label())
        self.ui.label_3.setStyleSheet(self.model.style_label())
        # self.ui.label_3.setFixedWidth(85)

        # self.ui.label_2_1.setStyleSheet(self.model.style_label())
        # self.ui.label_2_2.setStyleSheet(self.model.style_label())
        # self.ui.label_2_3.setStyleSheet(self.model.style_label())
        # self.ui.label_2_4.setStyleSheet(self.model.style_label())
        # self.ui.label_13.setStyleSheet(self.model.style_label())
        # self.ui.label_14.setStyleSheet(self.model.style_label())

        self.ui.label.setStyleSheet(self.model.style_label())
        self.ui.label.setFixedHeight(30)    # mengatur paksa tinggi label
        self.ui.label_2.setStyleSheet(self.model.style_label())
        self.ui.label_4.setStyleSheet(self.model.style_label())
        self.ui.label_5.setStyleSheet(self.model.style_label())
        # self.ui.label_6.setStyleSheet(self.model.style_label())
        # self.ui.label_7.setStyleSheet(self.model.style_label())
        # self.ui.label_9.setStyleSheet(self.model.style_label())
        # self.ui.label_10.setStyleSheet(self.model.style_label())
        # self.ui.label_11.setStyleSheet(self.model.style_label())
        self.ui.label_12.setStyleSheet(self.model.style_label())
        # self.ui.label_15.setStyleSheet(self.model.style_label())

        self.ui.line.setStyleSheet(self.model.style_line())
        self.ui.line_2.setStyleSheet(self.model.style_line())
        # self.ui.line_3.setStyleSheet(self.model.style_line())
        self.ui.line_4.setStyleSheet(self.model.style_line())
        # self.ui.line_5.setStyleSheet(self.model.style_line())

        # This is set up style button on bonding box ui
        self.ui.load_img1.setStyleSheet(self.model.style_pushbutton())
        # self.ui.multi_1.setStyleSheet(self.model.style_pushbutton())
        self.ui.params_1.setStyleSheet(self.model.style_pushbutton())
        self.ui.clear_1.setStyleSheet(self.model.style_pushbutton())

        # self.ui.load_img2.setStyleSheet(self.model.style_pushbutton())
        # self.ui.multi_2.setStyleSheet(self.model.style_pushbutton())
        # self.ui.params_2.setStyleSheet(self.model.style_pushbutton())
        # self.ui.clear_2.setStyleSheet(self.model.style_pushbutton())

        # This is set up to connect to other function to make action on ui
        self.ui.load_img1.clicked.connect(self.load_image_1)
        self.ui.params_1.clicked.connect(self.load_image_crop)
        self.ui.clear_1.clicked.connect(self.clearImg)

        # This is set up to connect to other function to make action on ui
        # self.ui.load_img2.clicked.connect(self.load_image_2)
        # self.ui.multi_2.clicked.connect(self.cam_params)
        # self.ui.params_2.clicked.connect(self.cam_params)
        # self.ui.clear_2.clicked.connect(self.cam_params)

    def clearImg(self):
        self.image = None
        self.render_image = False
        self.ui.label_14.setText("")
        # fungsi untuk membersihkan gambar yg sudah di load
        self.ui.label_ori_1.clear()
        self.ui.label_1_1.clear()
        self.ui.label_1_2.clear()
        self.ui.label_1_3.clear()
        self.ui.label_1_4.clear()

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
        dir_img_save_path = "./plugins/moilapp-plugin-histologi-bat/saved_img/HFD"

        gray = self.convert_grayscale(img)
        thresh = self.thresholding(gray)
        morpho = self.morphological_opr(thresh)
        cells, cell_count = self.count_cells(img_path, dir_img_save_path)

        self.model.show_image_to_label(self.ui.label_ori_1, self.image_original, 620)
        self.model.show_image_to_label(self.ui.label_1_1, thresh, 300)
        self.model.show_image_to_label(self.ui.label_1_2, cells, 300)
        self.model.show_image_to_label(self.ui.label_1_3, morpho, 300)
        # self.model.show_image_to_label(self.ui.label_1_4, , 300)

        self.ui.label_14.setText(f"{cell_count}")

        # self.crop_img(dir_img_save_path)
        self.graph()

    def show_to_ui_img_crop(self, img_path):
        # img = cv2.imread(img_path)

        dir_img_save_path = "./plugins/moilapp-plugin-histologi-bat-git/saved_img/crop"

        self.model.show_image_to_label(self.ui.label_ori_1, self.image_original, 620)
        self.crop_img(dir_img_save_path, img_path)

    def cam_params(self):
        self.model.form_camera_parameter()

    def thresholding(img):
        # konversi gambar abu-abu jadi biner
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        return thresh

    def morphological_opr(thresh):
        # membersihkan gambar biner atau menghilangkan noise pada objek
        img_mop = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)))
        img_mop = cv.morphologyEx(img_mop, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)))
        cv.imwrite("morp.png", img_mop)

    def switch():
        img = cv.imread("morp.png", 0)

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                px = 255 if img[i][j] == 0 else 0
                img[i][j] = px

        cv.imwrite("revert.png", img)

    def gray(citra):
        abuAbu = cv.cvtColor(citra, cv.COLOR_BGR2GRAY)
        ambang, citraBiner = cv.threshold(abuAbu, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        plt.subplot(242)
        plt.imshow(citraBiner, cmap="gray", vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])
        plt.title('citra biner')
        return citraBiner

    def morphology(citraBiner):
        kernel = np.ones((3, 3), np.uint8)
        pembukaan = cv.morphologyEx(citraBiner, cv.MORPH_OPEN, kernel, iterations=2)

        plt.subplot(243)
        plt.imshow(pembukaan, cmap="gray", vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])
        plt.title('pembukaan')
        return pembukaan, kernel

    def latarBelakang(pembukaan, kernel):
        latarBkg = cv.dilate(pembukaan, kernel, iterations=2)
        plt.subplot(244)
        plt.imshow(pembukaan, cmap="gray", vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])
        plt.title('latar belakang')
        return latarBkg

    def latarDepan(pembukaan):
        transformjarak = cv.distanceTransform(pembukaan, cv.DIST_L2, cv.DIST_MASK_5)
        ambang, latarDpn = cv.threshold(transformjarak, 0.24 * transformjarak.max(), 255, cv.THRESH_BINARY)

        plt.subplot(245)
        cv.imwrite("latarDepan.png", latarDpn)
        plt.imshow(latarDpn, cmap="gray", vmin=0, vmax=255)
        plt.xticks([]), plt.yticks([])
        plt.title('latar depan')
        return latarDpn

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
