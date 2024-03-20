from src.plugin_interface import PluginInterface
from PyQt6.QtWidgets import QWidget
from .ui_main import Ui_Form
import cv2


class Controller(QWidget):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = model
        self.image = None
        self.set_stylesheet()

    def set_stylesheet(self):
        # This is set up style label on bonding box ui
        self.ui.label_ori_1.setStyleSheet(self.model.style_label())
        self.ui.label_1_1.setStyleSheet(self.model.style_label())
        self.ui.label_1_2.setStyleSheet(self.model.style_label())
        self.ui.label_1_3.setStyleSheet(self.model.style_label())
        self.ui.label_1_4.setStyleSheet(self.model.style_label())
        self.ui.label_ori_2.setStyleSheet(self.model.style_label())
        self.ui.label_3.setStyleSheet(self.model.style_label())

        self.ui.label_2_1.setStyleSheet(self.model.style_label())
        self.ui.label_2_2.setStyleSheet(self.model.style_label())
        self.ui.label_2_3.setStyleSheet(self.model.style_label())
        self.ui.label_2_4.setStyleSheet(self.model.style_label())
        self.ui.label_13.setStyleSheet(self.model.style_label())
        self.ui.label_14.setStyleSheet(self.model.style_label())

        self.ui.label.setStyleSheet(self.model.style_label())
        self.ui.label_2.setStyleSheet(self.model.style_label())
        self.ui.label_4.setStyleSheet(self.model.style_label())
        self.ui.label_5.setStyleSheet(self.model.style_label())
        self.ui.label_6.setStyleSheet(self.model.style_label())
        self.ui.label_7.setStyleSheet(self.model.style_label())
        self.ui.label_9.setStyleSheet(self.model.style_label())
        self.ui.label_10.setStyleSheet(self.model.style_label())
        self.ui.label_11.setStyleSheet(self.model.style_label())
        self.ui.label_12.setStyleSheet(self.model.style_label())
        self.ui.label_15.setStyleSheet(self.model.style_label())

        self.ui.line.setStyleSheet(self.model.style_line())
        self.ui.line_2.setStyleSheet(self.model.style_line())
        self.ui.line_3.setStyleSheet(self.model.style_line())
        self.ui.line_4.setStyleSheet(self.model.style_line())
        self.ui.line_5.setStyleSheet(self.model.style_line())

        # This is set up style button on bonding box ui
        self.ui.load_img1.setStyleSheet(self.model.style_pushbutton())
        self.ui.multi_1.setStyleSheet(self.model.style_pushbutton())
        self.ui.params_1.setStyleSheet(self.model.style_pushbutton())
        self.ui.clear_1.setStyleSheet(self.model.style_pushbutton())

        self.ui.load_img2.setStyleSheet(self.model.style_pushbutton())
        self.ui.multi_2.setStyleSheet(self.model.style_pushbutton())
        self.ui.params_2.setStyleSheet(self.model.style_pushbutton())
        self.ui.clear_2.setStyleSheet(self.model.style_pushbutton())

        # This is set up to connect to other function to make action on ui
        self.ui.load_img1.clicked.connect(self.load_image_1)
        self.ui.multi_1.clicked.connect(self.cam_params)
        self.ui.params_1.clicked.connect(self.cam_params)
        self.ui.clear_1.clicked.connect(self.cam_params)

        # This is set up to connect to other function to make action on ui
        self.ui.load_img2.clicked.connect(self.load_image_2)
        self.ui.multi_2.clicked.connect(self.cam_params)
        self.ui.params_2.clicked.connect(self.cam_params)
        self.ui.clear_2.clicked.connect(self.cam_params)


    def load_image_1(self):
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            self.image_original = cv2.imread(file)
            self.image = self.image_original.copy()
            self.show_to_ui_img_1()

    def load_image_2(self):
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            self.image_original = cv2.imread(file)
            self.image = self.image_original.copy()
            self.show_to_ui_img_2()

    # def count_cells(image_path):
    #     image = cv2.imread(image_path)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     img_mop = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    #     img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    #     D = ndimage.distance_transform_edt(img_mop)
    #     localMax = peak_local_max(D, indices=False, min_distance=10, labels=img_mop)
    #     markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    #     labels = watershed(-D, markers, mask=img_mop)
    #
    #     cell_count = len(np.unique(labels)) - 1
    #
    #     for label in np.unique(labels):
    #         if label == 255:
    #             continue
    #         mask = np.zeros(D.shape, dtype="uint8")
    #         mask[labels == label] = 255
    #         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         cnts = imutils.grab_contours(cnts)
    #         c = max(cnts, key=cv2.contourArea)
    #         ((x, y), r) = cv2.minEnclosingCircle(c)
    #         cv2.circle(image, (int(x), int(y)), int(r), (255, 61, 139), 1, 5)
    #         cv2.putText(image, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)
    #
    #     return image, cell_count

    # Example usage:   # image_path = 'images/test_image1.png'
    #     # result_image, cell_count = count_cells(image_path)
    #     # cv2.imshow("Result Image", result_image)
    #     # print("White Adipose Count: {} ".format(cell_count))
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()


    def show_to_ui_img_1(self):
        self.model.show_image_to_label(self.ui.label_ori_1, self.image, 500)

    def show_to_ui_img_2(self):
        self.model.show_image_to_label(self.ui.label_ori_2, self.image, 500)

    def cam_params(self):
        self.model.form_camera_parameter()


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

