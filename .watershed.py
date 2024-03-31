import numpy as np
import cv2 as cv
from matplotlib import  pyplot as plt

def app():
    img = cv.imread("../historical_function/img_crop_2_4.png")

    # operasi mengganti objek yg dideteksi
    thres = thresholding(img)
    morphological_opr(thres)
    switch()

    # operasi mendeteksi objek yg telah diganti
    citra = cv.imread("revert.png")

    plt.subplot(241)
    plt.imshow(citra[..., :: -1])
    plt.xticks([]), plt.yticks([])
    plt.title('citra asal')

    citraBiner = gray(citra)
    pembukaan, kernel = morphology(citraBiner)
    ltrBelakang = latarBelakang(pembukaan, kernel)
    ltrDepan = latarDepan(pembukaan)
    drhTakBertuan = daerahTakBertuan(ltrDepan, ltrBelakang)
    label = penanda(ltrDepan)
    watershed(img, citra, label, drhTakBertuan)
    plt.show()

def thresholding(img):
    # konversi gambar abu-abu jadi biner
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    return thresh

def morphological_opr(thresh):
    # membersihkan gambar biner atau menghilangkan noise pada objek
    img_mop = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (50,50)))
    img_mop = cv.morphologyEx(img_mop, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1)))
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

def daerahTakBertuan(latarDepan, latarBelakang):
    latarDepan = np.uint8(latarDepan)
    daerahTakBertuan = cv.subtract(latarBelakang, latarDepan)
    plt.subplot(246)
    plt.imshow(daerahTakBertuan, cmap="gray", vmin=0, vmax=255)
    plt.xticks([]), plt.yticks([])
    plt.title('tak bertuan')

    return daerahTakBertuan

def penanda(latarDepan):
    latarDepan = np.uint8(latarDepan)
    jumObjek, penanda = cv.connectedComponents(latarDepan)
    print("jumlah koin:", jumObjek - 1)
    return penanda

def watershed(img, citra, penanda, daerahTakBertuan):
    penanda = penanda + 1
    penanda[daerahTakBertuan == 255] = 0

    penanda = cv.watershed(citra, penanda)
    plt.subplot(247)
    plt.imshow(penanda, cmap="jet")
    plt.xticks([]), plt.yticks([])
    plt.title('penanda')

    img[penanda == -1] = [0, 255, 0]
    plt.subplot(248)
    plt.imshow(img[..., :: -1])
    cv.imwrite("hasil.png", img)
    plt.xticks([]), plt.yticks([])
    plt.title('hasil akhir')

app()