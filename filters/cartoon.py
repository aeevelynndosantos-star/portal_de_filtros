import cv2

from .utils import cinza


def filtro_cartoon(img):
    g = cv2.medianBlur(cinza(img), 7)
    mascara = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    cor = cv2.bilateralFilter(img, 7, 120, 120)
    return cv2.bitwise_and(cor, cor, mask=mascara)
