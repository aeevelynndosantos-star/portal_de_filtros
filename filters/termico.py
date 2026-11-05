import cv2

from .utils import cinza


def filtro_termico(img):
    return cv2.applyColorMap(cinza(img), cv2.COLORMAP_JET)
