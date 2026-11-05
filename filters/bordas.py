import cv2
import numpy as np

from .utils import cinza


def filtro_bordas(img):
    bordas = cv2.Canny(
        cv2.GaussianBlur(cinza(img), (5, 5), 0), 60, 140
    )
    saida = np.zeros_like(img)
    saida[bordas > 0] = (0, 255, 255)
    return saida
