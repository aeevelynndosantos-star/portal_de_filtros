import cv2
import numpy as np

from .utils import cinza


def filtro_ciber_neon(img):
    g = cinza(img)
    tingida = cv2.applyColorMap(g, cv2.COLORMAP_COOL)
    base = cv2.addWeighted(tingida, 0.7, img, 0.3, 0)
    bordas = cv2.Canny(g, 80, 160)
    bordas = cv2.dilate(bordas, np.ones((2, 2), np.uint8))
    brilho = np.zeros_like(img)
    brilho[bordas > 0] = (255, 255, 0)
    brilho = cv2.GaussianBlur(brilho, (0, 0), 3)
    return cv2.add(base, brilho)
