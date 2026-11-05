import cv2
import numpy as np

from .utils import cinza, pequeno


def filtro_pixel_jogo(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    bloco = max(5, min(w, h) // 36)
    pequena = cv2.resize(
        img, (max(1, w // bloco), max(1, h // bloco)),
        interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(pequena, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) * 1.5, 0, 255).astype(np.uint8)
    pequena = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    pequena = (pequena // 64) * 64 + 32

    g = cinza(pequena)
    bordas = cv2.Canny(g, 40, 90)
    bordas = cv2.dilate(bordas, np.ones((2, 2), np.uint8))
    pequena[bordas > 0] = (25, 15, 25)
    return cv2.resize(pequena, (w, h), interpolation=cv2.INTER_NEAREST)
