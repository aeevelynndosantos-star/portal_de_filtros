import cv2
import numpy as np

from .utils import pequeno


def filtro_pixel_8bit(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    bloco = max(4, min(w, h) // 30)
    pequena = cv2.resize(
        img, (max(1, w // bloco), max(1, h // bloco)),
        interpolation=cv2.INTER_AREA
    )
    pequena = (pequena // 85) * 85
    hsv = cv2.cvtColor(pequena, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) * 1.15, 0, 255).astype(np.uint8)
    pequena = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    saida = cv2.resize(pequena, (w, h), interpolation=cv2.INTER_NEAREST)
    saida[2::3, :] = (saida[2::3, :] * 0.6).astype(np.uint8)
    return saida
