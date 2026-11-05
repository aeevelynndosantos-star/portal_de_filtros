import cv2
import numpy as np

from .utils import cinza, colorir, gradiente

PALETA = gradiente([
    (0, (5, 0, 20)),
    (80, (15, 15, 150)),
    (150, (25, 35, 205)),
    (210, (50, 175, 240)),
    (255, (190, 240, 255)),
])


def filtro_homem_de_ferro(img):
    g = cv2.convertScaleAbs(cinza(img), alpha=1.25, beta=-18)
    metal = colorir(cv2.merge([g, g, g]), PALETA)

    bordas = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 60, 140)
    bordas = cv2.dilate(bordas, np.ones((2, 2), np.uint8))
    dourado = np.zeros_like(metal)
    dourado[bordas > 0] = (70, 210, 255)

    raio = max(2, min(img.shape[:2]) / 80)
    brilho_dourado = cv2.GaussianBlur(dourado, (0, 0), raio)
    metal = cv2.add(metal, brilho_dourado)
    metal = cv2.add(metal, dourado)

    aura = cv2.GaussianBlur(metal, (0, 0), max(3, min(img.shape[:2]) / 35))
    return cv2.addWeighted(metal, 0.85, aura, 0.3, 0)
