import cv2
import numpy as np

from .utils import cinza, colorir, gradiente

PALETA = gradiente([
    (0, (35, 8, 20)),
    (110, (150, 55, 105)),
    (210, (232, 140, 195)),
    (255, (255, 205, 240)),
])


def filtro_gengar(img):
    g = cv2.medianBlur(cinza(img), 5)
    g = cv2.convertScaleAbs(g, alpha=1.15, beta=-12)
    roxo = colorir(cv2.merge([g, g, g]), PALETA)

    bordas = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 70, 150)
    bordas = cv2.dilate(bordas, np.ones((2, 2), np.uint8))
    roxo[bordas > 0] = (225, 110, 255)

    olhos = np.zeros_like(roxo)
    olhos[g > 238] = (50, 50, 255)
    brilho_olhos = cv2.GaussianBlur(olhos, (0, 0), max(2, min(img.shape[:2]) / 60))
    roxo = cv2.add(roxo, brilho_olhos)
    roxo = cv2.add(roxo, olhos)

    aura = cv2.GaussianBlur(roxo, (0, 0), max(3, min(img.shape[:2]) / 40))
    return cv2.addWeighted(roxo, 0.85, aura, 0.35, 0)
