import time

import cv2
import numpy as np

from .utils import cinza, colorir, gradiente, padrao_pontos, pequeno

PALETA = gradiente([
    (0, (45, 0, 25)),
    (90, (255, 0, 255)),
    (180, (255, 230, 0)),
    (255, (255, 255, 255)),
])


def filtro_aranhaverso(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    m = min(w, h)

    g = cv2.convertScaleAbs(cinza(img), alpha=1.25, beta=-15)
    colorida = colorir(cv2.merge([g, g, g]), PALETA)

    passo = max(6, m // 45)
    normal = g.astype(np.float32) / 255.0
    pontos = (padrao_pontos(h, w, passo) < (normal * 2 - 1)).astype(np.float32)
    colorida = colorida.astype(np.float32) * (0.5 + 0.5 * pontos)[:, :, None]
    colorida = np.clip(colorida, 0, 255).astype(np.uint8)

    desloc = max(2, w // 90) + int(2 * np.sin(time.time() * 18))
    saida = colorida.copy()
    saida[:, :, 2] = np.roll(colorida[:, :, 2], desloc, axis=1)
    saida[:, :, 0] = np.roll(colorida[:, :, 0], -desloc, axis=1)
    return saida
