import time

import cv2
import numpy as np

from .utils import pequeno


def filtro_vhs(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    desloc = max(2, w // 90)
    saida = cv2.GaussianBlur(img, (3, 3), 0)
    saida[:, :, 0] = np.roll(saida[:, :, 0], -desloc, axis=1)
    saida[:, :, 2] = np.roll(saida[:, :, 2], desloc, axis=1)

    ruido = np.random.randint(-18, 19, (h, w, 1)).astype(np.int16)
    saida = np.clip(saida.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    saida[::3, :] = (saida[::3, :] * 0.72).astype(np.uint8)

    rng = np.random.default_rng(int(time.time() * 8))
    for _ in range(2):
        y = int(rng.integers(0, max(1, h - h // 10)))
        alt = max(2, int(rng.integers(h // 40 + 1, h // 10 + 2)))
        faixa = saida[y:y + alt]
        saida[y:y + alt] = np.roll(faixa, int(rng.integers(-w // 8, w // 8 + 1)), axis=1)
    y = int(((time.time() * 0.25) % 1.0) * h)
    faixa = saida[y:y + max(2, h // 25)].astype(np.int16) + 40
    saida[y:y + max(2, h // 25)] = np.clip(faixa, 0, 255).astype(np.uint8)
    return saida
