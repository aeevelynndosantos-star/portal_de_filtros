import time

from .pixel_art import (
    brilhos_animados, desenhar_sprite, pixelar
)
from .utils import pequeno

BRILHOS = (
    (0.12, 0.14), (0.86, 0.18), (0.18, 0.82), (0.84, 0.86),
    (0.50, 0.07), (0.06, 0.50), (0.94, 0.52), (0.52, 0.93),
)
ITENS = (
    (0.06, 0.20, "coracao"), (0.84, 0.22, "diamante"),
    (0.08, 0.70, "espada"), (0.82, 0.70, "maca"),
)


def filtro_minecraft(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    m = min(w, h)
    bloco = max(6, m // 22)
    saida = pixelar(img, bloco)
    brilhos_animados(saida, time.time(), BRILHOS, m * 0.05)
    celula = max(2, m // 40)
    for px, py, nome in ITENS:
        desenhar_sprite(saida, nome, int(px * w), int(py * h), celula)
    return saida
