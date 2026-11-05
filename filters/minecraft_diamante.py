import time

from .pixel_art import brilhos_animados, desenhar_sprite, pixelar
from .utils import gradiente, pequeno

PALETA = gradiente([
    (0, (40, 25, 0)),
    (110, (170, 140, 15)),
    (200, (240, 235, 80)),
    (255, (255, 255, 235)),
])
BRILHOS = ((0.10, 0.12), (0.88, 0.16), (0.14, 0.86), (0.86, 0.84), (0.50, 0.06))
DIAMANTES = ((0.07, 0.30), (0.82, 0.45), (0.15, 0.62))


def filtro_minecraft_diamante(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    m = min(w, h)
    bloco = max(6, m // 22)
    saida = pixelar(img, bloco, PALETA, 6)
    brilhos_animados(saida, time.time(), BRILHOS, m * 0.05)
    celula = max(2, m // 38)
    for px, py in DIAMANTES:
        desenhar_sprite(saida, "diamante", int(px * w), int(py * h), celula)
    return saida
