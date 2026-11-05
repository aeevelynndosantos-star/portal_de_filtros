import time

from .pixel_art import particulas, pixelar
from .utils import gradiente, pequeno

PALETA = gradiente([
    (0, (5, 0, 20)),
    (90, (15, 15, 120)),
    (170, (20, 70, 220)),
    (255, (60, 200, 255)),
])
BRASAS = ((20, 130, 255), (40, 200, 255), (20, 70, 230))


def filtro_minecraft_nether(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    m = min(w, h)
    bloco = max(6, m // 22)
    saida = pixelar(img, bloco, PALETA, 6)
    particulas(saida, time.time(), BRASAS, 26, bloco, 0.18, sobe=True, semente=5)
    return saida
