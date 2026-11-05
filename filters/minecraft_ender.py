import time

from .pixel_art import particulas, pixelar
from .utils import gradiente, pequeno

PALETA = gradiente([
    (0, (10, 0, 10)),
    (100, (110, 20, 90)),
    (190, (230, 70, 190)),
    (255, (255, 200, 250)),
])
PARTICULAS = ((255, 90, 200), (200, 40, 150), (255, 160, 230))


def filtro_minecraft_ender(img):
    if pequeno(img):
        return img
    h, w = img.shape[:2]
    m = min(w, h)
    bloco = max(6, m // 22)
    saida = pixelar(img, bloco, PALETA, 6)
    particulas(saida, time.time(), PARTICULAS, 30, bloco, 0.08, sobe=False, semente=9)
    return saida
