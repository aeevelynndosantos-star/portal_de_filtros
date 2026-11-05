import cv2
import numpy as np

from .utils import cinza

SPRITES = {
    "coracao": (
        (
            ".XX.XX.",
            "XHXXXXX",
            "XXXXXXD",
            ".XXXXD.",
            "..XXD..",
            "...D...",
        ),
        {"X": (55, 55, 225), "H": (190, 190, 255), "D": (30, 30, 150)},
    ),
    "diamante": (
        (
            "..DDD..",
            ".DCHCD.",
            "DCHCCCD",
            "DCCCCCD",
            ".DCCCD.",
            "..DCD..",
            "...D...",
        ),
        {"D": (150, 125, 20), "C": (235, 230, 70), "H": (255, 255, 225)},
    ),
    "maca": (
        (
            "....G..",
            "...GL..",
            ".YYYYY.",
            "YHYYYYD",
            "YYYYYYD",
            ".YYYYD.",
            "..YYD..",
        ),
        {"Y": (45, 205, 250), "H": (190, 245, 255), "D": (20, 140, 200),
         "G": (30, 60, 110), "L": (60, 170, 60)},
    ),
    "espada": (
        (
            "......CW",
            ".....CCW",
            "....CCW.",
            "...CCW..",
            ".B.CW...",
            "..BB....",
            ".BBB....",
            "BB.B....",
        ),
        {"C": (235, 230, 70), "W": (255, 255, 225), "B": (40, 70, 120)},
    ),
}


def desenhar_sprite(img, nome, x, y, celula):
    linhas, paleta = SPRITES[nome]
    borda = max(1, celula // 2)
    for j, linha in enumerate(linhas):
        for i, c in enumerate(linha):
            if c != ".":
                x0, y0 = x + i * celula, y + j * celula
                cv2.rectangle(img, (x0 - borda, y0 - borda),
                              (x0 + celula + borda - 1, y0 + celula + borda - 1),
                              (255, 255, 255), -1)
    for j, linha in enumerate(linhas):
        for i, c in enumerate(linha):
            if c != ".":
                x0, y0 = x + i * celula, y + j * celula
                cv2.rectangle(img, (x0, y0),
                              (x0 + celula - 1, y0 + celula - 1), paleta[c], -1)


def desenhar_brilho(img, x, y, tamanho):
    s = max(2, tamanho)
    f = max(1, int(s * 0.22))
    pontos = np.array([
        (x, y - s), (x + f, y - f), (x + s, y), (x + f, y + f),
        (x, y + s), (x - f, y + f), (x - s, y), (x - f, y - f),
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, pontos, (255, 255, 255), cv2.LINE_AA)


def pixelar(img, bloco, lut=None, niveis=6):
    h, w = img.shape[:2]
    pequena = cv2.resize(
        img, (max(1, w // bloco), max(1, h // bloco)),
        interpolation=cv2.INTER_AREA
    )
    if lut is None:
        pequena = (pequena // 48) * 48 + 24
        hsv = cv2.cvtColor(pequena, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.int16) * 1.25, 0, 255
        ).astype(np.uint8)
        pequena = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        g = cv2.convertScaleAbs(cinza(pequena), alpha=1.2, beta=-15)
        passo = 256 // niveis
        g = (g // passo) * passo + passo // 2
        pequena = cv2.applyColorMap(g.astype(np.uint8), lut)
    saida = cv2.resize(pequena, (w, h), interpolation=cv2.INTER_NEAREST)
    saida[::bloco, :] = (saida[::bloco, :] * 0.85).astype(np.uint8)
    saida[:, ::bloco] = (saida[:, ::bloco] * 0.85).astype(np.uint8)
    return saida


def particulas(img, t, cores, quantidade, bloco, velocidade, sobe=True, semente=5):
    h, w = img.shape[:2]
    rng = np.random.default_rng(semente)
    xs = rng.random(quantidade)
    fases = rng.random(quantidade)
    tamanhos = rng.integers(1, 3, quantidade)
    for i in range(quantidade):
        prog = (t * velocidade + fases[i]) % 1.0
        y = int((1 - prog) * h) if sobe else int(prog * h)
        x = int(xs[i] * w + np.sin(t * 2 + i) * bloco * 0.5)
        lado = max(2, int(bloco * 0.5 * tamanhos[i]))
        cv2.rectangle(img, (x, y), (x + lado, y + lado), cores[i % len(cores)], -1)


def brilhos_animados(img, t, posicoes, escala):
    h, w = img.shape[:2]
    for n, (px, py) in enumerate(posicoes):
        pulso = 0.55 + 0.45 * np.sin(t * 4 + n * 1.7)
        desenhar_brilho(img, int(px * w), int(py * h), int(escala * pulso) + 2)
