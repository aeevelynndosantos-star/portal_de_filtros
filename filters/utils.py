import cv2
import numpy as np


def cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pequeno(img, minimo=24):
    return min(img.shape[:2]) < minimo


def brilho(camada, raio, forca=1.0):
    borrada = cv2.GaussianBlur(camada, (0, 0), max(0.5, raio))
    return cv2.addWeighted(camada, 1.0, borrada, forca, 0)


def gradiente(paradas):
    posicoes = [p for p, _ in paradas]
    cores = np.array([c for _, c in paradas], dtype=np.float32)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for k in range(3):
        lut[:, 0, k] = np.interp(np.arange(256), posicoes, cores[:, k])
    return lut


def colorir(img, lut):
    return cv2.applyColorMap(cinza(img), lut)


def ponto_polar(cx, cy, raio, graus):
    a = np.deg2rad(graus)
    return int(cx + raio * np.cos(a)), int(cy + raio * np.sin(a))


_padroes = {}


def padrao_pontos(h, w, passo):
    chave = (h, w, passo)
    if chave not in _padroes:
        if len(_padroes) > 8:
            _padroes.clear()
        yy, xx = np.indices((h, w))
        k = 2 * np.pi / passo
        _padroes[chave] = ((np.cos(xx * k) + np.cos(yy * k)) / 2).astype(np.float32)
    return _padroes[chave]
