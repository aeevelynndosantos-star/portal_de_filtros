import cv2

from .utils import cinza


def filtro_esboco(img):
    g = cinza(img)
    invertida = 255 - g
    borrada = cv2.GaussianBlur(invertida, (21, 21), 0)
    resultado = cv2.divide(g, 255 - borrada, scale=256)
    return cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
