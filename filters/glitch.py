import numpy as np


def filtro_glitch(img):
    saida = img.copy()
    h, w = img.shape[:2]
    deslocamento = max(w // 30, 2)
    saida[:, :, 0] = np.roll(img[:, :, 0], deslocamento, axis=1)
    saida[:, :, 2] = np.roll(img[:, :, 2], -deslocamento, axis=1)
    for _ in range(5):
        y = np.random.randint(0, max(1, h - 2))
        altura = np.random.randint(2, max(3, h // 8))
        movimento = np.random.randint(-max(1, w // 6), max(2, w // 6))
        saida[y:y + altura] = np.roll(
            saida[y:y + altura], movimento, axis=1
        )
    return saida
