import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from filters import FILTROS

LARGURA, ALTURA = 960, 540
MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODELO = Path(__file__).resolve().parent / "hand_landmarker.task"

PULSO = 0
POLEGAR, INDICADOR, MEDIO, ANELAR, MINDINHO = 4, 8, 12, 16, 20
PONTAS = (POLEGAR, INDICADOR, MEDIO, ANELAR, MINDINHO)

LIGACOES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
def garantir_modelo():
    if MODELO.exists() and MODELO.stat().st_size > 1_000_000:
        return

    print("Baixando o modelo de mãos (aprox. 7 MB)...")
    temporario = MODELO.with_suffix(".baixando")
    try:
        urllib.request.urlretrieve(MODELO_URL, temporario)
        if temporario.stat().st_size < 1_000_000:
            raise RuntimeError("download incompleto")
        temporario.replace(MODELO)
    except Exception as erro:
        if temporario.exists():
            temporario.unlink()
        raise RuntimeError(
            "Não consegui baixar o modelo de mãos. Verifique sua internet.\n"
            f"Detalhe: {erro}"
        )


def abrir_camera():
    backends = (
        [cv2.CAP_DSHOW, cv2.CAP_ANY]
        if sys.platform.startswith("win")
        else [cv2.CAP_ANY]
    )
    for indice in (0, 1, 2):
        for backend in backends:
            camera = cv2.VideoCapture(indice, backend)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = camera.read()
                if ok and frame is not None:
                    return camera
            camera.release()
    raise RuntimeError(
        "Não consegui abrir a câmera. Feche outros programas que possam "
        "estar usando-a e confira a permissão de câmera do Windows."
    )

def ponto(lm, indice, w, h):
    return np.array([lm[indice].x * w, lm[indice].y * h], dtype=np.float32)


def distancia(a, b):
    return float(np.linalg.norm(a - b))


def polegar_no_mindinho(lm, w, h):
    palma = distancia(ponto(lm, 0, w, h), ponto(lm, 9, w, h))
    if palma < 1:
        return False
    return distancia(
        ponto(lm, POLEGAR, w, h), ponto(lm, MINDINHO, w, h)
    ) < 0.38 * palma

def desenhar_mao(saida, lm, w, h):
    pts = [
        (int(p.x * w), int(p.y * h)) for p in lm
    ]

    # Linhas brancas finas
    for a, b in LIGACOES:
        cv2.line(
            saida,
            pts[a],
            pts[b],
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Pontinhos brancos pequenos
    for i, p in enumerate(pts):
        raio = 3 if i in PONTAS else 2
        cv2.circle(
            saida,
            p,
            raio,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


def pontos_inteiros(pontos):
    pontos = np.asarray(pontos, dtype=np.float64)
    pontos = np.nan_to_num(pontos, nan=0, posinf=1e5, neginf=-1e5)
    return np.round(np.clip(pontos, -20000, 20000)).astype(np.int32)


def cruzamento(a, b, c, d):
    r, s = b - a, d - c
    denominador = r[0] * s[1] - r[1] * s[0]
    if abs(denominador) < 1e-6:
        return None
    delta = c - a
    t = (delta[0] * s[1] - delta[1] * s[0]) / denominador
    u = (delta[0] * r[1] - delta[1] * r[0]) / denominador
    if 0 < t < 1 and 0 < u < 1:
        return a + t * r
    return None


def separar_area(pontos):
    """Se a moldura 2D cruzar, divide em duas áreas em vez de criar um X."""
    a, b, c, d = pontos
    x = cruzamento(b, c, d, a)
    if x is not None:
        return [np.array([x, a, b]), np.array([x, c, d])]
    x = cruzamento(a, b, c, d)
    if x is not None:
        return [np.array([x, b, c]), np.array([x, d, a])]
    return [np.array(pontos)]


def aplicar_filtro(saida, original, poligono, funcao_filtro):
    h, w = original.shape[:2]
    poligono = pontos_inteiros(poligono)
    x0, y0 = np.maximum(poligono.min(axis=0), (0, 0))
    x1, y1 = np.minimum(poligono.max(axis=0) + 1, (w, h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    recorte = np.ascontiguousarray(original[y0:y1, x0:x1])
    filtrado = funcao_filtro(recorte)
    mascara = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local = poligono - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(mascara, [local], 255)

    regiao = saida[y0:y1, x0:x1]
    regiao[mascara > 0] = filtrado[mascara > 0]

def main():
    garantir_modelo()
    camera = abrir_camera()

    opcoes = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELO)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(opcoes)

    indice_filtro = 0
    modo_3d = False
    mostrar_maos = True
    pontos_suaves = None
    gesto_filtro_ativo = False
    inicio = time.time()
    ultimo_timestamp = -1

    try:
        cv2.namedWindow("Portal de filtros", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Não consegui receber imagem da câmera.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            saida = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagem = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )
            timestamp = max(
                int((time.time() - inicio) * 1000), ultimo_timestamp + 1
            )
            ultimo_timestamp = timestamp

            resultado = detector.detect_for_video(imagem, timestamp)
            maos = resultado.hand_landmarks

            filtro_gesto_agora = any(
                polegar_no_mindinho(mao, w, h) for mao in maos
            )

            if filtro_gesto_agora and not gesto_filtro_ativo:
                indice_filtro = (indice_filtro + 1) % len(FILTROS)

            gesto_filtro_ativo = filtro_gesto_agora

            if len(maos) == 2:
                esquerda, direita = sorted(maos, key=lambda m: m[0].x)

                alvos = np.array(
                    [ponto(m, i, w, h) for m in (esquerda, direita) for i in PONTAS],
                    dtype=np.float32
                )

                if pontos_suaves is None:
                    pontos_suaves = alvos
                else:
                    pontos_suaves = 0.65 * alvos + 0.35 * pontos_suaves

                if modo_3d:
                    contorno = cv2.convexHull(
                        pontos_inteiros(pontos_suaves)
                    )
                    areas = [contorno.reshape(-1, 2)]
                else:
                    polegar_esq = pontos_suaves[0]
                    indicador_esq = pontos_suaves[1]
                    polegar_dir = pontos_suaves[5]
                    indicador_dir = pontos_suaves[6]

                    moldura = np.array(
                        [polegar_esq, indicador_esq,
                         indicador_dir, polegar_dir],
                        dtype=np.float32
                    )

                    contorno = pontos_inteiros(moldura)
                    areas = separar_area(moldura)

                for numero, area in enumerate(areas):
                    filtro = FILTROS[(indice_filtro + numero) % len(FILTROS)]
                    aplicar_filtro(saida, frame, area, filtro)

                cv2.polylines(
                    saida,
                    [pontos_inteiros(contorno).reshape(-1, 1, 2)],
                    True,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
            else:
                pontos_suaves = None

            if mostrar_maos:
                for mao in maos:
                    desenhar_mao(saida, mao, w, h)

            cv2.imshow("Portal de filtros", saida)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), 27):
                break
            elif tecla == ord("n"):
                indice_filtro = (indice_filtro + 1) % len(FILTROS)
            elif tecla == ord("p"):
                indice_filtro = (indice_filtro - 1) % len(FILTROS)
            elif tecla == ord("c"):
                modo_3d = not modo_3d
            elif tecla == ord("h"):
                mostrar_maos = not mostrar_maos

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\nERRO: {erro}")
        input("\nPressione ENTER para fechar...")