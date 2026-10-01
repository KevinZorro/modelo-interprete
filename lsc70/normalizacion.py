"""Normalización de landmarks y preprocesado de imagen, copiados del notebook v8.

`vector_features` es COPIA TEXTUAL de la celda 17 del notebook (el orden importa: si
difiere, el modelo devuelve basura SIN lanzar error). `tests/test_lsc70.py` compara
este archivo contra el notebook y falla si alguna vez se desvían.

Lo único que no viene del notebook es el pegamento: crear el detector, decodificar
imágenes en memoria y el preprocesado de ANH en versión "en memoria" (el notebook
escribía el lienzo a disco como .jpg y lo releía; aquí se emula esa ida y vuelta).
"""
import cv2
import numpy as np

# --- Constantes de la celda de configuración del notebook (celda 2) ---------------
MIN_CONF_DETECCION = 0.30
NORMALIZAR_MANO_IZQUIERDA = True    # vector_features la lee como global
TAM_SALIDA = 480                    # lado del lienzo con padding (solo ANH)
FACTOR_MARGEN = 0.60                # proporción del lienzo que ocupa la imagen (solo ANH)


# --- COPIA TEXTUAL de la celda 17 del notebook (no editar a mano) -----------------
def vector_features(landmarks, es_izquierda):
    """21 landmarks -> (63 features normalizadas, 3 valores crudos).

    La normalizacion (muneca al origen, escala unitaria) hace la forma invariante
    a donde este la mano y a que tamano tenga. Eso es lo correcto para clasificar
    una POSE, pero borra por completo el movimiento de la mano entre frames.

    La Ne es la N mas un movimiento de muneca. Si solo guardamos la parte
    normalizada, esa distincion es fisicamente irrecuperable. Por eso devolvemos
    tambien la posicion absoluta de la muneca y la escala: son exactamente
    las dos cosas que la normalizacion elimina.
    """
    pts = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    muneca_x, muneca_y = float(pts[0, 0]), float(pts[0, 1])   # ANTES de normalizar
    pts -= pts[0]
    escala = float(np.linalg.norm(pts, axis=1).max())
    if escala < 1e-6:
        return None, None
    pts /= escala
    if NORMALIZAR_MANO_IZQUIERDA and es_izquierda:
        pts[:, 0] *= -1              # espejo en X para unificar manos
        muneca_x = 1.0 - muneca_x    # el mismo espejo sobre la coordenada cruda
    return pts.flatten(), np.array([muneca_x, muneca_y, escala], dtype=np.float32)


def preparar_imagen_anh(img_bgr, ida_y_vuelta_jpg=True):
    """Preprocesado de ANH (celda 10): reescala con padding gris sobre un lienzo de 480.

    Devuelve (lienzo, (escala, x0, y0)); la tupla sirve para volver de coordenadas del
    lienzo a píxeles nativos: nativo = (px_lienzo - offset) / escala.

    El notebook guardaba el lienzo como .jpg y MediaPipe lo releía, o sea que hubo
    compresión JPEG antes de detectar. `ida_y_vuelta_jpg=True` la reproduce para que los
    landmarks sean comparables con los del cache; ponerlo en False sería una variante
    distinta del pipeline.
    """
    h, w = img_bgr.shape[:2]
    escala = (TAM_SALIDA * FACTOR_MARGEN) / max(h, w)
    nueva = cv2.resize(img_bgr, (int(w * escala), int(h * escala)),
                       interpolation=cv2.INTER_CUBIC)
    nh, nw = nueva.shape[:2]
    # Fondo gris neutro (misma decisión que el notebook): replicar el borde metería
    # texturas falsas que el detector podría tomar por estructura de mano.
    lienzo = np.full((TAM_SALIDA, TAM_SALIDA, 3), 128, dtype=np.uint8)
    y0, x0 = (TAM_SALIDA - nh) // 2, (TAM_SALIDA - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = nueva
    if ida_y_vuelta_jpg:
        ok, buf = cv2.imencode(".jpg", lienzo)          # calidad por defecto, como imwrite
        lienzo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return lienzo, (escala, x0, y0)


def preparar_imagen_nativa(img_bgr):
    """Sin preprocesado: la escena completa tal cual (LSC70AN).

    NO se reutiliza `preparar_imagen_anh` en AN a propósito: con una escena de 640x480
    reescalaría a 288x216 (factor 0.45), encogiendo la mano más de la mitad y sesgando
    el experimento en contra de AN antes de empezar.
    """
    return img_bgr, (1.0, 0, 0)


PREPARADORES = {"anh": preparar_imagen_anh, "nativo": preparar_imagen_nativa}


def crear_detector(ruta_modelo, num_hands=1):
    """Detector con las opciones de la celda 17. `num_hands=1` es lo que usa el
    entrenamiento; subirlo solo tiene sentido para medir (cuántas veces hay 2 manos)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta_modelo)),
            num_hands=num_hands,
            min_hand_detection_confidence=MIN_CONF_DETECCION,
            running_mode=vision.RunningMode.IMAGE,
        )
    )


def detectar(detector, img_bgr):
    """Corre MediaPipe sobre una imagen BGR en memoria. Devuelve el resultado crudo.

    El notebook leía con `mp.Image.create_from_file` (RGB); OpenCV decodifica en BGR,
    así que se convierte antes de construir la imagen de MediaPipe.
    """
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def es_mano_izquierda(res):
    """Misma línea que la celda 19 del notebook."""
    return bool(res.handedness) and res.handedness[0][0].category_name == "Left"


def landmarks_de_imagen(detector, img_bgr):
    """(vector 63, extras 3, resultado crudo, es_izquierda) o None si no hay mano.

    Es la composición exacta de la celda 19: detectar -> vector_features. `None`
    cubre tanto "MediaPipe no vio mano" como escala degenerada.
    """
    res = detectar(detector, img_bgr)
    if not res.hand_landmarks:
        return None
    es_izq = es_mano_izquierda(res)
    v, extras = vector_features(res.hand_landmarks[0], es_izq)
    if v is None:
        return None
    return v, extras, res, es_izq
