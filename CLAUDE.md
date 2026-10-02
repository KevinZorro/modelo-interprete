# SignaCO — modelo de reconocimiento (LSC)

Clasificador de poses estáticas del abecedario de la Lengua de Señas Colombiana. MediaPipe
extrae 21 landmarks; un MLP pequeño (`.tflite`, 59 KB) clasifica 63 floats normalizados en
28 clases (27 letras + `no_es_seña`). Todo corre en el dispositivo, sin internet. Proyecto
académico UFPS, equipo de 4; la app Android (Kotlin/Compose) vive en otro repo.

## Fuera de alcance
Traducir conversaciones en tiempo real. Este repo es aprendizaje del abecedario (y, para la
feria, exploración libre de poses).

## Estructura
- `modeloparaLSC70_v8 (1).ipynb` — notebook original (entrenamiento y rechazo). Referencia.
- `evaluacion_objetivo/` — modo objetivo: umbral por letra sobre el `.tflite` ya entrenado.
- `lsc70/` — pipeline por módulos para comparar ANH vs AN y entrenar letras + números.
- `configs/` — splits por participante (fuente única) y estructura confirmada de cada sección.
- `docs/experiments.md` — memoria de experimentos. Léelo antes de proponer uno.
- `PROTOCOLO_prueba_camara.md` — cómo hacer una prueba con cámara que aguante revisión.

## Comandos
```bash
pip install -r requirements.txt            # evaluacion_objetivo (sin TensorFlow)
python -m evaluacion_objetivo.pruebas
pip install -r requirements-lsc70.txt      # lsc70 (TensorFlow, MediaPipe)
pytest tests/ -q                            # -m "not slow" salta el entrenamiento
ruff check lsc70 tests --select E,F,W --ignore E501
```
MediaPipe necesita `libEGL.so.1` del sistema (en un contenedor: `apt-get install libegl1 libgles2`).
`LSC70_MODELO=ruta/hand_landmarker.task` activa la prueba con MediaPipe real.

## Reglas que no se rompen
- **La normalización de landmarks no se toca** y vive en `lsc70/normalizacion.py` como copia
  TEXTUAL del notebook (muñeca al origen, escala = norma máxima, dividir, espejar X si es
  izquierda, aplanar). Si difiere en el orden el modelo devuelve basura SIN lanzar error.
  Una prueba falla si el texto se desvía del notebook.
- **Split siempre por participante** (nunca por frame ni toma), en los tres niveles. Los
  splits salen de `configs/splits_participantes.json`; ANH y AN son las mismas 70 personas.
- La unidad de evaluación es la **toma** (participante + clase, ~6 frames promediados).
- Las imágenes nunca se suben al repo (datos personales, Ley 1581). Solo landmarks.
- Los resultados de dataset son hipótesis hasta probarlos con cámara y con protocolo.

## Hechos que cuesta redescubrir
- **88.8 % ya es top-1 entre las 27 clases**, por toma, persona-independiente (OOF 5 folds).
- El descarte por no-detección en ANH es **4.6 % global** (train 5.2 %, test 2.8 %): el 2.8 %
  que se cita es solo el test. Muy desigual: Z 25 %, H 24 %, Ñ 14 % en train.
- **ANH ya contiene los números** (`1,4,5,6,7,8,9,10,MIL,MILLON`; faltan 2 y 3). El notebook
  los mandaba a `none` a propósito. No hace falta AN para tenerlos.
- La Ñ estática es la N (co-activación 0.30/0.31): sin movimiento no se distinguen. Z y Ñ
  son dinámicas; un modelo de poses no las enseña.
- La prueba con cámara que excluyó C, E, H, K, M, N fue **inválida** (señas mal ejecutadas).
  Falló la prueba, no el modelo: en dataset C 0.94, K 0.90, M 0.96. Las flojas son H 0.61, E 0.67.
- Los negativos de HaGRID `two_up`, `palm`, `three2`, `one` son las letras R, L, L, S.
- Con n = 16 tomas por letra (held-out único) casi nada se decide; usar OOF (n = 70).
- AN (escena completa) NO tiene más píxeles de mano que ANH (63 vs 66 px, razón pareada 1.01) y
  rinde ~16 puntos peor (71.9 % vs 87.6 %). Con números (35 clases: 27 letras + 8 números) ANH da 87.6 %; los números
  cuestan 1.6 puntos. Pares equivalentes: N/Ñ y 1/6. En AN MediaPipe ve dos manos en 68.7 %.
- Exportar a .tflite SIN cuantizar: la cuantización dinámica rompía la paridad con Keras (~1 % de argmax distinto, casi-empates) y no hace falta (0.2 MB de 20). La mano en reposo es lo que más se acepta como seña (17.8 %): subir el umbral no lo arregla barato.
