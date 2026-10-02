# Experimentos

Memoria del proyecto: una entrada por experimento, la más reciente arriba. Incluye los
resultados negativos. Léelo antes de proponer un experimento para no repetir.

## 2026-10-02 · Modelo final 37 clases + `no_es_seña` (ANH, 600 negativos limpios)
- Hipótesis: un modelo con números y clase de rechazo se puede exportar y desplegar.
- Cambio: `lsc70.exportar_final` (5 folds por participante para medir el rechazo, entrenamiento
  final con los 70, exportación a .tflite). Negativos de HaGRID sin `one/palm/three2/two_up`.
- Resultado (Colab, 200 épocas, argmax de 38 clases, sin umbral): falso rechazo 0.6 %,
  **falsa aceptación 14.7 % por frame**, top-1 por toma contando el rechazo como fallo 86.3 %
  (sin rechazo era 87.6 %). El .tflite cuantizado NO coincidía con Keras: argmax 98.4 %, diff
  máx 0.12. Pesaba 0.060 MB.
- Causa de la paridad: la cuantización dinámica. Mismo modelo exportado en float32: diff 1e-6 y
  0 discrepancias; cuantizado: 0.75 % discrepa, todos casi-empates (margen 0.01). Sin cuantizar
  pesa 0.2 MB (7.66 MB con el detector, límite 20). Ahora float32 por defecto.
- Corrida corta de diagnóstico (SOLO letras, 25 épocas; no es resultado): subir el umbral baja
  la falsa aceptación pero cuesta mucho rechazo: umbral 0.7 -> FA 4.2 %, rechazo de señas
  buenas 28.8 %; 0.9 -> FA 1.5 %, rechazo 46.8 %. Por clase de negativo, la mano en reposo
  (`no_gesture`) es la que más se acepta como seña: 17.8 %.
- Conclusión: el umbral por confianza no es una buena palanca (el modelo no separa bien una mano
  relajada de un puño: A, S, E, M). Pendiente: repetir con los 37 clases completos, y medir si
  exigir k frames consecutivos ayuda; si no, para la feria habrá que pedir sostener la seña.
- Límites: los negativos no tienen id de persona (la falsa aceptación no es independiente por
  persona); sin prueba con cámara; sin umbrales calibrados.

## 2026-10-01 · ANH vs AN: letras + números, 37 clases (hipótesis de resolución refutada)
- Hipótesis: AN (escena completa 640x480) da más detalle de mano que ANH (recorte 120x120) y
  recupera C, K, M; además añade los números.
- Cambio respecto al baseline: sección de datos (ANH, AN, combinado) y clases (27 letras +
  `1,4,5,6,7,8,9,10`; sin MIL/MILLON, probablemente dinámicas, sin verificar).
- Config: red del notebook v8 · 5 folds por participante · 3 semillas (0, 42, 123) · 70
  participantes · commit `049eec6` (resultados en Drive `resultados_colab/`).
- Tamaño de la mano (`medir_mano`, 560 imágenes pareadas): lado mediano 66 px en ANH y 63 px
  en AN; razón pareada AN/ANH = 1.01 (AN mayor en 58 %). En AN la mano ocupa 0.75 % de la
  escena. **No hay más píxeles útiles en AN.** Detección 95.1 % (ANH) vs 99.3 % (AN).
- Resultado (top-1 estricto por toma, ensamble de semillas):

  | Experimento | Top-1 | Extremo a extremo | Con equivalencias |
  |---|---|---|---|
  | ANH, 27 letras | 89.2 % | 88.7 % | 90.5 % |
  | ANH, letras + números (37) | **87.6 %** | 87.1 % | 89.3 % |
  | AN, letras + números | 71.9 % | 71.6 % | 72.7 % |
  | ANH+AN, evaluado en AN | 73.2 % | 72.9 % | 73.8 % |

  Añadir números cuesta 1.6 puntos (letras 87.6 %, números 87.3 %). Equivalencias medidas:
  N/Ñ y **1/6** (co-activación ≈ 0.27/0.28). Recall con ANH / con AN: C 0.93/0.76,
  K 0.93/0.76, M 0.99/0.89, E 0.71/0.62, H 0.67/0.49. Las flojas siguen siendo H y E, no C/K/M.
- Conclusión: AN no mejora nada y empeora ~16 puntos; combinar no lo arregla (+1.3). Con ANH
  se queda. La causa NO es la resolución. Sospecha: MediaPipe ve dos manos en 68.7 % de AN
  (2.7 % en ANH) y la extracción usa `num_hands=1`, por lo que puede elegir la mano que no
  firma. **Sin comprobar**: ver `lsc70.diagnostico_mano`.
- Límites: no se corrió `comparar` (bootstrap pareado); sin verificación con cámara; el
  contraste del pipeline contra el cache viejo no se revisó.

## Pendiente · ¿AN elige la mano equivocada?
- Hipótesis: con dos manos en escena, `num_hands=1` se queda con la que no hace la seña.
- Cómo: `python -m lsc70.diagnostico_mano --anh ... --an ... --modelo ...` (usa el vector de ANH
  como referencia de la mano que firma; el umbral sale del ruido en imágenes de AN con una
  sola mano). Si da una tasa alta de mano equivocada, arreglar la selección de mano y repetir AN.

## 2026-10-01 · Control: baseline ANH (27 letras) con el pipeline `lsc70`
- Hipótesis: el pipeline nuevo reproduce el 88.8 % del notebook v8.
- Cambio respecto al baseline: ninguno (es el control).
- Config: red e hiperparámetros del notebook v8 · 5 folds por participante
  (`configs/splits_participantes.json`) · datos: `landmarks_cache.npz` (sha256 `22a4b753…`,
  solo letras) · commit: `e6189b3` · semilla: 42 (una sola: es una validación del código).
- Resultado: top-1 estricto por toma **88.7 %** (1 877 tomas, 70 participantes) vs 88.8 %.
  Con equivalencias 89.9 % (único par: N/Ñ). Recall: H 0.61, Ñ 0.65, E 0.67 (las flojas);
  C 0.94, K 0.90, M 0.96 (en la media o por encima).
- Conclusión: reproduce. **El 88.8 % ya es top-1 entre todas las clases (27), por toma y
  persona-independiente**; no es una métrica "fácil" que vaya a bajar al pasar a modo
  exploración. Lo que lo bajará es añadir clases cuya pose coincide con otra.

## 2026-09 · Modo objetivo ("¿es esta la letra pedida?") sobre el modelo desplegado
- Hipótesis: con la letra pedida conocida, la decisión binaria recupera letras excluidas.
- Cambio: post-proceso de probabilidades (`evaluacion_objetivo/`), sin reentrenar. Umbral
  por letra con leave-one-participant-out sobre 16 participantes held-out (n = 16 por letra).
- Resultado: viables con IC95 completo (dataset, **no verificadas con cámara**): E, G, I, S.
  C, K, M, N, H quedan indeterminadas por tamaño de muestra, no por fallo. R, en el
  vocabulario activo, es la más floja (umbral 0.44, falso rechazo 37.5 %).
- Negativos de HaGRID: 4 de las 8 clases difíciles eran letras (`two_up`=R, `palm` y
  `three2`=L, `one`=S) e inflaban los umbrales; excluidas.
- Conclusión: la prueba con cámara que excluyó C, E, H, K, M, N resultó inválida (se
  ejecutaban mal las señas). Falta una prueba con protocolo (`PROTOCOLO_prueba_camara.md`).
