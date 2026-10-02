# Experimentos

Memoria del proyecto: una entrada por experimento, la más reciente arriba. Incluye los
resultados negativos. Léelo antes de proponer un experimento para no repetir.

## 2026-10-03 · Antes vs ahora, separando negativos y números (chequeo con los mismos folds)
- Hipótesis: el modelo actual cambió respecto al desplegado por dos causas: 5x negativos de reposo y
  8 números. ¿Cuánto aporta cada una?
- Diseño: tres corridas de `exportar_final --solo-chequeo` (5 folds por participante, semilla 42,
  200 épocas) comparadas con `comparar_rechazo` (tomas pareadas, bootstrap por persona).
  No se evaluó el .tflite desplegado: se entrenó con 55 personas y el actual con 70.
  A = 27 letras + 600 negativos (antes) · D = 27 letras + 2200 · C = 35 clases + 2200 (ahora).
  Corrección: las corridas anteriores hablaban de "37 clases"; eran 35 (27 + 8 números; MIL y
  MILLON excluidos, 2 y 3 no existen).
- Resultado (argmax, sin umbral):

  | | Falso rechazo | Falsa aceptación | Top-1 con rechazo |
  |---|---|---|---|
  | A: 27 letras, 600 neg. | 0.9 % | 13.7 % | 88.4 % |
  | D: 27 letras, 2200 neg. | 1.7 % | 5.1 % | 88.7 % |
  | C: 35 clases, 2200 neg. | 1.6 % | 5.8 % | 85.9 % |

  Negativos (A->D): top-1 +0.2 [-0.9, +1.2] sin diferencia clara; falso rechazo +0.8 [+0.4, +1.3];
  la falsa aceptación baja de 13.7 % a 5.1 % y `no_gesture` de 18.0 % a 5.1-6.2 %.
  Números (D->C): top-1 **-2.8 [-3.8, -1.7]** (diferencia clara); falso rechazo +0.3 [-0.2, +0.7].
  Total (A->C): top-1 -2.6 [-3.7, -1.4].
- Dónde se pierde con los números (D->C): letras que ya no se aciertan van a `C` (10), `no_es_seña`
  (9), `10` (8), `4` (7), `S` (6), `N` (5). Por letra: E -16, B -13, H -11, Z -9, W -6, S -6.
  B -13 con 9 tomas perdidas y el `4` absorbiendo 7 sugiere que el 4 es la misma pose que B
  (HaGRID `four` ya salía como B); falta confirmar con el destino por clase.
- Conclusión: los negativos son una mejora clara y casi gratis; los números cuestan ~3 puntos de
  top-1, concentrados en letras que comparten pose con un número. Ruido de referencia: dos
  semillas difieren ±7-10 puntos por letra, así que solo cuentan los patrones y el top-1 total.
- Límites: una semilla; negativos de HaGRID sin id de persona (falsa aceptación optimista);
  sin cámara.

## 2026-10-02 · Más negativos de reposo: HaGRID `no_gesture` x5 (400 -> 2000)
- Hipótesis: la mano en reposo se acepta como seña (19.8 %) por falta de ejemplos negativos.
- Cambio respecto al modelo anterior: negativos `no_gesture_v2` (2000, misma fuente HaGRID, extraídos
  con `lsc70.extraer_negativos`) + los 200 gestos limpios; se excluye el `no_gesture` viejo para
  no duplicar. Negativos 2200 (≈5.5x una clase media). Descarte por no-detección 14.7 %
  (2344 leídas, 2000 con mano; el límite fue el tope, la fuente NO se agotó).
- Config: mismo modelo y datos de señas (35 clases, 14115 frames), float32, 200 épocas,
  `modelo_final_v2` en Drive, commit `0de3715`.
- Resultado (argmax, sin umbral): falso rechazo 1.6 % (antes 0.6 %), **falsa aceptación 5.8 %**
  (antes 14.7 %), top-1 con rechazo 85.9 % (antes 86.3 %). Por clase de negativo: `no_gesture`
  6.2 % (antes 19.8 %), `dislike` 4.0 % (10.0 %), `peace_inverted` 2.0 %, `stop_inverted` 2.0 %,
  `two_up_inverted` 0.0 %. Paridad float32 100 %, 0.208 MB.
  Con umbral el rechazo de señas buenas sigue siendo caro (0.5 -> 14.6 %; 0.7 -> 36.4 %).
- Conclusión: más negativos sí ayuda: la falsa aceptación baja ~60 % y cuesta ~1 punto de falso
  rechazo. Mejor punto de operación: argmax sin umbral.
- Límites (importantes): los negativos de HaGRID no tienen id de persona y el chequeo los reparte
  al azar, así que frames de la misma persona caen en train y test; la falsa aceptación es
  OPTIMISTA y más negativos de la misma fuente la inflan más. Las comparaciones entre corridas
  también mezclan composiciones distintas de negativos. `dislike` y los otros gestos tienen
  n = 50 (un solo acierto cambia 2 puntos). No se midió el costo por letra (A, S, E, M).
  Sin prueba con cámara. Siguiente: negativos grabados con la cámara de la app y de personas
  distintas, que dan una validación independiente.

## 2026-10-02 · Modelo final 35 clases + `no_es_seña` (ANH, 600 negativos limpios)
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
- Corrida completa con el script corregido (35 clases, float32, 200 épocas, `modelo_final_f32`):
  paridad con Keras 100 % (diff 9.5e-7), 0.208 MB (7.67 MB con el detector). Rechazo:

  | Umbral | Falso rechazo | Falsa aceptación | Top-1 con rechazo |
  |---|---|---|---|
  | ninguno | 0.6 % | 14.7 % | 86.3 % |
  | 0.5 | 15.3 % | 7.3 % | 78.3 % |
  | 0.7 | 36.7 % | 4.2 % | 61.4 % |
  | 0.9 | 60.1 % | 2.2 % | 39.6 % |

  Falsa aceptación por clase de negativo (sin umbral): `no_gesture` (mano en reposo) 19.8 %,
  `dislike` 10.0 %, `peace_inverted` 4.0 %, `stop_inverted` 4.0 %, `two_up_inverted` 0.0 %.
- Conclusión: el umbral por confianza es una mala palanca (con 0.7 se rechaza 1 de cada 3
  señas buenas). La mano en reposo es lo más aceptado como seña (19.8 %), justo el fallo que más
  rompe la confianza en una demo. Es probable (sin comprobar) que una mano relajada se parezca a un
  puño (A, S, E, M); falta medir qué letras absorben esos falsos positivos.
- Pendiente: más negativos de reposo, y medir si exigir k frames seguidos ayuda (los frames de
  una mano quieta están correlacionados, así que puede no ayudar).
- Límites: los negativos no tienen id de persona (la falsa aceptación no es independiente por
  persona); sin prueba con cámara; sin umbrales calibrados.

## 2026-10-01 · ANH vs AN: letras + números, 35 clases (hipótesis de resolución refutada)
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
  | ANH, letras + números (35) | **87.6 %** | 87.1 % | 89.3 % |
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
