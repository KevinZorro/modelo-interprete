# Experimentos

Memoria del proyecto: una entrada por experimento, la más reciente arriba. Incluye los
resultados negativos. Léelo antes de proponer un experimento para no repetir.

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

## Pendiente · ANH vs AN (hipótesis de resolución)
- Hipótesis (a refutar): C, K, M "fallan por articulación fina que no se resuelve a 120x120"
  y mejoran con más resolución. Ojo: en dataset C/K/M ya están en la media (0.94/0.90/0.96);
  las flojas son H y E. Antes de entrenar, `lsc70.medir_mano` dice si AN tiene más píxeles
  de mano que ANH.
- Regla de decisión fijada de antemano: ver `lsc70/comparar.py`.
- Cómo correrlo: `docs/lsc70_colab.md`.
