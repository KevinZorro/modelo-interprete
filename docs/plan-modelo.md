# Plan del modelo Signaco

Estados: pendiente, en curso, en revisión, hecha y bloqueada.

## Hitos

| Hito | Criterio de salida | Fecha objetivo |
| --- | --- | --- |
| M1 Línea base honesta | Split por persona, test congelado y F1 macro, falsa aceptación y falso rechazo del modelo actual medidos | 2026-11-29 |
| M2 Estático mejorado | Falsa aceptación menor al 5 % con falso rechazo menor al 10 % en test por persona, exportado a TFLite con paridad | 2027-01-24 |
| M3 Dinámico v1 | Modelo dinámico y enrutador evaluados con el mismo protocolo | 2027-03-14 |
| M4 Datos propios | Descubrimiento de fuentes, cola de revisión y grabador con consentimiento funcionando, con procedencia por muestra | 2027-03-28 |
| M5 SDK | Librería con contrato de entrada y salida, model card y latencia medida | 2027-04-18 |

## Tareas

| ID | Tarea | Hito | Dónde | Depende de | Estado | Criterio de aceptación |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | Auditar el repo del modelo (preprocesamiento, entrenamiento, evaluación y exportación) y crear CLAUDE.md y docs/ si faltan | M1 | Claude Code | — | en revisión | Informe del estado actual con deudas y comandos que funcionan |
| T-02 | Inventariar los datasets con su licencia, escribir la datasheet y redactar los correos para pedir permiso comercial | M1 | Chat | — | pendiente | Tabla de fuentes con licencia, uso permitido y acción |
| T-03 | Crear el split por persona, congelar el test y volver a medir el modelo actual | M1 | Claude Code | T-01 | en revisión | F1 macro, matriz de confusión, falsa aceptación y falso rechazo en test por persona |
| T-04 | Crear el módulo único de limpieza y normalización con pruebas y vectores de prueba | M2 | Claude Code | T-01 | hecha | Pruebas en verde y contrato de entrada documentado |
| T-05 | Entrenar baselines y comparar modelos con tres semillas y MLflow | M2 | Claude Code | T-03, T-04 | en revisión | Tabla comparativa contra el modelo actual |
| T-06 | Ampliar la clase "no seña" y calibrar umbrales por clase | M2 | Claude Code | T-05 | hecha | Curva de falsa aceptación contra falso rechazo y umbral elegido con su porqué |
| T-07 | Ajustar hiperparámetros con Optuna sobre el mejor modelo | M2 | Claude Code | T-06 | hecho | Mejora en validación confirmada una vez en test |
| T-08 | Exportar a TFLite y medir paridad y latencia | M2 | Claude Code | T-07 | en revisión (paridad cumplida; latencia estimada, no medida) | Diferencia máxima de salidas y latencia en un celular de gama media |
| T-09 | Diseñar el modelo dinámico (datos, formato de secuencias, arquitectura y enrutador) | M3 | Chat | T-02 | pendiente | Documento de diseño aprobado por Kevin |
| T-10 | Construir el pipeline de secuencias y entrenar el modelo dinámico v1 | M3 | Claude Code | T-09 | pendiente | Métricas del dinámico con el mismo protocolo |
| T-11 | Implementar el enrutador estático y dinámico y evaluar el sistema completo | M3 | Claude Code | T-08, T-10 | pendiente | Métricas del sistema completo |
| T-12 | Diseñar el pipeline de datos (metadatos de procedencia, formato de consentimiento y flujo de revisión) | M4 | Chat | T-02 | pendiente | Esquema y formato aprobados por Kevin |
| T-13 | Construir el scraper de landmarks (descubre videos públicos, los procesa en temporal y guarda solo landmarks con procedencia) | M4 | Claude Code | T-12 | en revisión (`lsc70/scraper.py`) | Landmarks con URL, fecha y canal por muestra; ningún video ni foto en disco; etiqueta siempre vacía hasta revisión humana |
| T-14 | Construir la cola de revisión con similitud contra referencias nativas | M4 | Claude Code | T-04, T-12 | en revisión (`lsc70/review_queue.py`) | Herramienta que ordena candidatos y registra la validación |
| T-15 | Construir el grabador con consentimiento que guarda landmarks | M4 | Claude Code | T-12 | pendiente | Muestras con procedencia y autorización registradas |
| T-16 | Empaquetar el SDK del motor con contrato y model card | M5 | Claude Code | T-11 | pendiente | Librería integrable y documentada |

## Decisiones

| Fecha | Decisión | Porqué |
| --- | --- | --- |
| 2026-10-18 | ~~El scraper descubre fuentes y pide permiso; no descarga videos~~ (reemplazada el 2026-10-25) | Términos de las plataformas, Ley 1581 y valor comercial del dataset |
| 2026-10-25 | El scraper procesa videos públicos de forma temporal y guarda solo landmarks, aunque contravenga los términos de las plataformas; sin grabaciones propias | Decisión de Kevin: los landmarks no son datos personales ni imagen, el video nunca se guarda ni publica; el riesgo legal de los términos de uso lo asume el proyecto |
| 2026-10-18 | El proyecto trabaja el modelo y no la app | El modelo es el producto principal |

- Decisión de Kevin, 2026-10-25: 1=F, 2=V y 3=W son la misma pose (`configs/alias_poses.json`). Al entrenar con números el modelo predice la pose; evaluar con `lsc70.alias_poses.pose_accuracy`. Pendiente que T-10 y T-11 lo usen.
