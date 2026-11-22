# T-14b — Filtro de la cola de revisión (solo poses sostenidas)

**Estado:** implementado, probado con videos sintéticos y medido sobre los landmarks reales de Kevin (2137 muestras, 63 videos). Umbrales sin cambios.

**Problema:** la página mostraba un fotograma por cada tramo, incluidos los de un solo fotograma (mano en movimiento o transición) y las señas "dinámicas" saltaban el requisito de pose sostenida. Kevin: ~80 % no era seña.

**Cambios** (`lsc70/review_queue.py`, `review_server.py`, `queue_audit.py`):
- Un tramo "sostenido" exige además que la mano no se desplace (> `STILL_MOVE` = 0,10 tamaños de mano): antes solo miraba la forma, y una mano que viaja con la misma forma contaba como quieta.
- `review_items(strict=True)` (por defecto) solo muestra poses sostenidas ≥ 1 s (`HOLD_S`) a distancia coseno ≤ 0,17 de alguna pose de referencia (p95 de poses reales del test), como máximo 3 por video, las más largas y cercanas primero. Lo ya decidido siempre se queda. `--all` en el servidor restaura todo.
- `python -m lsc70.queue_audit --out data/scraper --refs landmarks_cache.npz` mide la basura sin humanos: aislada, en movimiento (posición cruda) o sin parecido a ninguna pose.

**Medición (sintética, 3 semillas, 30 videos cada una):** poses reales del dataset como "sostenidas" y transiciones con mano viajera; verdad conocida.

| | Muestras en página | Basura real | Poses sostenidas cubiertas |
|---|---|---|---|
| antes | 265-268 | 69-73 % | 96-100 % |
| después | 56-60 | 0 % | 85-88 % |

**Medición real** (`queue_audit`, commit de esta rama; 63 videos):

| Conjunto | n | Basura | Aislada | En movimiento | Sin parecido |
|---|---|---|---|---|---|
| todas las muestras | 2137 | 62,6 % | 2,9 % | 55,5 % | 16,8 % |
| página antes del filtro | 1706 | 76,3 % | 3,7 % | 69,4 % | 19,1 % |
| página después (total) | 365 | 63,3 % | 6,6 % | 53,4 % | 18,4 % |
| **página después, solo pendientes** | **43** | **0,0 %** | 0 | 0 | 0 |

Por qué "después (total)" engaña: 322 de las 365 ya tenían decisión humana y la página las conserva siempre (así no se pierde el trabajo), y ese 63 % es la basura de antes. Lo que el filtro selecciona de verdad son las 43 pendientes, con 0 % de basura. Por eso `queue_audit` ahora devuelve `page_after_pending`. Deriva de mano entre fotogramas vecinos en esas 43 (en tamaños de mano): p50 0,023, p90 0,047, p95 0,059, frente a `MOVING` 0,15; distancia coseno máxima 0,151 frente a `FIT_MAX_DIST` 0,17; sostenidas 1,0-4,6 s. Hay holgura en ambos umbrales, así que no se ajustan: endurecerlos bajaría n (43, el piso útil es ~30) sin quitar basura medible.

**Corrida de ampliación (2026-11-21, solo landmarks; video temporal, nunca guardado):** `discover --preset` para `lsc` y `numeros` (+70 candidatos), luego dos `run` (25 videos máx. cada uno), sin 429 ni bloqueo de bot. Procesados 14 + 8 = 22 videos buenos (+3179 muestras; 28 fallidos, sin formato disponible: YouTube no resolvió el desafío JS de yt-dlp). Umbrales sin cambios.

| Conjunto | Antes (63 videos) | Después (85 videos) |
|---|---|---|
| todas las muestras | 2137 | 5316 |
| página antes del filtro (n / basura) | 1706 / 76,3 % | 3935 / 73,6 % |
| página después, total (n / basura) | 365 / 63,3 % | 416 / 55,5 % |
| **pendientes (n / basura)** | **43 / 0 %** | **94 / 0 %** |

Quedan 94 pendientes limpias frente a la meta de ~200: 22 videos aportaron 51 poses nuevas (~2,3 por video), así que alcanzar 200 pide unos 45 videos buenos más, y el cuello de botella es que muchos videos no se pueden descargar. Basura 0 % solo prueba que el filtro no deja pasar movimiento ni poses sin parecido; si las poses son correctas lo dirá la revisión humana.

**Segunda ampliación (2026-11-21):** el fallo de descarga era el desafío JS de yt-dlp. Se arregló solo con `pip install -U "yt-dlp[default]"` (yt-dlp 2026.10.21 -> 2026.10.26; el extra instala `yt-dlp-ejs` 0.8.0 con el solver local) y el Node v24 ya presente; no hizo falta el solver remoto de GitHub ni cambios de código. Los 28 fallidos se reintentaron solos (nunca entraron a `done.txt`, sin duplicar muestras). Dos `run` más: 23 + 25 = 48 videos buenos (+6764 muestras), 2 fallidos nuevos, sin 429 ni bloqueo de bot, sin cookies. Umbrales sin cambios.

| Conjunto | 63 videos | 85 videos | 132 videos |
|---|---|---|---|
| todas las muestras | 2137 | 5316 | 12080 |
| página antes del filtro (n / basura) | 1706 / 76,3 % | 3935 / 73,6 % | 8604 / 68,1 % |
| página después, total (n / basura) | 365 / 63,3 % | 416 / 55,5 % | 526 / 43,9 % |
| **pendientes (n / basura)** | 43 / 0 % | 94 / 0 % | **204 / 0 %** |

Meta de ~200 pendientes cumplida (204). Siguen siendo candidatas: la basura 0 % solo dice que no pasan movimiento ni poses sin parecido; falta la revisión humana, que dirá si son poses correctas. Los candidatos de `discover` se agotaron (107 videos hechos).

**Con `REVIEW_HOLD_S` 0,6 y `MAX_PER_VIDEO` 30 (PR #25, mismos datos, 132 videos):** pendientes 204 -> 1621 (basura 0 % -> 6,7 %; en movimiento 6,1 %, aislada 0,6 %, sin parecido 0 %); página total 528 -> 1945 (basura 43,8 % -> 17,4 %). 98 de 132 videos aportan pendientes; por video mediana 17,5, p90 30, máximo 30 (tope), 82 videos con >=5. Toda la basura (108) está en 5 videos, 4 de ellos topados en 30: un video con mano viajera sigue colando poses. El 71 % de las pendientes tiene otra casi idéntica (coseno < 0,02) en el mismo video, y la clase sugerida B es el 29 % de las pendientes: revisar por duplicados antes de contarlas como poses distintas.

**Con colapso de poses casi idénticas por video (PR #26, mismos datos, 132 videos):** pendientes 1621 -> 921 (basura 6,7 % -> 10,4 %: en movimiento 10,1 %, aislada 0,3 %, sin parecido 0 %); página total 1945 -> 1245 (basura 17,4 % -> 26,3 %). 98 videos aportan; por video mediana 10, p90 18, máximo 30; 73 con >=5. Pendientes con otra casi idéntica (coseno < 0,02) en el mismo video: 0,0 % (antes 71 %). Clases sugeridas: B 199, F 104, K 87, R 70, G 59 (B baja de 29 % a 22 %). Los 5 videos con más basura aportan 115 pendientes (12,5 %) y toda la basura (96); sin ellos la basura pendiente es 0 %. La tasa sube porque el colapso quitó duplicados limpios y la basura ya estaba en videos de mano viajera.

**Exclusión de videos mayormente en movimiento (PR #26, `VIDEO_MOTION_MAX` 0,75, sin cambiar):** `video_motion_share` sobre los 132 videos: p10 0,0, p50 0,39, p90 0,79, máx 1,0. No hay hueco que separe los 5 videos con más basura (share 0,80 / 0,85 / 0,91 / 0,65 / 0,22) de los buenos: hay videos sin basura con 0,83-0,94, y uno de los 5 tiene solo 0,22. Por eso no se calibró y se mantiene 0,75. Medido con el código real: pendientes 921 -> **796**, basura 10,4 % -> **2,0 %** (16 de 796: 14 de un video con share 0,65 y 2 de otro con 0,22), 13 videos excluidos, 45 poses no basura perdidas. Otros umbrales (excluidos / pendientes / basura): 0,60 → 29 / 692 / 0,3 % (135 buenas perdidas); 0,80 → 7 / 828 / 1,9 % (13); 0,85 → 3 / 889 / 8,0 % (7); 0,95 → 0 / 921 / 10,4 %. **Costo aceptado:** la exclusión usa el mismo criterio de movimiento que la auditoría (`moving`), así que este 2 % ya no es una medida independiente; solo la revisión humana dirá si las poses son correctas.

**Evidencia:** rama `claude/project-thread-q2rjdn`; `python -m pytest tests/test_review_queue.py -q` → 16 passed (incluye la prueba sintética); `ruff check lsc70 tests --select E,F,W --ignore E501` limpio; auditoría con `python -m lsc70.queue_audit --out data/scraper --refs landmarks_cache.npz`. El resto de la suite no corre aquí por falta de `cv2` (ya ocurría antes).

**Riesgos:** (1) los videos sintéticos son más limpios que los reales: la mano real tiembla y MediaPipe oscila, así que `STILL_MOVE` y `FIT_MAX_DIST` podrían ser muy estrictos o laxos; hay que correr `queue_audit` en el PC de Kevin y mirar `page_after.n`. (2) En datos reales el criterio "en movimiento" de la auditoría se parece al del filtro, no es independiente. (3) La causa de "material poco favorable" puede ser también la selección de videos (`discover`), no tocada. (4) Las señas dinámicas (6-10, G, H, J, Ñ, S, Z) ya no aparecen como fotograma; requieren revisión por secuencia.

**Siguiente:** revisar las 43 pendientes; si la tasa de rechazo humano es alta, el criterio automático no basta y habría que mirar la selección de videos. n=43 es poco: más videos (`discover`) darían más pendientes.

## Ajuste: señas rápidas encadenadas (2026-11-22)
Kevin vio que en un clip de ~4 s con 5 señas solo se tomaba la última: con muestreo de 3/s cada seña rápida deja 2-3 muestras (0,7-1,0 s) y el umbral de 1 s las descartaba; además el tope de 3 poses por video cortaba los abecedarios (27 letras). Cambios: `REVIEW_HOLD_S` 1,0 → 0,6 s (dos muestras seguidas, misma regla de quietud y parecido) y `MAX_PER_VIDEO` 3 → 30. Sintético con 70 % de señas rápidas, 3 semillas: poses cubiertas 51-66 % → 89-96 %; basura 0 % → 3-9 % (mayormente la muestra pegada a una pose, que el criterio sintético cuenta como transición). Pendiente medir con los datos reales de Kevin.

**Duplicados (2026-11-22):** con los datos reales el ajuste de señas rápidas subió los pendientes de 204 a 1621 y el 71 % tenía otra pose casi idéntica (coseno < 0,02) en el mismo video. `best_per_video` ahora colapsa esas repeticiones (se queda la de mayor duración): una revisión cubre la pose. Sin cambio medible en la prueba sintética (sus poses son todas distintas); pendiente medir con los datos reales.
