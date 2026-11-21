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

**Evidencia:** rama `claude/project-thread-q2rjdn`; `python -m pytest tests/test_review_queue.py -q` → 16 passed (incluye la prueba sintética); `ruff check lsc70 tests --select E,F,W --ignore E501` limpio; auditoría con `python -m lsc70.queue_audit --out data/scraper --refs landmarks_cache.npz`. El resto de la suite no corre aquí por falta de `cv2` (ya ocurría antes).

**Riesgos:** (1) los videos sintéticos son más limpios que los reales: la mano real tiembla y MediaPipe oscila, así que `STILL_MOVE` y `FIT_MAX_DIST` podrían ser muy estrictos o laxos; hay que correr `queue_audit` en el PC de Kevin y mirar `page_after.n`. (2) En datos reales el criterio "en movimiento" de la auditoría se parece al del filtro, no es independiente. (3) La causa de "material poco favorable" puede ser también la selección de videos (`discover`), no tocada. (4) Las señas dinámicas (6-10, G, H, J, Ñ, S, Z) ya no aparecen como fotograma; requieren revisión por secuencia.

**Siguiente:** revisar las 43 pendientes; si la tasa de rechazo humano es alta, el criterio automático no basta y habría que mirar la selección de videos. n=43 es poco: más videos (`discover`) darían más pendientes.
