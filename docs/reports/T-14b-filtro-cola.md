# T-14b — Filtro de la cola de revisión (solo poses sostenidas)

**Estado:** implementado y probado con videos sintéticos; falta una corrida sobre los landmarks reales de Kevin (un comando, sin revisar nada a mano).

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

**Evidencia:** rama `claude/project-thread-q2rjdn`; `python -m pytest tests/test_review_queue.py -q` → 16 passed (incluye la prueba sintética); ruff limpio. El resto de la suite no corre aquí por falta de `cv2` (ya ocurría antes).

**Riesgos:** (1) los videos sintéticos son más limpios que los reales: la mano real tiembla y MediaPipe oscila, así que `STILL_MOVE` y `FIT_MAX_DIST` podrían ser muy estrictos o laxos; hay que correr `queue_audit` en el PC de Kevin y mirar `page_after.n`. (2) En datos reales el criterio "en movimiento" de la auditoría se parece al del filtro, no es independiente. (3) La causa de "material poco favorable" puede ser también la selección de videos (`discover`), no tocada. (4) Las señas dinámicas (6-10, G, H, J, Ñ, S, Z) ya no aparecen como fotograma; requieren revisión por secuencia.

**Siguiente:** Kevin ejecuta `queue_audit` (30 s) y me pasa el JSON; ajustar umbrales con eso.
