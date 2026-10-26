# T-13 — Scraper de imágenes web (piloto)

**Estado:** implementado, probado y corrido sobre 103 candidatos; 27 landmarks nuevos. Es un piloto: el rendimiento es bajo.

**Decisión de Kevin (2026-10-25):** además de video, buscar imágenes en la web general y aceptar el riesgo de bloqueo. Sigue valiendo: la imagen se procesa en memoria y solo se guardan landmarks con procedencia (URL de página, URL de imagen, sitio, licencia si la fuente la da, fecha); sin cookies; límite de 5 s; si un sitio rechaza (403/429) se salta, sin evadir; 5 rechazos seguidos paran la corrida.

**Cambios:** `lsc70/image_scraper.py` (fuentes `ddg`, `openverse`, `commons`; `discover` y `run`, reanudable), integración en `review_queue` y en la página (muestra la imagen en vez del video, siempre "enseñada", sin modo dinámico, enlace a la página de origen), `tests/test_image_scraper.py` (8 pruebas).

| Paso | Resultado |
|---|---|
| Candidatos (DuckDuckGo, 9 consultas con pista) | 103 |
| Procesadas | 92 |
| Con mano válida | 27 (29 %) |
| Fallos | 8 (4 certificado SSL, 1 sitio con 403, 2 URL con tildes ya corregidas, 1 otro) |
| Landmarks | 27: 17 números, 10 abecedario; 19 sitios distintos |

**Evidencia:** commit de esta rama; `python -m lsc70.image_scraper discover "..." --kind lsc --hint B --source ddg --out data/scraper` y luego `run --model hand_landmarker.task --limit-images 110`; `pytest tests/test_image_scraper.py ...` → 27 passed. Sin bloqueos del buscador. Commons y Openverse devolvieron 0 con consultas en español; Commons sí responde en inglés.

**Criterios:** procedencia por muestra ✓; sin imágenes en disco ✓ (la prueba revisa los campos); `label=null` ✓; la pista de la consulta solo sugiere ✓.

**Riesgos:** (1) La pista no sirve: ninguna de las 27 coincide con la clase más cercana (0/27), así que no se debe usar para ordenar ni etiquetar. (2) Rinden poco porque muchas imágenes son dibujos o tienen varias manos. (3) Mezcla ASL con LSC; solo el conocedor puede separarlas. (4) Licencias: DuckDuckGo no las informa; el derecho de autor sigue aunque se guarden solo landmarks.

**Siguiente:** revisar las 27 en la página; si la tasa de aceptadas es buena, ampliar consultas (más pistas, Commons en inglés).
