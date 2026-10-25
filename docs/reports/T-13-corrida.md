# T-13 — Corrida real del scraper (equipo de Kevin, Windows, Python 3.11)

**Estado:** dos corridas reales (21 y luego 52 candidatos). Acumulado: 37 videos procesados, 2110 landmarks, 23 canales. La tabla y el texto de abajo describen la primera corrida (14 videos); la segunda (23 videos nuevos) está al final.

**Cambios** (`lsc70/scraper.py`): (1) selector de formato que acepta streams de solo video H.264 ≤480p, porque YouTube ya no sirve formatos combinados y fallaban los 20 videos; así tampoco se baja audio. (2) `js_runtimes: node`, que yt-dlp exige para resolver las firmas del reproductor.

| Tipo de búsqueda | Candidatos | Procesados | Videos con mano | Landmarks |
|---|---|---|---|---|
| lsc (abecedario) | 8 | 8 | 8 | 420 |
| numeros | 7 | 2 | 2 | 120 |
| reposo | 6 | 4 | 3 | 240 |
| **Total** | 21 | 14 | 13 | **780** |

Los 6 fallos son "This video is not available" (retirados o con restricción), sin 429 ni bloqueo de bot en esta IP. 8 canales distintos; 282 de 780 muestras son mano izquierda.

**Detección:** 13/14 videos (93 %) dieron mano. La tasa por fotograma no se puede leer: los 13 llegaron al tope de 60 muestras por video (`DEFAULT_MAX_PER_VIDEO`) y el scraper no registra los fotogramas intentados. Hay que añadir ese contador antes de citar un porcentaje.

**Evidencia:** base `5dc3edd`; `python -m lsc70.scraper run --model hand_landmarker.task --limit-videos 20 --out data/scraper`; `pytest tests/test_scraper.py` → 5 passed. Modelo: `hand_landmarker.task` del ejemplo oficial de mediapipe-samples ya presente en el equipo.

**Criterios:** procedencia por muestra ✓ (URL, fecha, canal); `label=null`, `review_status=pending` en 780/780 ✓; ningún .mp4/.webm/.jpg en el repo tras la corrida ✓; límite de 5 s ✓; sin cookies ✓.

**Riesgos:** (1) las etiquetas "lsc/numeros/reposo" son de la búsqueda, no de la seña: un video de "reposo" puede traer señas y viceversa. (2) 60 muestras seguidas por video están muy correlacionadas; para entrenar cuenta como mucho menos de 780 ejemplos. (3) Con 8 canales el sesgo de grabación es alto. (4) `data/scraper/` no se versiona; los 780 landmarks viven solo en el equipo de Kevin.

**Segunda corrida (con contador de fotogramas):** 23 videos nuevos de números y reposo (12 y 11), 2369 fotogramas intentados, 1330 con mano válida → **56,1 %** por fotograma (números 678/1132 = 59,9 %; reposo 652/1237 = 52,7 %); ningún video sin mano. Total por tipo: lsc 420, numeros 798, reposo 892. 15 candidatos más dieron "not available". El 56 % es sobre fotogramas tomados cada 10 hasta el tope de 60; no es el FR/FA del modelo.
Cambio extra: lectura/escritura en UTF-8 (en Windows `read_text()` rompía con acentos). Los videos "reposo" venían de búsquedas de saludos y clases, así que muchos traen señas, no reposo.
**Siguiente:** T-14 (ver `T-14.md`).
