# T-17 · Decisión: deletreo en la app y mejora del modelo (solo investigación, nada implementado)

**Estado:** informe de decisión. No hay métricas nuevas: todo número sale de reportes T-05..T-16 y de leer el repo de la app (commit `281771a`). Lo no medido está marcado.

## Lo que la app ya hace (leído en `KevinZorro/SignaCo`)
- `SignClassifier.clasificar(63 floats)` devuelve letra, no reconocida o sin mano. No expone el top-k, solo la mejor candidata.
- `EstabilizadorDePredicciones` confirma una letra UNA vez por sostén (pestillo) y se reinicia al quitar la mano: sirve para practicar una letra, no para deletrear. Una letra repetida (LL, RR) o dos seguidas sin soltar la mano no se pueden escribir.
- `VocabularioActivo` solo habilita 19 letras (sin C, E, H, K, M, N, Z, Ñ), por una prueba con cámara que `CLAUDE.md` de este repo declara inválida. G, J y S siguen activas aunque Kevin las marcó como dinámicas: hay que confirmar si la app las enseña solo como pose inicial.
- Corre el `.tflite` de 28 clases (59 KB), no el T-08. Sin permiso de internet, y el CLAUDE.md de la app deja las señas dinámicas fuera de alcance.

## Objetivo 1 · Deletreo
| Opción | Qué es | Pros | Contras / riesgo | Costo |
|---|---|---|---|---|
| A. Estático con palabras cerradas | Máquina de estados sobre el flujo de frames (sostener, aceptar, exigir soltar o pausa para repetir) y decodificación contra una lista de palabras con distancia de edición | Sin reentrenar; el contrato del clasificador no cambia; corrige letras confundidas | Solo palabras sin letras dinámicas; depende de reactivar letras; en 5 letras con 88.8 % por letra se acierta ~53 % sin corrección (cálculo, no medido) | 1 hilo + PR en la app |
| B. LM de caracteres + beam search (pedido en las instrucciones) | Modelo de lenguaje abierto en español | Admite palabras libres | Exige top-k por frame (cambia el contrato); con letras que fallan un 30-40 % el LM inventa palabras; corpus y tamaño por decidir | 2-3 hilos |
| C. Dinámico + enrutador (T-09..T-11) | Secuencias para J, Z, Ñ, G, H, S, LL, RR | Único camino a deletreo completo | Sin datos de secuencias (LSC70 son imágenes, no hay grabaciones propias); solo videos del scraper revisados por Kevin; meses | M3, 2027-03 |
**Recomiendo A, con la lista de palabras como puente hacia B.** Entrega valor pronto, no necesita datos nuevos y fija el contrato (flujo de entrada, eventos de salida) que B y C reutilizarían. B abierto no se justifica mientras la letra individual falle tanto; C se decide cuando se sepa cuántas secuencias salen del scraper.

## Objetivo 2 · Mejor modelo
| Opción | Efecto esperado | Contras | Costo |
|---|---|---|---|
| 1. Negativos reales del scraper (T-13/T-14) | Ataca el cuello medido: mano en reposo aceptada 17.8 %, FA dev 6.3 % (meta <5 %). Los videos de deletreo traen justo las transiciones que el deletreo necesita rechazar | Kevin revisa ≥200 muestras (una por video) | Tiempo de Kevin; pocos tokens |
| 2. Cambiar el modelo en la app por el T-08 (mismo contrato v1, umbral por clase) | Quita el desfase entre lo medido y lo desplegado | La app dice "modelo resuelto": es decisión de alcance; FA de campo 20.7 % no se ha vuelto a medir | 1 PR pequeño |
| 3. Números (T-16, reentrenar 36 clases por folds) | Solo si la app los quiere; cuestan 1.6 pt y chocan 1=F, 2=V, 3=W | No ayuda al deletreo | 1-2 hilos |
| 4. Más arquitectura/HPO | T-05: MLP ≈ logreg; T-07 ganó 2 pt de FA | Rendimientos decrecientes | No lo recomiendo |
**Recomiendo 1, y luego 2.** La evidencia dice que faltan negativos reales, no un modelo mejor.

## Orden propuesto (T-xx con dependencias)
| ID | Tarea | Depende de | Criterio verificable |
|---|---|---|---|
| T-17a | Prueba con cámara de C, E, H, K, M, N (y G, J, S) con `PROTOCOLO_prueba_camara.md` | — | Recall por letra con IC95; decide qué letras se reactivan |
| T-18 | Kevin revisa ≥200 muestras de la cola (PR #17 ya fusionado) | — | `validated.jsonl` con revisor y fecha; % de aciertos de la sugerencia |
| T-19 | Reentrenar con negativos reales (reposo, transición) y medir en dev por persona | T-18 | FA <5 % con FR <10 %, 3 semillas; test una sola vez |
| T-20 | Especificar el motor de deletreo (entrada: frames con tiempo; salida: eventos letra/palabra) con vectores de prueba y referencia en Python | T-17a | Pruebas en verde; casos LL, RR, soltar y repetir |
| T-21 | Port a Kotlin en la app y cambio del modelo | T-19, T-20, decisión de alcance | PR en SignaCo con pruebas de dominio |
| T-22 | Decidir dinámico (T-09) según secuencias disponibles | T-18 | Cuántas secuencias revisadas hay |

## Decisiones para Kevin
1. ¿Aprobar A como camino del deletreo (sin LM abierto por ahora)?
2. ¿Autorizar cambios en la app (nueva pantalla, estabilizador sin pestillo, modelo T-08)? Es alcance de otro repo.
3. ¿Quién revisa las muestras y cuánto tiempo puede dar por semana (T-18)?
4. ¿G, J y S se enseñan como pose estática o se sacan del vocabulario hasta tener dinámico?
5. ¿Hacer la prueba con cámara T-17a antes de reactivar letras?
**Riesgos:** todo es hipótesis hasta probarlo con cámara; el test de 15 personas sigue contaminado; no medí cuántas palabras cubre el vocabulario actual.
