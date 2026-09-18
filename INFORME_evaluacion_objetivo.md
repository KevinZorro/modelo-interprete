# Evaluación dirigida por objetivo — resultados

Post-proceso sobre el modelo ya desplegado (`signaco_abecedario_rechazo.tflite`).
**No se reentrenó nada.** La app siempre sabe qué letra pidió, así que la pregunta
deja de ser "¿cuál de 28 clases es?" y pasa a ser "¿esto es la letra objetivo?".

## Protocolo

Se reconstruyó el split exacto con el que se entrenó el modelo desplegado
(`GroupShuffleSplit`, semilla 42, 15/70 participantes, letras + 800 negativos de
HaGRID agrupados individualmente). La reconstrucción se valida contra las cifras
impresas en el notebook: **train 8992 / test 2617 / 171 negativos / 428 tomas**,
todas exactas. Eso permite evaluar sobre los **16 participantes que el modelo
nunca vio**, sin reentrenar.

- Evaluación **por toma** (promedio de los ~6 frames), comparable con el 88.8 %.
  Referencia de sanidad: 90.9 % por toma en modo 27 clases sobre este held-out.
- Umbrales calibrados **leave-one-participant-out**: el umbral de cada persona se
  fija con las otras 15. Ninguna persona influye en el umbral con que se la juzga.
- El percentil de falso rechazo se estima sobre **frames** (~90 por letra) porque
  con 15 tomas "el umbral más alto que no rechaza ninguna" es el mínimo de esas
  15 — un estadístico de orden extremo que producía un FRR artificial de 1/16 en
  casi todas las letras. La **decisión** se sigue evaluando por toma.

## Clases de equivalencia: ninguna entre letras

`analisis_equivalencias.py` busca pares con confusión simétrica **y**
co-activación alta en ambas direcciones (probabilidad media de la clase j cuando
la real es i — mide masa repartida, no solo quién gana el argmax).

| par | conf. a→b | conf. b→a | co-act. a→b | co-act. b→a | veredicto |
|---|---|---|---|---|---|
| **N / Ñ** | 2 | 3 | **0.300** | **0.305** | equivalencia real |
| I / J | 1 | 1 | 0.089 | 0.126 | error |
| H / R | 1 | 1 | 0.093 | 0.086 | error |
| C / E | 1 | 1 | 0.082 | 0.143 | error |
| F / L | 1 | 0 | 0.113 | 0.047 | error |
| S / Z | 0 | 3 | 0.044 | 0.161 | error |

El corte es limpio: el siguiente candidato tras N/Ñ está en 0.143 en su mejor
dirección, menos de la mitad del mínimo de N/Ñ.

**N/Ñ es una equivalencia real y por eso mismo no se usa**: en pose estática Ñ es
N, solo las separa el movimiento. Aceptar el grupo implicaría no poder enseñar
ninguna de las dos. Ñ ya estaba excluida por dinámica; esto explica por qué.

Las equivalencias con gestos de HaGRID (peace→V 0.930, three→W 0.985, ok→T 0.983,
like→G 0.959) **no se codifican**: el modelo no tiene columnas para esos gestos.
Documentan por qué esas clases no podían usarse como negativos, nada más.

## Regla de decisión

Por orden de prioridad:

1. **Tope duro de falsa aceptación cruzada: 20 %.** Si la lección de la X aprueba
   más de una de cada cinco ejecuciones de otras letras, no enseña nada.
2. Dentro de lo que cumple el tope, el umbral **más alto** que respeta el
   presupuesto de falso rechazo (5 % a nivel de frame): máximo margen sin frustrar.
3. Si no hay umbral que cumpla ambas, gana el tope de falsa aceptación y el falso
   rechazo sube — se marca en `motivo_umbral`.

## Las 8 letras excluidas

Criterio: **viable** solo si el IC95 entero queda bajo ambos criterios (FRR ≤ 20 %,
FA cruzada ≤ 20 %); **no viable** si el IC entero queda del lado malo;
**indeterminada** si el intervalo cruza el criterio.

| letra | umbral | FRR (IC95) | FA cruzada (IC95) | FRR k=2 | estado |
|---|---|---|---|---|---|
| E | 0.06 | 0.0 % [0.0–19.4] | 3.2 % [1.9–5.3] | 0.0 % | **viable** |
| C | 0.25 | 6.2 % [1.1–28.3] | 0.7 % [0.2–2.1] | 6.2 % | indeterminada |
| K | 0.85 | 6.2 % [1.1–28.3] | 0.0 % [0.0–0.9] | 12.5 % | indeterminada |
| M | 0.21 | 6.2 % [1.1–28.3] | 0.0 % [0.0–0.9] | 0.0 % | indeterminada |
| N | 0.25 | 6.7 % [1.2–29.8] | 2.2 % [1.2–4.1] | 6.7 % | indeterminada |
| H | 0.01 | 0.0 % [0.0–19.4] | **17.2 % [13.9–21.2]** | 18.8 % | indeterminada |
| Ñ | — | — | — | — | no viable (dinámica) |
| Z | — | — | — | — | no viable (dinámica) |

**Lectura honesta:** con 16 tomas por letra, una sola toma rechazada empuja el IC
hasta el 28 % y deja la letra sin resolver. Solo **E** queda por encima del
criterio con el intervalo completo. C, K, M y N apuntan bien (FA cruzada entre
0.0 % y 2.2 %, que es la parte medida con precisión: n≈416) pero fallaron una toma
y eso, con este n, no se puede distinguir del ruido.

**H es distinta de las otras cuatro**: su problema no es el falso rechazo sino que
para no rechazar hay que bajar el umbral a 0.01, y ahí la lección aprueba el
17.2 % de las ejecuciones de otras letras (IC hasta 21.2 %, rozando el tope). H no
está indeterminada por falta de datos, sino porque **no hay umbral que la salve**.

## Con los negativos exportados: los umbrales ya no son provisionales

Los 800 negativos se exportaron y se recalibró con la tercera restricción activa
(FA de reposo ≤ 5 %). Los umbrales subieron, como estaba previsto:

| letra | umbral provisional | umbral final | FRR (IC95) | estado | en vocabulario activo |
|---|---|---|---|---|---|
| R | 0.01 | **0.61** | **62.5 % [38.6–81.5]** | **no viable** | sí |
| S | 0.05 | 0.35 | 18.8 % [6.6–43.0] | indeterminada | sí |
| H | 0.01 | 0.23 | 18.8 % [6.6–43.0] | indeterminada | no |
| L | 0.54 | 0.83 | 12.5 % [3.5–36.0] | indeterminada | sí |
| P | 0.04 | 0.20 | 6.2 % [1.1–28.3] | indeterminada | sí |
| J | 0.02 | 0.12 | 0.0 % [0.0–20.4] | indeterminada | sí |
| T, W, D | 0.01 | 0.03–0.05 | 6.2 % [1.1–28.3] | indeterminada | sí |

Ganancia: **G e I** pasan a `viable` (antes indeterminadas), así que el conjunto
con confianza es **E, G, I**. Pérdida grande: **R se cae a `no_viable`** con un
falso rechazo del 62.5 %, y está en el vocabulario activo de la app.

## Pero los negativos traen letras disfrazadas

La exportación incluyó 8 clases de negativos difíciles además de `no_gesture`:
`dislike`, `one`, `palm`, `peace_inverted`, `stop_inverted`, `three2`, `two_up`,
`two_up_inverted`. `analisis_negativos.py` mide qué tan letra parecen:

| letra | negativos con p>0.5 | con p>0.8 | con p>0.95 | máximo |
|---|---|---|---|---|
| R | 63 | 18 | 3 | 0.974 |
| L | 61 | 42 | **19** | 0.998 |
| S | 35 | 21 | 7 | 0.986 |
| Q | 17 | 9 | 5 | 0.994 |
| K | 14 | 3 | 0 | 0.904 |
| V | 13 | 8 | 3 | 0.991 |

Diecinueve negativos puntúan por encima de 0.95 como L, y el máximo llega a
0.998. Eso no es un modelo confundido: es la misma pose. Comparar con el sondeo
del propio notebook, que daba 0.930 para peace→V y 0.985 para three→W — el mismo
orden de magnitud, y ahí la conclusión fue que esas clases **no pueden ser
negativos**.

Consecuencia directa: el umbral de L (0.54 → 0.83) y el de R (0.01 → 0.61) están
inflados por negativos que probablemente **son** esas letras. El 62.5 % de falso
rechazo de R puede ser un defecto del conjunto de negativos, no del modelo.

### Por qué el sondeo de colisiones no lo atrapó

Las 8 clases usadas salieron de la lista `CANDIDATAS A NEGATIVO DIFÍCIL` del
sondeo del notebook, que las aprobó correctamente según su criterio: colisiona
una clase si **≥50 % de sus muestras** caen en una letra con **confianza ≥0.75**.

| clase usada | % como letra | letra dominante | conf. |
|---|---|---|---|
| `two_up` | 12.5 % | **R** | 0.618 |
| `two_up_inverted` | 22.5 % | **R** | 0.654 |
| `peace_inverted` | 15.0 % | **R** | 0.542 |
| `one` | 22.5 % | S | 0.597 |
| `three2` | 20.0 % | V | 0.577 |
| `dislike` | 20.0 % | P | 0.574 |
| `stop_inverted` | 17.5 % | Q | 0.535 |

El criterio es **por clase y sobre el promedio**, así que una clase donde solo un
15–22 % de las muestras son letras clarísimas pasa el filtro: el promedio la
diluye. Pero esas muestras minoritarias siguen dentro de los negativos, y el tope
del 5 % de FA de reposo es tan estricto que un puñado de ellas basta para mover
el umbral.

Dos cosas que el filtro por clase no vio:

1. **Tres de las ocho clases apuntan a R** (`two_up`, `two_up_inverted`,
   `peace_inverted`). Juntas son 150 negativos con un 12–22 % de muestras
   R-como-R: del orden de 25 negativos empujando el umbral de R, que está en el
   vocabulario activo. Cuadra con los 63 negativos que puntúan >0.5 como R.
2. **Ninguna clase tiene L como letra dominante**, y aun así hay 19 negativos por
   encima de 0.95 como L. Un subconjunto fuerte dentro de una clase "segura" es
   invisible para un criterio que promedia.

La lección para el pipeline: el filtro de negativos debe ser **por muestra**, no
por clase. Una clase puede aportar el 80 % de negativos legítimos y el 20 % de
letras disfrazadas, y con un tope del 5 % ese 20 % decide el umbral.

**Esto no se puede resolver con lo exportado:** el `.npz` guarda la lista de las 9
clases pero no la clase de cada muestra, así que se ve *qué* letras están
afectadas pero no *qué clase* las infla. La celda 15-bis del notebook ya guarda
`clase_por_muestra`; al reexportar, `analisis_negativos.py` señala la clase
culpable y el veredicto por clase.

Hasta entonces, los umbrales de R, L, S, Q, K y V deben tratarse como **inflados
al alza**, y el `no_viable` de R como **no confirmado**.

## Estas son hipótesis, no resultados

Todas las cifras de arriba son de dataset. M y N tenían 0.92 y 0.96 en dataset y
fallan con cámara real. Por eso cada letra sale en el JSON con
`verificada_con_camara: false`, y las recuperadas van a `vocabulario_propuesto`,
**nunca a `vocabulario_activo`**. Ninguna entra a la app sin prueba con cámara.

## Cómo correrlo

```bash
pip install -r requirements.txt
python -m evaluacion_objetivo.pruebas                 # lógica de decisión
python -m evaluacion_objetivo.analisis_equivalencias  # evidencia de equivalencias
python -m evaluacion_objetivo.evaluar --negativos negativos_reposo.npz
python -m evaluacion_objetivo.analisis_negativos      # ¿hay letras disfrazadas?
```
