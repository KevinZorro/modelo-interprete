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

## Lo que NO se midió

La **falsa aceptación de mano en reposo** no se pudo recalcular: los 800 negativos
de HaGRID no están en el cache (solo tiene letras). No se hereda la cifra vieja
porque los umbrales cambiaron. Mientras falte, `config_objetivo.json` lleva
`umbrales_provisionales: true` y cada letra `fa_reposo: null` con estado
`pendiente_export_negativos`.

Para cerrarlo: correr la celda **15-bis. Exportar los negativos de reposo** del
notebook (va justo después de la recolección de HaGRID y no reentrena nada) y
volver a correr con `--negativos`. Cuando el archivo exista, la calibración añade
una tercera restricción (FA de reposo ≤ 5 %) y los umbrales bajos —D, H, R, T, W,
Y, que hoy están en 0.01— **van a subir**.

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
python -m evaluacion_objetivo.evaluar                 # métricas + config_objetivo.json
python -m evaluacion_objetivo.evaluar --negativos negativos_reposo.npz
```
