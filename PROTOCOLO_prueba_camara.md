# Protocolo de prueba con cámara

## Por qué hace falta un protocolo

La prueba que excluyó 8 letras del vocabulario resultó **inválida**: quien probaba
ejecutaba mal esas señas, así que la prueba midió a la persona, no al modelo. Al
repetirla con las imágenes de referencia a la vista, C, E, K, M y N funcionan.

La repetición tampoco es concluyente: una persona, informal, sabiendo de antemano
qué letra esperaba. Es la misma clase de evidencia que ya se equivocó una vez,
solo que con el signo cambiado. Este protocolo existe para que el resultado
aguante que te lo revisen en la sustentación.

## Qué se mide

Dos cosas, y la segunda es la que suele faltar:

1. **Falso rechazo** — la persona hace bien la letra y la app la rechaza. Frustra
   y hace abandonar.
2. **Falsa aceptación dirigida** — la persona hace *otra* letra a propósito y la
   app se la da por buena. Es el error grave: una lección que aprueba cualquier
   cosa no enseña nada.

Sin el bloque 2 se mide si la app aprueba, no si enseña.

## Antes de empezar

- **Imagen de referencia siempre visible.** Se prueba la app, no la memoria de la
  persona. Si tiene que recordar la seña, vuelves al error de la prueba anterior.
- **Nadie corrige durante la prueba.** Quien opera no dice "más arriba", "cierra
  más". Se anota y se sigue.
- **Cada persona usa su propia planilla**, con las letras en orden aleatorio
  distinto. En orden alfabético la persona anticipa y practica mientras espera.
- **Mismo dispositivo y misma distancia** para todos, o se anota en `notas` cuando
  cambie. Si una letra falla solo en un teléfono, eso es un hallazgo, no ruido.

## Cuántas personas

Seis personas × tres intentos por letra = **18 observaciones**, que es el mínimo
para que una letra pueda salir *viable*. Con 18 observaciones y ningún fallo, el
intervalo de confianza al 95 % llega hasta 17.6 %, justo por debajo del criterio
del 20 %. Con menos, el intervalo es tan ancho que **ninguna letra puede
resolverse**:

| observaciones | IC95 superior con 0 fallos | ¿puede salir viable? |
|---|---|---|
| 6 | 39.0 % | no |
| 12 | 24.3 % | no |
| 16 | 19.4 % | justo |
| **18** | **17.6 %** | **sí** |
| 24 | 13.8 % | con margen |

Cuantas más personas distintas, mejor: tres intentos de la misma persona están
correlacionados y cuentan menos que tres personas. Si podéis conseguir gente de
fuera del equipo, mejor todavía — el modelo nunca vio a ninguno de vosotros, pero
vosotros ya sabéis cómo hacer las señas, y eso sesga hacia arriba.

## Generar las planillas

```bash
python -m evaluacion_objetivo.prueba_camara plantilla --personas 6 --intentos 3
```

Crea `pruebas_camara/persona_N.csv`, una por persona, con 150 filas cada una:

| columna | qué va |
|---|---|
| `bloque` | `correcta` o `confusion` (ya viene puesto) |
| `letra_objetivo` | la letra que la app pide (ya viene puesta) |
| `sena_a_ejecutar` | **la que la persona debe hacer** (ya viene puesta) |
| `resultado` | `aceptada` o `rechazada` ← lo llenas tú |
| `intentos` | cuántas veces lo intentó antes de ese resultado ← lo llenas tú |
| `notas` | lo que sea raro: mano cortada, contraluz, dedo tapado |

En el bloque `confusion`, `sena_a_ejecutar` es distinta de `letra_objetivo` a
propósito: es la letra con la que el modelo más confunde a la objetivo, sacada de
la co-activación medida sobre los 16 participantes held-out. Si el modelo va a
aceptar una seña equivocada, será esa. **Ahí `aceptada` es un fallo de la app**,
al revés que en el bloque `correcta`.

## Durante la prueba

1. La persona mira la imagen de referencia de `sena_a_ejecutar`.
2. Hace la seña y la sostiene ~1 segundo frente a la cámara.
3. Se anota `aceptada` / `rechazada` y cuántos intentos le costó.
4. Siguiente fila. Sin corregir, sin comentar.

Unos 15–20 minutos por persona.

## Analizar

```bash
python -m evaluacion_objetivo.prueba_camara analizar "pruebas_camara/*.csv"
```

Da por letra el falso rechazo y la falsa aceptación dirigida, cada uno con su
intervalo de confianza, y un veredicto con los mismos criterios que se usaron
sobre el dataset (falso rechazo ≤ 20 %, falsa aceptación ≤ 20 %, decidido con el
intervalo completo):

- **viable** — el intervalo entero está del lado bueno. Puede pasar a
  `verificada_con_camara: true` y entrar al vocabulario.
- **indeterminada** — el intervalo cruza el criterio. No alcanzan los datos; hacen
  falta más personas, no más interpretación.
- **no_viable** — el intervalo entero está del lado malo. Fuera del vocabulario.

## Las tres preguntas que esta prueba tiene que contestar

1. **¿Vuelven C, E, H, K, M, N?** El dataset dice que sí (recall 0.81–1.00). Si la
   prueba lo confirma, el vocabulario pasa de 19 a 25 letras sin entrenar nada.
2. **¿R falla de verdad?** Es la sospechosa principal: recall 0.84 out-of-fold, se
   confunde con S y H en ambas direcciones, y aun con los negativos limpios
   necesita umbral 0.44 y da 37.5 % de falso rechazo. Está en el vocabulario
   activo, así que si falla, es un problema ya desplegado.
3. **¿Las de movimiento aciertan o solo aceptan?** G, H, J, S, Z parecen funcionar
   porque el modelo pilla el frame final. El bloque `confusion` lo distingue: si
   acepta la seña equivocada, no está reconociendo — está admitiendo cualquier
   cosa parecida.

   **Ñ es el caso claro y ya está resuelto:** la Ñ estática es la N
   (co-activación 0.300 y 0.305). Si la app pregunta "¿esto es una Ñ?" y la
   persona hace una N, se la va a dar por buena. Parece acertar justamente porque
   no puede distinguirlas. Está excluida de las planillas junto con Z por ser
   dinámicas.

## Qué hacer con el resultado

Las letras `viable` pasan a `verificada_con_camara: true` en `config_objetivo.json`
y entran al vocabulario de la app. Las demás se quedan fuera, con su número y su
intervalo escritos — que es muy distinto de quedarse fuera "porque no funcionaba".

Y si la prueba se graba con la app, os quedáis además con el set de cámara propio:
datos del dominio correcto, que es lo que hoy no existe.
