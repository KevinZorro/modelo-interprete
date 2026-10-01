# Correr la comparación ANH vs AN en Colab

Las imágenes están en tu Drive y no se suben al repo (son datos personales: Ley 1581), así
que esto se corre en Colab. Cada paso imprime algo que hay que mirar antes del siguiente.

```python
# 0. Preparar
!git clone https://github.com/KevinZorro/modelo-interprete && cd modelo-interprete && git checkout <rama>
%cd modelo-interprete
!pip -q install mediapipe
!wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
from google.colab import drive; drive.mount('/content/drive')
D = "/content/drive/MyDrive/LSC70"
```

## 1. Estructura (tarea 1) — NO asumas, mira
```python
!python -m lsc70.explorar_estructura --fuente {D}/LSC70AN.zip  --nombre an
!python -m lsc70.explorar_estructura --fuente {D}/LSC70ANH.zip --nombre anh
```
Revisa el árbol, los conteos por clase y los tamaños de imagen. Si son correctos:
```python
!sed -i 's/"confirmado": false/"confirmado": true/' configs/estructura_an.json configs/estructura_anh.json
```
Sin esto, los pasos siguientes se niegan a correr.

## 2. Píxeles de mano (tarea 2) — ~5 min
```python
!python -m lsc70.medir_mano --anh {D}/LSC70ANH.zip --an {D}/LSC70AN.zip --modelo hand_landmarker.task
```
Imprime el tamaño de la mano en píxeles nativos de cada sección, pareado imagen a imagen,
y un veredicto de cuál tiene más detalle.

## 3-4. Extraer landmarks y descarte — ~15 min cada una
```python
!python -m lsc70.extraer --fuente {D}/LSC70ANH.zip --nombre anh --modelo hand_landmarker.task \
    --salida landmarks_anh.npz --contrastar-cache landmarks_cache.npz
!python -m lsc70.extraer --fuente {D}/LSC70AN.zip  --nombre an  --modelo hand_landmarker.task \
    --salida landmarks_an.npz
```
La primera comprueba que este pipeline es el del notebook: compara con el cache viejo y
la **mediana de la diferencia debe ser ≈ 0**. Si sale > 1e-2, para y avísame.

## 5-6. Entrenar y evaluar (letras + números) — ~20 min por experimento con 3 semillas
```python
!python -m lsc70.entrenar_evaluar --exp anh_ln --npz anh=landmarks_anh.npz --entrenar-con anh --evaluar-en anh
!python -m lsc70.entrenar_evaluar --exp an_ln  --npz an=landmarks_an.npz   --entrenar-con an  --evaluar-en an
!python -m lsc70.entrenar_evaluar --exp comb_ln --npz anh=landmarks_anh.npz an=landmarks_an.npz \
    --entrenar-con anh an --evaluar-en an
```
Para comparar con el 88.8 % (solo letras), añade `--clases letras` a un experimento.

## 7. Decidir
```python
!python -m lsc70.comparar --base resultados/anh_ln --otros resultados/an_ln resultados/comb_ln
```
Sube `resultados/*/metricas.json` y `configs/estructura_*.json` al repo (no los `.npz`
con landmarks si prefieres no subirlos; las imágenes nunca).
