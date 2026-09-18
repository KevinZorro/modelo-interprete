"""Snippet para Colab: exporta los negativos de 'mano en reposo' a un .npz.

Pégalo en una celda NUEVA del notebook v8 DESPUÉS de la celda que construye
`X_neg` (sección 15, la del cupo de HaGRID). No hay que reentrenar nada ni
volver a extraer landmarks del abecedario: solo guardar lo que ya está en RAM.

Luego, en local:
    python -m evaluacion_objetivo.evaluar --negativos negativos_reposo.npz

Sin este archivo los umbrales de config_objetivo.json son PROVISIONALES: no hay
forma de comprobar que un umbral bajo no apruebe a un usuario que simplemente
dejó la mano quieta, que es el fallo que rompe la confianza en la app.
"""
import numpy as np

RUTA = "/content/drive/MyDrive/LSC70/negativos_reposo.npz"

np.savez_compressed(
    RUTA,
    X_neg=X_neg.astype(np.float32),                     # noqa: F821  (viene del notebook)
    clases=np.array(CLASES_NEG_FACILES + CLASES_NEG_DIFICILES),  # noqa: F821
)
print(f"Guardado {RUTA}: {X_neg.shape[0]} negativos x {X_neg.shape[1]} floats")  # noqa: F821
