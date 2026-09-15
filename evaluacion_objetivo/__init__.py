"""Evaluación dirigida por objetivo para SignaCO.

Post-proceso sobre las probabilidades del modelo ya entrenado: la app siempre
sabe qué letra pidió, así que la pregunta no es "¿cuál de 28 clases es?" sino
"¿esto es la letra objetivo, sí o no?". No se reentrena nada.
"""
