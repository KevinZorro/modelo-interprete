"""Métricas de la decisión binaria, todas con intervalo de confianza.

Con 16 tomas por letra (una por participante held-out) cualquier porcentaje
tiene un intervalo ancho. Reportar el punto sin el intervalo sería fingir una
precisión que no existe, así que todo pasa por Wilson.
"""
import math

import numpy as np


def wilson(exitos, n, z=1.96):
    """IC de Wilson al 95 %. Preferido sobre el normal porque no se rompe con
    proporciones de 0 o 1, que es justo lo que pasa con n=16."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = exitos / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centro - margen), min(1.0, centro + margen))


def puntaje_objetivo(P, objetivo, grupos_equivalencia=None):
    """Probabilidad de que la seña sea la letra objetivo.

    Si el objetivo tiene grupo de equivalencia (misma configuración de mano con
    otro nombre), se SUMAN las probabilidades del grupo: el modelo reparte masa
    entre clases que en realidad son la misma pose y penalizarlo por eso sería
    contar como error algo bien hecho.
    """
    columnas = [objetivo] + list((grupos_equivalencia or {}).get(objetivo, []))
    return P[:, columnas].sum(axis=1)


def frr(P, y, objetivo, umbral, equivalencias=None):
    """Falso rechazo: el usuario hizo bien la letra objetivo y no se la aceptan.

    Es el error caro en una app de aprendizaje: frustra y hace abandonar.
    """
    m = y == objetivo
    if m.sum() == 0:
        return (float("nan"), 0.0, 1.0, 0)
    s = puntaje_objetivo(P[m], objetivo, equivalencias)
    p, lo, hi = wilson(int((s < umbral).sum()), int(m.sum()))
    return (p, lo, hi, int(m.sum()))


def fa_cruzada(P, y, objetivo, umbral, equivalencias=None):
    """Falsa aceptación cruzada: se pide la letra X, el usuario hace OTRA letra
    y el sistema la aprueba.

    Es el error pedagógico grave: la lección deja de enseñar. Se mide pasando
    las tomas reales de las otras 26 letras como si el objetivo fuera X.
    Las letras del grupo de equivalencia del objetivo se excluyen del conteo:
    son la misma pose, aceptarlas es correcto por definición.
    """
    equivalentes = set((equivalencias or {}).get(objetivo, []))
    m = (y != objetivo) & ~np.isin(y, list(equivalentes)) if equivalentes else (y != objetivo)
    if m.sum() == 0:
        return (float("nan"), 0.0, 1.0, 0)
    s = puntaje_objetivo(P[m], objetivo, equivalencias)
    p, lo, hi = wilson(int((s >= umbral).sum()), int(m.sum()))
    return (p, lo, hi, int(m.sum()))


def fa_reposo(P_neg, objetivo, umbral, equivalencias=None):
    """Falsa aceptación de mano en reposo: el usuario no hace nada y se le
    aprueba la letra. Rompe la confianza en la app más rápido que cualquier
    otro fallo. Devuelve None si no hay negativos exportados."""
    if P_neg is None:
        return None
    s = puntaje_objetivo(P_neg, objetivo, equivalencias)
    return wilson(int((s >= umbral).sum()), len(P_neg))


def acepta_k_de_n(P_frames, objetivo, umbral, k, equivalencias=None):
    """Regla temporal: aceptar si k de los n frames de la toma pasan el umbral.

    La app ve ~6 frames en medio segundo; exigir varios frames filtra el frame
    suelto afortunado sin volver la lección más estricta de lo necesario.
    """
    s = puntaje_objetivo(P_frames, objetivo, equivalencias)
    return int((s >= umbral).sum()) >= k
