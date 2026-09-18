"""Pruebas mínimas de la lógica de decisión. `python -m evaluacion_objetivo.pruebas`

No prueban el modelo (eso lo hace la evaluación), sino que la regla haga lo que
dice: que el puntaje sume el grupo de equivalencia, que el tope de falsa
aceptación mande sobre el presupuesto de falso rechazo, y que un veredicto solo
se emita cuando el intervalo de confianza entero cae de un lado.
"""
import numpy as np

from .calibracion import clasificar_estado, elegir_umbral
from .metricas import acepta_k_de_n, puntaje_objetivo, wilson


def prueba_wilson():
    p, lo, hi = wilson(0, 16)
    assert p == 0 and lo == 0 and 0.19 < hi < 0.20, (p, lo, hi)
    # Con n grande el intervalo se cierra: es lo que hace decidible la FA cruzada.
    _, lo2, hi2 = wilson(0, 416)
    assert hi2 < 0.01


def prueba_equivalencias():
    P = np.array([[0.4, 0.3, 0.3]])
    assert puntaje_objetivo(P, 0)[0] == 0.4
    # Con grupo de equivalencia se SUMA la masa repartida entre la misma pose.
    assert abs(puntaje_objetivo(P, 0, {0: [1]})[0] - 0.7) < 1e-6


def prueba_regla_umbral():
    pos = np.linspace(0.3, 1.0, 100)      # ejecuciones correctas, puntaje alto
    neg = np.linspace(0.0, 0.3, 100)      # otras letras, puntaje bajo
    u, motivo = elegir_umbral(pos, neg, presupuesto_frr=0.05, tope_fa=0.20)
    # El más alto que respeta el presupuesto de falso rechazo: máximo margen.
    assert motivo == "frr_y_fa" and 0.30 <= u <= 0.34, (u, motivo)

    # Si las otras letras puntúan alto, el tope de FA obliga a subir el umbral
    # aunque el presupuesto de falso rechazo pida uno más bajo.
    u2, motivo2 = elegir_umbral(pos, np.linspace(0.5, 1.0, 100), 0.05, 0.20)
    assert motivo2 == "tope_fa" and u2 > u, (u2, motivo2)

    # Positivos con un 10 % de frames basura: ningún umbral respeta el
    # presupuesto de falso rechazo, y la regla debe avisarlo en vez de callarlo.
    malos = np.concatenate([np.zeros(10), np.linspace(0.5, 1.0, 90)])
    u3, motivo3 = elegir_umbral(malos, neg, 0.05, 0.20)
    assert motivo3 == "frr_inalcanzable", motivo3


def prueba_estados():
    viable = {"frr": (0.0, 0.0, 0.19), "fa_cruzada": (0.0, 0.0, 0.01)}
    assert clasificar_estado(viable, 0.20, 0.20) == "viable"
    # IC que cruza el criterio: n insuficiente, no se fuerza veredicto.
    dudosa = {"frr": (0.06, 0.01, 0.28), "fa_cruzada": (0.0, 0.0, 0.01)}
    assert clasificar_estado(dudosa, 0.20, 0.20) == "indeterminada"
    mala = {"frr": (0.5, 0.28, 0.72), "fa_cruzada": (0.0, 0.0, 0.01)}
    assert clasificar_estado(mala, 0.20, 0.20) == "no_viable"


def prueba_k_de_n():
    frames = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    assert acepta_k_de_n(frames, 0, umbral=0.5, k=2)        # 2 de 3 pasan
    assert not acepta_k_de_n(frames, 0, umbral=0.5, k=3)


if __name__ == "__main__":
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("prueba_"):
            fn()
            print(f"ok  {nombre}")
    print("\nTodas las pruebas pasaron.")
