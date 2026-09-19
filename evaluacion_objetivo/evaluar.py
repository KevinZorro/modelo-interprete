"""CLI principal: evalúa el modo objetivo y exporta config_objetivo.json.

Uso:
    python -m evaluacion_objetivo.evaluar [--negativos ruta.npz]

No reentrena nada: corre el .tflite desplegado sobre los 16 participantes que
ese modelo nunca vio, y decide con post-proceso sobre sus probabilidades.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import datos as mod_datos
from .calibracion import clasificar_estado, umbral_para, umbrales_lopo
from .metricas import acepta_k_de_n, fa_reposo, wilson

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CACHE = RAIZ / "landmarks_cache.npz"
RUTA_MODELO = RAIZ / "signaco_abecedario_rechazo (1).tflite"
RUTA_CONFIG = RAIZ / "config_evaluacion.json"
RUTA_SALIDA = RAIZ / "config_objetivo.json"
RUTA_CONFIG_SALIDA = RUTA_SALIDA   # nombre explícito para otros módulos


def metricas_k_de_n(d, cfg, equivalencias, P_neg=None):
    """Igual que la evaluación por toma, pero exigiendo k frames sobre el umbral.

    Es lo que la app hace de verdad: mira ~6 frames en medio segundo y decide.
    Se calibra con el mismo esquema LOPO para no medir sobre datos propios.
    """
    P, y, part = d["P_toma"], d["y_toma"], d["part_toma"]
    frames = d["P_frames_por_toma"]
    personas = sorted(set(part))
    k = cfg["k_de_n"]
    salida = {}

    for i, letra in enumerate(d["etiquetas"]):
        rech = n_pos = acep = n_neg = 0
        for persona in personas:
            fuera = part == persona
            u, _ = umbral_para(d, i, ~fuera, cfg["presupuesto_frr"],
                               cfg["tope_fa_cruzada"], equivalencias, P_neg,
                               cfg.get("tope_fa_reposo"))
            for j in np.where(fuera)[0]:
                acepta = acepta_k_de_n(frames[j], i, u, k, equivalencias)
                if y[j] == i:
                    n_pos += 1
                    rech += not acepta
                else:
                    n_neg += 1
                    acep += acepta
        salida[letra] = {"frr": wilson(rech, n_pos), "fa_cruzada": wilson(acep, n_neg)}
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--negativos", default=None,
                    help=".npz con landmarks de mano en reposo (X_neg) exportado de Colab")
    args = ap.parse_args()

    cfg = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    equivalencias = {k: v for k, v in cfg["grupos_equivalencia"].items()
                     if not k.startswith("_")}

    d = mod_datos.datos_evaluacion(RUTA_CACHE, RUTA_MODELO)
    P_neg = mod_datos.cargar_negativos(args.negativos, RUTA_MODELO,
                                       cfg.get("clases_negativas_excluidas", ()))

    print(f"Tomas held-out: {len(d['y_toma'])} | participantes: "
          f"{len(set(d['part_toma']))} (ninguno visto en entrenamiento)")
    acc = (d["P_toma"][:, :27].argmax(1) == d["y_toma"]).mean()
    print(f"Referencia — exactitud por toma en modo 27 clases: {acc:.3f}\n")

    res = umbrales_lopo(d, cfg["presupuesto_frr"], cfg["tope_fa_cruzada"],
                        equivalencias, P_neg, cfg.get("tope_fa_reposo"))
    res_k = metricas_k_de_n(d, cfg, equivalencias, P_neg)
    crit = cfg["criterio_viabilidad"]

    excl = cfg["excluidas_actuales"]
    fuera_por_confusion = set(excl["por_confusion"])
    dinamicas = set(excl["dinamicas"])

    print(f"{'letra':>5} {'umbral':>7} {'FRR (IC95)':>22} {'FA cruzada (IC95)':>24} "
          f"{'FRR k=2':>9} {'estado':>15}")
    letras = {}
    for letra, m in res.items():
        estado = clasificar_estado(m, crit["frr_max"], crit["fa_cruzada_max"])
        if letra in dinamicas:
            # El modelo es de poses estáticas: ninguna métrica puede rescatarlas.
            estado = "no_viable_dinamica"
        f, flo, fhi = m["frr"]
        a, alo, ahi = m["fa_cruzada"]
        fk = res_k[letra]["frr"][0]
        print(f"{letra:>5} {m['umbral']:>7.2f} "
              f"{100*f:>6.1f}% [{100*flo:>5.1f}-{100*fhi:>5.1f}] "
              f"{100*a:>7.1f}% [{100*alo:>5.1f}-{100*ahi:>5.1f}] "
              f"{100*fk:>8.1f}% {estado:>15}")

        reposo = fa_reposo(P_neg, d["etiquetas"].index(letra), m["umbral"], equivalencias)
        letras[letra] = {
            "umbral": m["umbral"],
            "motivo_umbral": m["motivo_umbral"],
            "grupo_equivalencia": equivalencias.get(letra, []),
            "estado": estado,
            "en_vocabulario_activo": letra in cfg["vocabulario_activo_actual"],
            "verificada_con_camara": False,
            "metricas": {
                "protocolo": "leave-one-participant-out sobre 16 participantes held-out, por toma",
                "n_tomas": m["n_tomas"],
                "frr": round(f, 4),
                "frr_ic95": [round(flo, 4), round(fhi, 4)],
                "fa_cruzada": round(a, 4),
                "fa_cruzada_ic95": [round(alo, 4), round(ahi, 4)],
                "n_tomas_otras_letras": m["n_tomas_otras"],
                "frr_k_de_n": round(res_k[letra]["frr"][0], 4),
                "fa_cruzada_k_de_n": round(res_k[letra]["fa_cruzada"][0], 4),
                "fa_reposo": None if reposo is None else round(reposo[0], 4),
                "estado_fa_reposo": "pendiente_export_negativos" if reposo is None else "medida",
                "umbral_lopo_min_mediana_max": m["umbrales_lopo"],
            },
        }

    recuperables = sorted(l for l in fuera_por_confusion
                          if letras[l]["estado"] == "viable")
    indeterminadas = sorted(l for l in fuera_por_confusion
                            if letras[l]["estado"] == "indeterminada")

    salida = {
        "modo": "objetivo",
        "version_modelo": RUTA_MODELO.name,
        "indice_rechazo": mod_datos.IDX_RECHAZO,
        "etiquetas": d["etiquetas"],
        "k_de_n": cfg["k_de_n"],
        "criterio_viabilidad": crit,
        "vocabulario_activo": cfg["vocabulario_activo_actual"],
        "vocabulario_propuesto": recuperables,
        "indeterminadas": indeterminadas,
        "umbrales_provisionales": P_neg is None,
        "advertencia_umbrales_provisionales": (
            "Si umbrales_provisionales es true, los umbrales se calibraron SIN "
            "negativos de mano en reposo: no está comprobado que un umbral bajo "
            "no apruebe a un usuario que dejó la mano quieta. Reejecutar con "
            "--negativos antes de usarlos en la app."
        ) if P_neg is None else None,
        "advertencia": (
            "Las letras de 'vocabulario_propuesto' son HIPÓTESIS medidas sobre "
            "dataset. Ninguna entra al vocabulario de la app mientras "
            "verificada_con_camara sea false, pero por falta de una prueba "
            "válida, no por sospecha de que el modelo falle: la prueba con "
            "cámara que excluyó C, E, H, K, M, N resultó inválida (quien probaba "
            "hacía mal las seña), y al repetirla ejecutándolas bien esas letras "
            "funcionan. Hace falta una prueba con protocolo: varias personas, "
            "imagen de referencia a la vista, orden aleatorio, y ejecuciones "
            "incorrectas a propósito para comprobar que se rechazan."
        ),
        "letras": letras,
    }
    RUTA_SALIDA.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRecuperables con confianza : {recuperables or 'ninguna'}")
    print(f"Indeterminadas (n insufic.): {indeterminadas or 'ninguna'}")
    print(f"\nEscrito: {RUTA_SALIDA}")
    if P_neg is None:
        print("FA de reposo: PENDIENTE — faltan los negativos (--negativos X_neg.npz)")


if __name__ == "__main__":
    main()
