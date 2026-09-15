"""Calibración del umbral por letra objetivo, sin reentrenar y sin hacer trampa.

El umbral no puede elegirse y medirse sobre los mismos datos: saldría optimista.
Aquí se usa leave-one-participant-out sobre los 16 participantes held-out:
para cada persona, el umbral se fija con las otras 15 y se evalúa con ella.
Ninguna persona influye en el umbral con el que se la juzga.

POR QUÉ EL FRR SE CALIBRA CON FRAMES Y NO CON TOMAS
---------------------------------------------------
Con 15 tomas de calibración y un presupuesto del 5 %, "el umbral más alto que no
rechaza ninguna" es exactamente el mínimo de esas 15: un estadístico de orden
extremo. En LOPO eso hace fallar siempre a la toma de menor puntaje y produce un
FRR artificial de 1/16 = 6.2 % en casi todas las letras. El percentil se estima
sobre los ~90 frames de esas 15 personas, que es donde hay datos suficientes, y
la DECISIÓN se sigue evaluando por toma (promedio de frames), que es lo que hace
la app. El umbral así calibrado es conservador para la decisión por toma, porque
el promedio de la toma es más alto que sus frames sueltos.

Regla de selección, en este orden de prioridad:

1. La falsa aceptación cruzada es un TOPE DURO. Si la lección de la X aprueba
   más del 20 % de las ejecuciones de otras letras, no está enseñando nada.
2. Dentro de lo que cumple ese tope, se toma el umbral MÁS ALTO que respeta el
   presupuesto de falso rechazo (deja el mayor margen posible contra la falsa
   aceptación sin frustrar al usuario).
3. Si ningún umbral cumple las dos cosas, gana el tope de falsa aceptación y el
   falso rechazo sube. Se reporta explícitamente: es una letra en aprietos.
"""
import numpy as np

from .metricas import fa_cruzada, frr, puntaje_objetivo, wilson

REJILLA = np.round(np.arange(0.01, 1.00, 0.01), 2)


def elegir_umbral(s_pos, s_neg, presupuesto_frr, tope_fa,
                  s_reposo=None, tope_reposo=None):
    """(umbral, motivo) a partir de los puntajes de calibración ya calculados.

    s_pos    : puntajes de ejecuciones CORRECTAS de la letra objetivo (frames).
    s_neg    : puntajes de ejecuciones de OTRAS letras (tomas) -> tope de FA cruzada.
    s_reposo : puntajes de mano en reposo (negativos de HaGRID), si se exportaron.
               Es la restricción que impide que un umbral bajo apruebe a alguien
               que simplemente dejó la mano quieta. Sin este archivo la
               restricción no se puede aplicar y el umbral queda provisional.
    """
    frr_u = np.array([(s_pos < u).mean() for u in REJILLA])
    admisible = np.array([(s_neg >= u).mean() for u in REJILLA]) <= tope_fa
    if s_reposo is not None and tope_reposo is not None:
        admisible &= np.array([(s_reposo >= u).mean() for u in REJILLA]) <= tope_reposo

    i_frr = np.where(frr_u <= presupuesto_frr)[0]
    i_ok = np.where(admisible)[0]
    if not len(i_ok):
        # Ni con el umbral más alto se contiene la falsa aceptación: la letra es
        # inviable, pero se devuelve el máximo para que las métricas lo muestren.
        return float(REJILLA[-1]), "fa_incontenible"
    ambos = np.intersect1d(i_frr, i_ok)
    if len(ambos):
        # El más alto de los que cumplen ambas: máximo margen contra la FA.
        return float(REJILLA[ambos[-1]]), "frr_y_fa"
    if not len(i_frr):
        # Ningún umbral respeta el presupuesto de falso rechazo: la letra tiene
        # frames malos hasta con el umbral mínimo. Se elige el más permisivo que
        # sí contiene la falsa aceptación y se marca para que no pase inadvertido.
        return float(REJILLA[i_ok[0]]), "frr_inalcanzable"
    # El tope de falsa aceptación manda sobre el presupuesto de falso rechazo.
    return float(REJILLA[i_ok[0]]), "tope_fa"


def _puntajes_calibracion(d, objetivo, mascara_tomas, equivalencias):
    """Puntajes de calibración: positivos por frame, negativos por toma."""
    y, frames = d["y_toma"], d["P_frames_por_toma"]
    pos = [puntaje_objetivo(frames[j], objetivo, equivalencias)
           for j in np.where(mascara_tomas & (y == objetivo))[0]]
    equivalentes = list((equivalencias or {}).get(objetivo, []))
    m_neg = mascara_tomas & (y != objetivo)
    if equivalentes:
        m_neg &= ~np.isin(y, equivalentes)
    s_neg = puntaje_objetivo(d["P_toma"][m_neg], objetivo, equivalencias)
    return np.concatenate(pos) if pos else np.zeros(0), s_neg


def umbral_para(d, objetivo, mascara_tomas, presupuesto_frr, tope_fa,
                equivalencias, P_reposo=None, tope_reposo=None):
    """Umbral calibrado usando SOLO las tomas marcadas en `mascara_tomas`."""
    s_pos, s_neg = _puntajes_calibracion(d, objetivo, mascara_tomas, equivalencias)
    s_rep = None if P_reposo is None else puntaje_objetivo(P_reposo, objetivo,
                                                           equivalencias)
    return elegir_umbral(s_pos, s_neg, presupuesto_frr, tope_fa, s_rep, tope_reposo)


def umbrales_lopo(d, presupuesto_frr, tope_fa, equivalencias=None,
                  P_reposo=None, tope_reposo=None):
    """Métricas leave-one-participant-out + umbral final para exportar.

    Por cada letra devuelve:
      - `umbral`: el calibrado con los 16 participantes (el que se exporta).
      - `frr` / `fa_cruzada`: medidos out-of-fold, con IC de Wilson. Son las
        cifras honestas, porque cada toma se juzgó con un umbral que no la vio.
      - `umbrales_lopo`: dispersión del umbral entre folds. Si varía mucho,
        el umbral es inestable y la letra no es de fiar aunque su FRR salga bien.
    """
    P, y, part = d["P_toma"], d["y_toma"], d["part_toma"]
    personas = sorted(set(part))
    todas = np.ones(len(y), bool)
    resultado = {}

    for i, letra in enumerate(d["etiquetas"]):
        rechazos = aceptadas = n_pos = n_neg = 0
        us = []
        for persona in personas:
            fuera = part == persona
            u, _ = umbral_para(d, i, ~fuera, presupuesto_frr, tope_fa,
                               equivalencias, P_reposo, tope_reposo)
            us.append(u)

            # Evaluación sobre la persona excluida, con ESE umbral.
            f_p, _, _, n_p = frr(P[fuera], y[fuera], i, u, equivalencias)
            a_p, _, _, n_a = fa_cruzada(P[fuera], y[fuera], i, u, equivalencias)
            rechazos += 0 if np.isnan(f_p) else round(f_p * n_p)
            aceptadas += 0 if np.isnan(a_p) else round(a_p * n_a)
            n_pos, n_neg = n_pos + n_p, n_neg + n_a

        u_final, motivo = umbral_para(d, i, todas, presupuesto_frr, tope_fa,
                                      equivalencias, P_reposo, tope_reposo)
        resultado[letra] = {
            "umbral": u_final,
            "motivo_umbral": motivo,
            "frr": wilson(int(rechazos), int(n_pos)),
            "fa_cruzada": wilson(int(aceptadas), int(n_neg)),
            "n_tomas": int(n_pos),
            "n_tomas_otras": int(n_neg),
            "umbrales_lopo": [float(min(us)), float(np.median(us)), float(max(us))],
        }
    return resultado


def clasificar_estado(m, frr_max, fa_max):
    """Veredicto por letra, respetando la incertidumbre de n=16.

    Con 16 tomas, el IC95 de un 0/16 llega hasta 19.4 %: hay letras sobre las que
    estos datos SIMPLEMENTE NO ALCANZAN para decidir. Esas se marcan
    `indeterminada` en vez de forzar un veredicto — es mejor recuperar tres
    letras con confianza que seis dudosas.
    """
    _, frr_lo, frr_hi = m["frr"]
    _, fa_lo, fa_hi = m["fa_cruzada"]
    if frr_lo > frr_max or fa_lo > fa_max:
        return "no_viable"          # el IC entero está del lado malo
    if frr_hi <= frr_max and fa_hi <= fa_max:
        return "viable"             # el IC entero está del lado bueno
    return "indeterminada"          # el IC cruza el criterio: n insuficiente
