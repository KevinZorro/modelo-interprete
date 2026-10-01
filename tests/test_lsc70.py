"""Pruebas del pipeline lsc70. `pytest tests/ -q` (las lentas requieren TensorFlow)."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lsc70 import dataset, splits
from lsc70.normalizacion import preparar_imagen_anh, vector_features

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "modeloparaLSC70_v8 (1).ipynb"


def _lm(pts):
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]


def _mano(rng):
    return rng.normal(size=(21, 3)).astype(np.float32) * 0.1 + 0.5


# --- Normalización: el riesgo silencioso del proyecto --------------------------------
def test_vector_features_es_copia_textual_del_notebook():
    """Si alguien edita la normalización a mano, esta prueba falla."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    fuente = next(
        "".join(c["source"])
        for c in nb["cells"]
        if "def vector_features" in "".join(c["source"])
    )
    esperado = fuente[fuente.index("def vector_features") :].rstrip()
    modulo = (RAIZ / "lsc70" / "normalizacion.py").read_text(encoding="utf-8")
    assert esperado in modulo


def test_normalizacion_invariantes():
    rng = np.random.default_rng(0)
    pts = _mano(rng)
    v, extras = vector_features(_lm(pts), es_izquierda=False)
    p = v.reshape(21, 3)
    assert np.allclose(p[0], 0)  # muñeca al origen
    assert np.isclose(
        np.linalg.norm(p, axis=1).max(), 1.0, atol=1e-5
    )  # escala unitaria
    # invariante a traslación y a escala uniforme de la mano
    v2, _ = vector_features(_lm(pts * 3.0 + 0.2), es_izquierda=False)
    assert np.allclose(v, v2, atol=1e-5)
    # espejo: mano izquierda == derecha con X invertida
    vi, _ = vector_features(_lm(pts), es_izquierda=True)
    esperado = v.reshape(21, 3).copy()
    esperado[:, 0] *= -1
    assert np.allclose(vi.reshape(21, 3), esperado, atol=1e-6)


def test_mano_degenerada_devuelve_none():
    v, e = vector_features(_lm(np.zeros((21, 3))), es_izquierda=False)
    assert v is None and e is None


def test_preprocesado_anh_es_un_lienzo_de_480_y_no_encoge_ans():
    img = np.full((120, 120, 3), 200, np.uint8)
    lienzo, (escala, x0, y0) = preparar_imagen_anh(img)
    assert lienzo.shape == (480, 480, 3) and np.isclose(escala, 2.4)
    assert (x0, y0) == (96, 96)


# --- Splits ------------------------------------------------------------------------
def test_splits_reproducen_la_lista_de_test_del_notebook():
    s = splits.construir([f"Per{i:02d}" for i in range(1, 71)])
    assert s["test"] == [
        "Per05",
        "Per08",
        "Per18",
        "Per19",
        "Per25",
        "Per27",
        "Per33",
        "Per38",
        "Per41",
        "Per55",
        "Per60",
        "Per62",
        "Per63",
        "Per64",
        "Per70",
    ]
    assert not set(s["test"]) & set(s["train"])
    assert sorted(np.bincount(list(s["fold_de"].values()))) == [14] * 5


def test_splits_rechazan_participantes_desconocidos(tmp_path):
    ruta = tmp_path / "s.json"
    splits.guardar(
        splits.construir(["Per01", "Per02", "Per03"], n_test=1, n_folds=3), ruta
    )
    with pytest.raises(SystemExit):
        splits.cargar(["Per01", "PerNUEVO"], ruta)


# --- Exploración -------------------------------------------------------------------
def _zip_sintetico(ruta, raiz="LSC70AN"):
    with zipfile.ZipFile(ruta, "w") as z:
        for part in ("Per01", "Per02"):
            for cls in ("A", "1"):
                for k in range(6):
                    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
                    z.writestr(
                        f"{raiz}/{part}/{cls}/{part}_{cls}_{k}.jpg", buf.tobytes()
                    )
        z.writestr(f"{raiz}/LEEME.txt", "hola")
    return ruta


def test_parsear_y_raiz(tmp_path):
    f = dataset.Fuente(_zip_sintetico(tmp_path / "x.zip"))
    assert f.raiz == "LSC70AN" and len(f.imagenes()) == 24
    assert dataset.parsear(f.relativa(f.imagenes()[0])) == ("Per01", "1", 0)
    assert dataset.parsear("suelto.jpg") is None


def test_explorar_escribe_config_sin_confirmar_y_extraer_se_niega(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(dataset, "DIR_CONFIGS", tmp_path / "configs")
    dataset.explorar(_zip_sintetico(tmp_path / "x.zip"), "an")
    cfg = json.loads((tmp_path / "configs" / "estructura_an.json").read_text())
    assert cfg["confirmado"] is False and cfg["n_participantes"] == 2
    assert cfg["frames_por_toma"] == {"6": 4} and cfg["tamanos_muestra"] == {
        "64x48": 24
    }
    assert cfg["preprocesado"] == "nativo"
    with pytest.raises(SystemExit):  # la salvaguarda funciona
        dataset.cargar_estructura("an")


# --- Extracción con la detección sustituida (prueba la lógica, no MediaPipe) ----------
def test_extraer_cuenta_descarte(tmp_path, monkeypatch):
    from lsc70 import extraer

    monkeypatch.setattr(dataset, "DIR_CONFIGS", tmp_path / "configs")
    z = _zip_sintetico(tmp_path / "x.zip")
    cfg = {
        "idx_participante": 0,
        "idx_clase": 1,
        "preprocesado": "nativo",
        "n_imagenes": 24,
    }
    llamadas = {"n": 0}

    def falso(detector, img):  # las 6 últimas imágenes (Per02/A) sin mano
        llamadas["n"] += 1
        return (
            None
            if llamadas["n"] > 18
            else (np.zeros(63, np.float32), np.zeros(3, np.float32), None, False)
        )

    monkeypatch.setattr(extraer, "landmarks_de_imagen", falso)
    datos, en_disco, ret, ilegibles = extraer.extraer(
        dataset.Fuente(z), cfg, None, reporte=0
    )
    assert sum(en_disco.values()) == 24 and sum(ret.values()) == 18 and ilegibles == 0
    assert list(datos["tomas_perdidas"]) == ["Per02__A"]
    assert datos["X"].shape == (18, 63)


# --- Comparador --------------------------------------------------------------------
def test_bootstrap_de_participantes_detecta_diferencia_real_y_ausencia():
    from lsc70.comparar import bootstrap_participantes

    rng = np.random.default_rng(0)
    part = np.repeat([f"P{i}" for i in range(40)], 6)
    ok_a = rng.random(240) < 0.80
    ok_b = ok_a | (rng.random(240) < 0.5)  # b claramente mejor
    d, lo, hi = bootstrap_participantes(ok_a, ok_b, part, n=500)
    assert lo > 0
    d0, lo0, hi0 = bootstrap_participantes(ok_a, ok_a.copy(), part, n=500)
    assert d0 == 0 and lo0 <= 0 <= hi0


# --- Entrenamiento (lento, necesita TensorFlow) --------------------------------------
@pytest.mark.slow
def test_entrenar_evaluar_de_punta_a_punta_con_datos_separables(tmp_path):
    pytest.importorskip("tensorflow")
    from lsc70 import entrenar_evaluar

    rng = np.random.default_rng(0)
    partes = [f"Per{i:02d}" for i in range(1, 21)]
    X, y, p, nom = [], [], [], []
    centros = rng.normal(size=(4, 63)) * 2
    for part in partes:
        for k, cls in enumerate(["A", "B", "C", "D"]):
            for fr in range(6):
                X.append(centros[k] + rng.normal(size=63) * 0.3)
                y.append(cls)
                p.append(part)
                nom.append(f"{part}__{cls}__{part}_{cls}_{fr}.jpg")
    npz = tmp_path / "s.npz"
    np.savez(
        npz,
        X=np.array(X, np.float32),
        y=np.array(y),
        participante=np.array(p),
        nombre=np.array(nom),
    )
    splits.RUTA = tmp_path / "splits.json"
    args = SimpleNamespace(
        exp="t",
        npz=[f"s={npz}"],
        entrenar_con=["s"],
        evaluar_en=["s"],
        clases="letras",
        clases_lista="A,B,C,D",
        incluir_palabras=False,
        semillas=[0],
        salida=str(tmp_path / "res"),
    )
    entrenar_evaluar.EPOCHS_MAX = 15
    entrenar_evaluar.correr(args)
    m = json.loads((tmp_path / "res" / "t" / "metricas.json").read_text())
    assert m["n_tomas"] == 80 and m["top1_estricto_ensamble"] > 0.95


# --- Camino real de MediaPipe (solo si hay modelo .task; no detecta manos aquí) ------
@pytest.mark.skipif(
    "LSC70_MODELO" not in __import__("os").environ,
    reason="define LSC70_MODELO=ruta/hand_landmarker.task",
)
def test_mediapipe_real_sin_mano_se_descarta_sin_romper(tmp_path, monkeypatch):
    """Imagen sin mano: crear_detector, el preprocesado y el descarte funcionan de punta
    a punta con la librería real, tanto en 'anh' como en 'nativo'. NO prueba que detecte
    manos de verdad: eso solo se puede ver con imágenes reales del dataset."""
    import os
    from lsc70 import extraer, medir_mano
    from lsc70.normalizacion import crear_detector, landmarks_de_imagen

    det = crear_detector(os.environ["LSC70_MODELO"], num_hands=1)
    assert landmarks_de_imagen(det, np.full((480, 640, 3), 90, np.uint8)) is None
    z = _zip_sintetico(tmp_path / "x.zip")
    for prep in ("nativo", "anh"):
        cfg = {
            "idx_participante": 0,
            "idx_clase": 1,
            "preprocesado": prep,
            "n_imagenes": 24,
        }
        datos, disco, ret, _ = extraer.extraer(dataset.Fuente(z), cfg, det, reporte=0)
        assert sum(disco.values()) == 24 and sum(ret.values()) == 0
        assert len(datos["tomas_perdidas"]) == 4  # sin mano: todas las tomas se pierden
        f = dataset.Fuente(z)
        r = medir_mano.medir_imagen(det, f, f.imagenes()[0], cfg)
        assert r["detectada"] is False and r["tam_original"] == [64, 48]
    det.close()


# --- Diagnóstico de la mano ----------------------------------------------------------
def test_diagnostico_detecta_mano_equivocada_y_calibra_el_ruido():
    from lsc70.diagnostico_mano import evaluar_imagen, resumir

    rng = np.random.default_rng(0)
    ref = rng.normal(size=63)
    otra = rng.normal(size=63) * 2  # otra mano: otra pose
    # elegida correcta (ruido pequeño) vs elegida equivocada con la buena disponible
    ok = evaluar_imagen(ref, ref + 0.01, [ref + 0.01, otra])
    mal = evaluar_imagen(ref, otra, [otra, ref + 0.01])
    assert ok["elegida_es_la_mejor"] and not mal["elegida_es_la_mejor"]

    filas = []
    for i in range(60):  # 60 imágenes con una sola mano: definen el ruido
        f = evaluar_imagen(ref, ref + rng.normal(size=63) * 0.01, [ref])
        f["clave"] = ["P", "A", i]
        filas.append(f)
    for i in range(10):  # 10 con dos manos, todas mal elegidas
        f = evaluar_imagen(ref, otra, [otra, ref + 0.01])
        f["clave"] = ["P", "B", i]
        filas.append(f)
    r = resumir(filas)
    assert r["n_dos_manos"] == 10 and r["dos_manos"]["mano_equivocada_pct"] == 100.0
    assert r["por_clase_mano_equivocada_pct"] == {"B": 100.0}
    assert r["mano_equivocada_sobre_todas_pct"] == pytest.approx(100 * 10 / 70)
