"""Acceso al dataset LSC70 (zip o carpeta) y exploración de su estructura (tarea 1).

Principio: NO se asume la estructura. `explorar` imprime el árbol real y los conteos y
escribe `configs/estructura_<nombre>.json` con `confirmado: false`. Las etapas
siguientes (extraer, medir) se niegan a correr hasta que una persona revise lo
impreso y ponga `confirmado: true`. Así la estructura la valida un humano, no una
suposición heredada de ANH.

    python -m lsc70.explorar_estructura --fuente LSC70AN.zip --nombre an
"""

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

EXT_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")
DIR_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class Fuente:
    """Un zip o una carpeta con imágenes. Se lee sin extraer a disco."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self._zip = zipfile.ZipFile(self.ruta) if self.ruta.is_file() else None
        todas = (
            self._zip.namelist()
            if self._zip
            else [
                p.relative_to(self.ruta).as_posix()
                for p in self.ruta.rglob("*")
                if p.is_file()
            ]
        )
        self.todas = [n for n in todas if not n.endswith("/")]
        self.raiz = self._detectar_raiz()

    def _detectar_raiz(self):
        """Carpeta única que envuelve todo (p. ej. 'LSC70ANH/'). Se quita para que los
        índices de participante y clase no dependan de ese envoltorio."""
        primeros = {n.split("/")[0] for n in self.todas if "/" in n}
        sueltos = [n for n in self.todas if "/" not in n]
        return next(iter(primeros)) if len(primeros) == 1 and not sueltos else ""

    def relativa(self, nombre):
        return nombre[len(self.raiz) + 1 :] if self.raiz else nombre

    def imagenes(self):
        return sorted(n for n in self.todas if n.lower().endswith(EXT_IMAGEN))

    def leer(self, nombre):
        return (
            self._zip.read(nombre) if self._zip else (self.ruta / nombre).read_bytes()
        )

    def leer_imagen(self, nombre):
        """BGR en memoria, o None si el archivo está corrupto."""
        buf = np.frombuffer(self.leer(nombre), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def parsear(rel, idx_participante=0, idx_clase=1):
    """'Per01/A/Per01_A_3.jpg' -> ('Per01', 'A', 3). None si no encaja.

    Misma convención que la celda 8 del notebook (clase en mayúsculas). El índice de
    toma sale del último '_' del nombre; si no es numérico, queda en None.
    """
    partes = rel.split("/")
    if len(partes) < 3:
        return None
    m = re.search(r"_(\d+)\.[A-Za-z]+$", partes[-1])
    return (
        partes[idx_participante],
        partes[idx_clase].upper(),
        (int(m.group(1)) if m else None),
    )


def id_toma(participante, clase):
    """Una toma = (participante, clase). Supuesto de ANH, que explorar() verifica."""
    return f"{participante}__{clase}"


def arbol(rutas, max_prof=3, max_hijos=6):
    """Árbol de carpetas truncado, construido solo a partir de los nombres."""

    def nodo():
        return defaultdict(nodo)

    raiz = nodo()
    for r in rutas:
        actual = raiz
        for p in r.split("/")[:-1]:
            actual = actual[p]
    lineas = []

    def rec(n, prof, pref):
        hijos = sorted(n)
        for h in hijos[:max_hijos]:
            lineas.append(f"{pref}{h}/")
            if prof < max_prof:
                rec(n[h], prof + 1, pref + "    ")
        if len(hijos) > max_hijos:
            lineas.append(f"{pref}... (+{len(hijos) - max_hijos} más)")

    rec(raiz, 1, "")
    return "\n".join(lineas)


def explorar(ruta, nombre, idx_participante=0, idx_clase=1, n_tamanos=30):
    """Imprime la estructura real y escribe la configuración pendiente de confirmar."""
    f = Fuente(ruta)
    imgs = f.imagenes()
    otras = Counter(Path(n).suffix.lower() for n in f.todas if n not in set(imgs))
    rels = [f.relativa(n) for n in imgs]

    print(f"Fuente: {f.ruta.name} | entradas: {len(f.todas)} | imágenes: {len(imgs)}")
    print(f"Carpeta raíz detectada: {f.raiz or '(ninguna)'}")
    if otras:
        print(f"Archivos que no son imagen: {dict(otras)}")
    print("\n=== ÁRBOL (profundidad 3, máx. 6 hijos por nivel) ===")
    print(arbol(rels))

    por_clase, por_part, frames_toma = Counter(), Counter(), Counter()
    clases_de = defaultdict(set)
    no_parseadas = []
    for r in rels:
        p = parsear(r, idx_participante, idx_clase)
        if p is None:
            no_parseadas.append(r)
            continue
        part, cls, _ = p
        por_clase[cls] += 1
        por_part[part] += 1
        frames_toma[(part, cls)] += 1
        clases_de[part].add(cls)

    print(f"\nNo parseadas: {len(no_parseadas)} {no_parseadas[:3]}")
    print(
        f"Participantes: {len(por_part)} | Clases: {len(por_clase)} | Tomas: {len(frames_toma)}"
    )
    print("\n=== IMÁGENES POR CLASE ===")
    for c, n in sorted(por_clase.items()):
        print(f"  {c:>8} {n:>6}")
    print("\n=== FRAMES POR TOMA (¿se cumple el supuesto de 6?) ===")
    print("  ", dict(sorted(Counter(frames_toma.values()).items())))
    todas_clases = set(por_clase)
    incompletos = {
        p: sorted(todas_clases - c) for p, c in clases_de.items() if todas_clases - c
    }
    print(f"\nParticipantes con clases ausentes: {len(incompletos)}")
    for p, falta in sorted(incompletos.items())[:10]:
        print(f"  {p}: faltan {falta}")

    # Tamaño real de las imágenes: es lo primero que hay que saber de la sección.
    paso = max(1, len(imgs) // n_tamanos)
    tamanos = Counter()
    for n in imgs[::paso][:n_tamanos]:
        im = f.leer_imagen(n)
        if im is not None:
            tamanos[f"{im.shape[1]}x{im.shape[0]}"] += 1
    print(
        f"\nTamaños (ancho x alto) en {sum(tamanos.values())} imágenes muestreadas: {dict(tamanos)}"
    )

    DIR_CONFIGS.mkdir(exist_ok=True)
    salida = DIR_CONFIGS / f"estructura_{nombre}.json"
    salida.write_text(
        json.dumps(
            {
                "nombre": nombre,
                "fuente": f.ruta.name,
                "raiz_detectada": f.raiz,
                "idx_participante": idx_participante,
                "idx_clase": idx_clase,
                "preprocesado": "anh" if nombre.lower().startswith("anh") else "nativo",
                "n_imagenes": len(imgs),
                "n_participantes": len(por_part),
                "clases": dict(sorted(por_clase.items())),
                "frames_por_toma": {
                    str(k): v for k, v in sorted(Counter(frames_toma.values()).items())
                },
                "tamanos_muestra": dict(tamanos),
                "no_parseadas": len(no_parseadas),
                "confirmado": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nEscrito: {salida}  (confirmado: false)")
    print(
        'Revisa el árbol y los conteos. Si son correctos, pon "confirmado": true en ese archivo;'
    )
    print("sin eso, extraer y medir se niegan a correr.")


def cargar_estructura(nombre):
    """Lee la configuración y exige que una persona la haya confirmado."""
    ruta = DIR_CONFIGS / f"estructura_{nombre}.json"
    if not ruta.exists():
        raise SystemExit(
            f"Falta {ruta}. Corre primero: python -m lsc70.explorar_estructura"
        )
    cfg = json.loads(ruta.read_text(encoding="utf-8"))
    if not cfg.get("confirmado"):
        raise SystemExit(
            f"{ruta.name} tiene confirmado=false. Revisa el árbol que imprimió "
            "explorar_estructura y cámbialo a true si la estructura es la correcta."
        )
    return cfg
