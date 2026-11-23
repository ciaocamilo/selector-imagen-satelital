"""Utilidades de prueba: configuración aislada, escenas sintéticas en la malla real y proveedor GEE falso."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mensual  # noqa: E402,F401  (prepara PROJ/GDAL)
from mensual.configuracion import cargar_configuracion  # noqa: E402
from mensual.geometria import caja_en_pixeles, preparar_contexto  # noqa: E402
from mensual.gee import ErrorRed  # noqa: E402
from mensual.productos import derivar_productos  # noqa: E402

CORTE = "2026-11-22T12:00:00Z"
ROCA = (0.08, 0.10, 0.12, 0.25, 0.25)
NIEVE = (0.80, 0.80, 0.80, 0.70, 0.10)
NUBE = (0.70, 0.70, 0.70, 0.70, 0.50)


def cfg_prueba(tmp_path, **cambios):
    base = {"salida__directorio": str(tmp_path / "salida"), "periodo__corte_utc": CORTE,
            "salida__verificar_ultralytics": False}
    base.update(cambios)
    return cargar_configuracion(None, **base)


@pytest.fixture(scope="session")
def ctx():
    return preparar_contexto(cargar_configuracion())


def _pinta(refl, valores, sl):
    for k, v in enumerate(valores):
        refl[k][sl] = v


def ventana_sector(ctx, nombre, alto=20, ancho=20, df=0, dc=0):
    """Rebanada centrada en un sector del núcleo."""
    gp = {n: g for n, _, g in ctx.sectores}[nombre]
    c0, f0, c1, f1 = caja_en_pixeles(ctx.malla, gp)
    fc, cc = int((f0 + f1) / 2) + df, int((c0 + c1) / 2) + dc
    return np.s_[max(fc - alto // 2, 0):fc + alto // 2, max(cc - ancho // 2, 0):cc + ancho // 2]


def escena_sintetica(ctx, tipo, perfil="L2A"):
    """Arrays (refl[5,H,W] float32, qa[H,W]) para los escenarios de las pruebas."""
    H, W = ctx.malla.alto, ctx.malla.ancho
    refl = np.empty((5, H, W), np.float32)
    for k, v in enumerate(ROCA):
        refl[k] = v
    qa = np.full((H, W), 4, np.uint8) if perfil == "L2A" else np.full((H, W), 0.95, np.float32)
    nube_qa = 9 if perfil == "L2A" else 0.1
    if tipo in ("nieve", "nieve_y_nube_nucleo", "ambigua"):
        _pinta(refl, NIEVE, ventana_sector(ctx, "centro"))
    if tipo == "nieve2":
        _pinta(refl, NIEVE, ventana_sector(ctx, "centro"))
        _pinta(refl, NIEVE, ventana_sector(ctx, "sur", 10, 10))
    if tipo in ("nube_nucleo", "nieve_y_nube_nucleo"):
        sl = ventana_sector(ctx, "norte", 400, 400)
        _pinta(refl, NUBE, sl)
        qa[sl] = nube_qa
    if tipo == "ambigua":  # nube en contacto con el parche
        sl = ventana_sector(ctx, "centro", 6, 6, df=-12)
        _pinta(refl, NUBE, sl)
        qa[sl] = nube_qa
    return refl, qa


def sumas_desde(ctx, refl, qa, perfil="L2A"):
    m = ctx.mascaras()
    d = derivar_productos(refl, qa, m["aoi"], perfil)
    cal = d["calidad"]
    return {n: {"total": float(mk.sum() * 400), "datos": float(((cal >= 1) & mk).sum() * 400),
                "util": float(((cal == 2) & mk).sum() * 400)} for n, mk in m.items()}


def t_ms(anio, mes, dia, hora=15):
    return int(datetime(anio, mes, dia, hora, 26, tzinfo=timezone.utc).timestamp() * 1000)


class ProveedorFalso:
    """Implementa inventario() y obtener_bloque() con escenas sintéticas; registra llamadas."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.inventarios = {}        # (perfil, "YYYY-MM") -> [cand] o excepción
        self.escenas = {}            # id -> (refl, qa) real (puede diferir del inventario)
        self.fallos_bloque = {}      # id -> excepción a lanzar
        self.llamadas_inv = []
        self.llamadas_bloque = 0

    def agregar(self, perfil, mes, dia, tipo, tipo_real=None, hora=15, qa_disponible=True, sufijo=""):
        anio, m = map(int, mes.split("-"))
        indice = f"{anio:04d}{m:02d}{dia:02d}T{hora:02d}2649_{anio:04d}{m:02d}{dia:02d}T{hora:02d}2651{sufijo}_T18NVL"
        col = "COPERNICUS/S2_SR_HARMONIZED" if perfil == "L2A" else "COPERNICUS/S2_HARMONIZED"
        refl, qa = escena_sintetica(self.ctx, tipo, perfil)
        cand = {"id": f"{col}/{indice}", "indice": indice, "t": t_ms(anio, m, dia, hora), "tesela": "18NVL",
                "satelite": "Sentinel-2A", "orbita": 25, "baseline": "05.10", "nube_granulo_pct": 5.0,
                "qa_disponible": qa_disponible, "sumas": sumas_desde(self.ctx, refl, qa, perfil)}
        self.escenas[cand["id"]] = escena_sintetica(self.ctx, tipo_real, perfil) if tipo_real else (refl, qa)
        self.inventarios.setdefault((perfil, mes), []).append(cand)
        return cand

    def inventario(self, perfil, inicio, fin_consulta):
        mes = f"{inicio:%Y-%m}"
        self.llamadas_inv.append((perfil, mes))
        v = self.inventarios.get((perfil, mes), [])
        if isinstance(v, Exception):
            raise v
        return [dict(c) for c in v]

    def obtener_bloque(self, perfil, cand, producto, malla, ventana):
        self.llamadas_bloque += 1
        if cand["id"] in self.fallos_bloque:
            raise self.fallos_bloque[cand["id"]]
        f0, c0, h, w = ventana
        refl, qa = self.escenas[cand["id"]]
        if producto == "reflectancia":
            return refl[:, f0:f0 + h, c0:c0 + w].copy()
        return qa[None, f0:f0 + h, c0:c0 + w].copy()


@pytest.fixture
def falso(ctx):
    return ProveedorFalso(ctx)


__all__ = ["cfg_prueba", "ProveedorFalso", "ErrorRed", "escena_sintetica", "ventana_sector", "CORTE"]
