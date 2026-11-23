"""Regresión sin red con la escena piloto real del selector."""
import shutil

import pytest

from mensual.configuracion import cargar_configuracion
from mensual.geometria import Malla, preparar_contexto
from mensual.piloto import RAIZ_SELECTOR, configuracion_piloto, localizar_piloto, regresion_piloto, revision_publicada
from mensual.utilidades import leer_json

CFG = cargar_configuracion()
pytestmark = pytest.mark.skipif(not (RAIZ_SELECTOR / "datos_sentinel2" / "santa_isabel_6db5e66f6c").is_dir(),
                                reason="escena piloto del selector no disponible")
PILOTO = localizar_piloto(preparar_contexto(CFG))


def test_aoi_del_dataset_alineado_dentro_de_la_malla_del_piloto():
    ctx = preparar_contexto(CFG)
    assert PILOTO is not None and ctx.id_aoi != "santa_isabel_6db5e66f6c"
    m = Malla.desde_dict(leer_json(PILOTO / "metadata.json")["malla"])
    dc, df = (ctx.malla.transform[2] - m.transform[2]) / 10, (m.transform[5] - ctx.malla.transform[5]) / 10
    assert dc == int(dc) and df == int(df) and dc >= 0 and df >= 0  # submalla exacta de la del selector
    assert dc + ctx.malla.ancho <= m.ancho and df + ctx.malla.alto <= m.alto


def test_malla_y_id_aoi_del_selector():
    ctx = preparar_contexto(configuracion_piloto(CFG, PILOTO))
    assert ctx.id_aoi == "santa_isabel_6db5e66f6c"
    assert (ctx.malla.ancho, ctx.malla.alto) == (999, 996)
    assert ctx.malla.transform == (10.0, 0.0, 454530.0, 0.0, -10.0, 536100.0)


def test_productos_se_reproducen_bit_a_bit_y_conteos():
    inf = regresion_piloto(PILOTO, CFG)
    assert inf["ok"], inf["problemas"]
    assert all(inf["productos_reproducidos"].values())
    assert inf["conteos_calidad"] == {0: 2214, 1: 11472, 2: 981318} and inf["soporte_px"] == 991475
    assert inf["decision_automatica"]["decision"] == "ambigua"  # sin SCL no se puede corroborar agua/oscuros
    aceptados = [p for p in inf["parches"] if p["decision_automatica"] == "aceptado"]
    assert len(aceptados) == 1 and len(inf["parches"]) == 9


def test_revision_publicada_ignora_temporales(tmp_path):
    rev = revision_publicada(PILOTO)
    assert rev["revision"] == "0001" and len(rev["instancias_txt"]) == 7
    # Copia con un borrador temporal (como _tmp_0002_* del selector): se ignora aunque parezca más reciente.
    copia = tmp_path / "piloto"
    shutil.copytree(PILOTO / "anotaciones", copia / "anotaciones")
    tmp = copia / "anotaciones" / "revisiones" / "_tmp_0002_prueba"
    shutil.copytree(copia / "anotaciones" / "revisiones" / "0001", tmp)
    (tmp / "borrador.json").write_text("{}", encoding="utf-8")
    rev2 = revision_publicada(copia)
    assert rev2["revision"] == "0001" and "_tmp_0002_prueba" in rev2["ignoradas"]
