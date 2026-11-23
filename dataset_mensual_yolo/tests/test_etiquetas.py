"""Pseudoetiquetado sobre escenas pequeñas en memoria: nieve, laguna, sombra, borde sin datos, parche diminuto,
negativa observable y negativa incierta."""
import numpy as np
import pytest

from mensual.configuracion import cargar_configuracion
from mensual.etiquetas import decidir_escena, leer_txt_yolo, reglas_perfil, serializar_yolo, verificar_ida_vuelta

H = W = 60


def escena(perfil="L2A"):
    ndsi = np.full((H, W), -0.4, np.float32)
    b8 = np.full((H, W), 0.25, np.float32)
    calidad = np.full((H, W), 2, np.uint8)
    soporte = np.ones((H, W), bool)
    qa = np.full((H, W), 4, np.uint8) if perfil == "L2A" else np.full((H, W), 0.9, np.float32)
    return {"ndsi": ndsi, "b8": b8, "calidad": calidad, "soporte": soporte, "qa": qa, "perfil": perfil,
            "ancho": W, "alto": H}


def mascaras():
    m = {"aoi": np.ones((H, W), bool)}
    for k, n in enumerate(("norte", "centro", "sur")):
        x = np.zeros((H, W), bool)
        x[k * 20:(k + 1) * 20, 10:50] = True
        m[n] = x
    return m


@pytest.fixture(scope="module")
def reglas():
    cfg = cargar_configuracion()
    return {p: reglas_perfil(cfg, p) for p in ("L2A", "L1C")}


def nieve(e, sl, ndsi=0.8, b8=0.7):
    e["ndsi"][sl] = ndsi
    e["b8"][sl] = b8


def test_nieve_visible_es_positiva_con_ida_y_vuelta(reglas, tmp_path):
    e = escena()
    nieve(e, np.s_[25:35, 20:32])
    nieve(e, np.s_[5:9, 40:44])
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "positiva" and len(r["poligonos"]) == 2
    ruta = tmp_path / "x.txt"
    ruta.write_text(serializar_yolo(r["poligonos"], W, H), encoding="utf-8")
    assert verificar_ida_vuelta(ruta, r["poligonos"], W, H) <= 0.01
    assert all(c == 0 for c, _ in leer_txt_yolo(ruta))
    areas = sorted(c["area_px"] for c in r["componentes"])
    assert areas == [16.0, 120.0]


def test_laguna_l2a_se_excluye_y_la_escena_es_negativa(reglas):
    e = escena()
    sl = np.s_[25:33, 20:30]
    nieve(e, sl, ndsi=0.7, b8=0.02)
    e["qa"][sl] = 6
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["agua"] == 1 and r["decision"] == "negativa" and not r["poligonos"]


def test_laguna_l1c_sin_mascara_es_ambigua_y_con_mascara_negativa(reglas):
    e = escena("L1C")
    sl = np.s_[25:33, 20:30]
    nieve(e, sl, ndsi=0.7, b8=0.02)
    assert decidir_escena(e, mascaras(), reglas["L1C"])["decision"] == "ambigua"
    agua = np.zeros((H, W), bool)
    agua[sl] = True
    r = decidir_escena(e, mascaras(), reglas["L1C"], agua_aux=agua)
    assert r["agua"] == 1 and r["decision"] == "negativa"


def test_sombra_no_visible_bloquea(reglas):
    e = escena()
    sl = np.s_[25:33, 20:30]
    nieve(e, sl, ndsi=0.75, b8=0.05)
    e["calidad"][sl] = 1  # sombra excluida por la máscara de calidad
    e["qa"][sl] = 3
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "ambigua"
    assert {"util_insuficiente", "oscuro_sin_corroboracion"} <= set(r["componentes"][0]["motivos"])


def test_borde_sin_datos_hace_ambiguo(reglas):
    e = escena()
    nieve(e, np.s_[25:35, 20:30])
    e["soporte"][:, 30:33] = False
    e["calidad"][:, 30:33] = 0
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "ambigua" and "contacto_ocultacion" in r["componentes"][0]["motivos"]
    e2 = escena()
    nieve(e2, np.s_[0:8, 20:30])
    assert "truncado_borde" in decidir_escena(e2, mascaras(), reglas["L2A"])["componentes"][0]["motivos"]


def test_parche_diminuto_no_se_exporta_como_negativa(reglas):
    e = escena()
    nieve(e, np.s_[30:31, 30:33])
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "ambigua" and r["componentes"][0]["motivos"] == ["bajo_minimo"]


def test_huecos_pequenos_se_rellenan_y_grandes_bloquean(reglas):
    e = escena()
    nieve(e, np.s_[20:32, 20:32])
    e["ndsi"][25, 25] = -0.2
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "positiva" and r["componentes"][0]["huecos_rellenados"] == 1
    e["ndsi"][26:29, 26:29] = -0.2
    assert decidir_escena(e, mascaras(), reglas["L2A"])["decision"] == "ambigua"


def test_negativa_totalmente_observable(reglas):
    r = decidir_escena(escena(), mascaras(), reglas["L2A"])
    assert r["decision"] == "negativa" and r["barrido_negativa"]["plausibles"] == 0


def test_negativa_incierta_por_barrido_o_visibilidad(reglas):
    e = escena()
    nieve(e, np.s_[25:35, 20:30], ndsi=0.35)
    assert decidir_escena(e, mascaras(), reglas["L2A"])["decision"] == "negativa_incierta"
    e = escena()
    e["calidad"][0:10, 10:50] = 1
    r = decidir_escena(e, mascaras(), reglas["L2A"])
    assert r["decision"] == "negativa_incierta" and "norte" in r["motivo"]
