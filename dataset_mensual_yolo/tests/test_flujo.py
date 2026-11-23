"""Enrutamiento de fuentes y selección dentro del flujo, con el proveedor falso (sin red)."""
from mensual.flujo import ejecutar
from mensual.gee import ErrorAutenticacion
from mensual.utilidades import leer_csv, leer_json

from conftest import ErrorRed, cfg_prueba

SILENCIO = {"avance": lambda *_: None}


def meses(res):
    return {f["mes"]: f for f in leer_csv(res["meses_csv"])}


def test_2016_usa_l1c_con_cloud_score_por_defecto(tmp_path, falso):
    falso.agregar("L1C", "2016-06", 14, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2016-06"]), proveedor=falso, **SILENCIO)
    f = meses(res)["2016-06"]
    assert falso.llamadas_inv == [("L1C", "2016-06")]
    assert f["perfil"] == "L1C" and f["estado"] == "exportada" and f["clave_escena"].endswith("_l1c")
    meta = leer_json(next((tmp_path / "salida" / "cache_escenas").rglob("metadata.json")))
    assert meta["coleccion"] == "COPERNICUS/S2_HARMONIZED" and "Cloud Score+" in meta["metodo_qa"]


def test_solo_l2a_deja_2016_fuera_de_coleccion(tmp_path, falso):
    falso.agregar("L1C", "2016-06", 14, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2016-06"], fuentes__modo="solo_l2a"), proveedor=falso,
                   **SILENCIO)
    assert meses(res)["2016-06"]["estado"] == "fuera_coleccion" and falso.llamadas_inv == []


def test_l2a_ausente_pasa_a_l1c_y_qa_ausente_es_sin_qa(tmp_path, falso):
    falso.agregar("L1C", "2018-05", 10, "nieve", qa_disponible=False)
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2018-05"]), proveedor=falso, **SILENCIO)
    f = meses(res)["2018-05"]
    assert falso.llamadas_inv == [("L2A", "2018-05"), ("L1C", "2018-05")]
    assert f["estado"] == "sin_qa" and "L2A vacío" in f["ruta_fuentes"] and falso.llamadas_bloque == 0


def test_error_de_red_es_reintentable_y_no_activa_l1c(tmp_path, falso):
    falso.inventarios[("L2A", "2018-05")] = ErrorRed("timeout")
    falso.agregar("L1C", "2018-05", 10, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2018-05"]), proveedor=falso, **SILENCIO)
    assert meses(res)["2018-05"]["estado"] == "error_reintentable"
    assert falso.llamadas_inv == [("L2A", "2018-05")]


def test_l2a_nublado_no_activa_respaldo(tmp_path, falso):
    falso.agregar("L2A", "2019-03", 5, "nube_nucleo")
    falso.agregar("L1C", "2019-03", 5, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2019-03"]), proveedor=falso, **SILENCIO)
    assert meses(res)["2019-03"]["estado"] == "sin_escena_viable"
    assert ("L1C", "2019-03") not in falso.llamadas_inv
    c = leer_csv(res["candidatos_csv"])[0]
    assert "nubes_nucleo" in c["motivos"] and float(c["peor_sector_pct"]) < 90


def test_fallo_local_pasa_al_siguiente_candidato(tmp_path, falso):
    # El inventario (20 m) promete visibilidad pero la escena real tiene nube en las cumbres.
    falso.agregar("L2A", "2020-01", 16, "nieve", tipo_real="nieve_y_nube_nucleo")
    falso.agregar("L2A", "2020-01", 21, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"]), proveedor=falso, **SILENCIO)
    f = meses(res)["2020-01"]
    cands = {c["orden"]: c for c in leer_csv(res["candidatos_csv"])}
    assert cands["1"]["resultado"] == "calidad_local" and "nubes_nucleo" in cands["1"]["detalle"]
    assert f["estado"] == "exportada" and f["clave_escena"].startswith("20200121") and f["seleccion_incompleta"] == "False"


def test_error_de_descarga_marca_seleccion_incompleta(tmp_path, falso):
    c1 = falso.agregar("L2A", "2020-01", 16, "nieve")
    falso.agregar("L2A", "2020-01", 21, "nieve")
    falso.fallos_bloque[c1["id"]] = ErrorRed("HTTP 503")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"]), proveedor=falso, **SILENCIO)
    f = meses(res)["2020-01"]
    assert f["estado"] == "exportada" and f["seleccion_incompleta"] == "True"
    # Siguiente ejecución: el mes se reintenta y el candidato superior sustituye al anterior.
    del falso.fallos_bloque[c1["id"]]
    res2 = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"]), proveedor=falso, **SILENCIO)
    f2 = meses(res2)["2020-01"]
    assert f2["clave_escena"].startswith("20200116") and f2["seleccion_incompleta"] == "False"
    assert res2["version_dataset"] == "v0002" and (tmp_path / "salida" / "datasets" / "v0001").is_dir()


def test_escena_ambigua_no_se_exporta(tmp_path, falso):
    falso.agregar("L2A", "2021-07", 3, "ambigua")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2021-07"]), proveedor=falso, **SILENCIO)
    assert meses(res)["2021-07"]["estado"] == "etiqueta_ambigua"
    assert not list((tmp_path / "salida" / "datasets" / "v0001" / "rgb" / "labels").rglob("*.txt"))


def test_reprocesado_de_la_misma_adquisicion_no_se_descarga_dos_veces(tmp_path, falso):
    falso.agregar("L1C", "2016-01", 20, "ambigua")
    falso.agregar("L1C", "2016-01", 20, "ambigua", sufijo="R")  # mismo datatake, otro procesamiento
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2016-01"]), proveedor=falso, **SILENCIO)
    resultados = sorted(c["resultado"] for c in leer_csv(res["candidatos_csv"]))
    assert resultados == ["duplicado_adquisicion", "etiqueta_ambigua"]
    assert meses(res)["2016-01"]["estado"] == "etiqueta_ambigua"
    assert falso.llamadas_bloque == 8  # 4 bloques de reflectancia + 4 de QA de una sola descarga


def test_sin_autenticacion_detiene_red_y_empaqueta(tmp_path, falso):
    falso.inventarios[("L2A", "2022-01")] = ErrorAutenticacion("invalid_grant")
    falso.agregar("L2A", "2022-02", 3, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2022-01", "2022-02"]), proveedor=falso, **SILENCIO)
    m = meses(res)
    assert m["2022-01"]["estado"] == m["2022-02"]["estado"] == "error_reintentable"
    assert "sin_autenticacion" in m["2022-01"]["motivo"] and falso.llamadas_inv == [("L2A", "2022-01")]
    assert (tmp_path / "salida" / "datasets" / "v0001" / "rgb" / "data.yaml").is_file()
