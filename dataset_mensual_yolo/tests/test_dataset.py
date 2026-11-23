"""Empaquetado de los dos datasets, cargador de Ultralytics, fallos, reanudación y reutilización."""
import os

import pytest
import yaml

import mensual.dataset as dataset_mod
from mensual.dataset import regenerar_yaml
from mensual.flujo import ejecutar
from mensual.utilidades import Bloqueo, ErrorBloqueo, leer_csv, leer_json, sha256_archivo

from conftest import cfg_prueba

SILENCIO = {"avance": lambda *_: None}
MESES = ["2020-01", "2020-02", "2024-01", "2025-06"]  # train, train, val, test con 132 meses (92/19/21)


def preparar(falso):
    falso.agregar("L2A", "2020-01", 16, "nieve")
    falso.agregar("L2A", "2020-02", 10, "negativa")
    falso.agregar("L2A", "2024-01", 23, "nieve2")
    falso.agregar("L2A", "2025-06", 2, "nieve")


def test_dos_datasets_con_splits_identicos_y_cargador_ultralytics(tmp_path, falso):
    preparar(falso)
    cfg = cfg_prueba(tmp_path, periodo__meses=MESES, salida__verificar_ultralytics=True)
    res = ejecutar(cfg, proveedor=falso, **SILENCIO)
    v = tmp_path / "salida" / "datasets" / res["version_dataset"]
    resumen = leer_json(v / "resumen.json")
    assert resumen["cortes"]["meses_por_split"] == {"train": 92, "val": 19, "test": 21}
    assert resumen["cortes"]["corte_val"] == "2023-09" and resumen["cortes"]["corte_test"] == "2025-04"
    assert resumen["estado_producto"] == {"rgb": "entrenable", "espectral": "entrenable"}
    assert resumen["conteos"]["train"]["negativas"] == 1 and resumen["conteos"]["val"]["instancias"] == 2
    for variante in ("rgb", "espectral"):
        ul = resumen["verificacion"]["ultralytics"][variante]
        assert ul["train"]["imagenes"] == 2 and ul["test"]["instancias"] == 1
        assert not list((v / variante / "labels").glob("*.cache"))
        datos = yaml.safe_load((v / variante / "data.yaml").read_text(encoding="utf-8"))
        assert datos["path"] == (v / variante).resolve().as_posix() and datos["names"] == {0: "nieve_hielo_visible"}
    filas = leer_csv(v / "manifiesto.csv")
    assert len({f["nombre"] for f in filas}) == 4
    assert {f["mes"]: f["split"] for f in filas} == dict(zip(MESES, ["train", "train", "val", "test"]))
    for s in ("train", "val", "test"):
        rgb = sorted(p.name for p in (v / "rgb" / "labels" / s).iterdir())
        esp = sorted(p.name for p in (v / "espectral" / "labels" / s).iterdir())
        assert rgb == esp
        for n in rgb:
            assert sha256_archivo(v / "rgb" / "labels" / s / n) == sha256_archivo(v / "espectral" / "labels" / s / n)
        imgs = sorted(p.stem for p in (v / "rgb" / "images" / s).iterdir())
        assert imgs == [n[:-4] for n in rgb]
    neg = next(f for f in filas if f["decision"] == "negativa")
    assert (v / "rgb" / "labels" / "train" / f"{neg['nombre']}.txt").read_text() == ""


def test_reejecucion_sin_cambios_no_crea_version_ni_red(tmp_path, falso):
    preparar(falso)
    cfg = cfg_prueba(tmp_path, periodo__meses=MESES)
    r1 = ejecutar(cfg, proveedor=falso, **SILENCIO)
    inv, bloques = len(falso.llamadas_inv), falso.llamadas_bloque
    r2 = ejecutar(cfg, proveedor=falso, **SILENCIO)
    assert r2["version_dataset"] == r1["version_dataset"] and not r2["dataset_nuevo"]
    assert (len(falso.llamadas_inv), falso.llamadas_bloque) == (inv, bloques)
    r3 = ejecutar(cfg, sin_red=True, **SILENCIO)  # también sin red, todo desde la caché
    assert r3["version_dataset"] == r1["version_dataset"]


def test_cambio_de_umbral_reutiliza_descargas(tmp_path, falso):
    preparar(falso)
    r1 = ejecutar(cfg_prueba(tmp_path, periodo__meses=MESES), proveedor=falso, **SILENCIO)
    inv, bloques = len(falso.llamadas_inv), falso.llamadas_bloque
    cfg2 = cfg_prueba(tmp_path, periodo__meses=MESES)
    cfg2["etiquetas"]["L2A"]["min_pixeles"] = 6
    r2 = ejecutar(cfg2, proveedor=falso, **SILENCIO)
    assert (len(falso.llamadas_inv), falso.llamadas_bloque) == (inv, bloques)
    assert r2["huellas"]["reglas"]["L2A"] != r1["huellas"]["reglas"]["L2A"]
    assert r2["version_dataset"] == "v0002" and r2["meses_dataset"] == 4


def test_descarga_interrumpida_no_publica_y_se_reanuda(tmp_path, falso):
    c = falso.agregar("L2A", "2020-01", 16, "nieve")
    falso.fallos_bloque[c["id"]] = OSError("conexión cortada a mitad de bloque")
    r1 = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"]), proveedor=falso, **SILENCIO)
    assert r1["estados_ejecucion"] == {"error_reintentable": 1}
    cache = tmp_path / "salida" / "cache_escenas"
    assert not list(cache.rglob("metadata.json"))
    del falso.fallos_bloque[c["id"]]
    r2 = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"]), proveedor=falso, **SILENCIO)
    assert r2["estados_ejecucion"] == {"exportada": 1}
    assert not [p for p in cache.rglob("_tmp_*")]


def test_empaquetado_interrumpido_no_publica(tmp_path, falso, monkeypatch):
    preparar(falso)
    cfg = cfg_prueba(tmp_path, periodo__meses=MESES)

    def falla(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset_mod, "verificar_paquete", falla)
    with pytest.raises(KeyboardInterrupt):
        ejecutar(cfg, proveedor=falso, **SILENCIO)
    ds = tmp_path / "salida" / "datasets"
    assert not list(ds.glob("v*")) and not list(ds.glob("_tmp_*")) and not (ds / "ultima.json").exists()
    assert not (tmp_path / "salida" / ".bloqueo").exists()
    monkeypatch.undo()
    res = ejecutar(cfg, proveedor=falso, **SILENCIO)
    assert res["version_dataset"] == "v0001" and res["estados_ejecucion"] == {"exportada": 4}


def test_regenerar_yaml_tras_mover(tmp_path, falso):
    falso.agregar("L2A", "2020-01", 16, "nieve")
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2020-01"], salida__enlaces="copia"), proveedor=falso,
                   **SILENCIO)
    nuevo = tmp_path / "movido"
    os.rename(res["carpeta_dataset"], nuevo)
    regenerar_yaml(nuevo)
    datos = yaml.safe_load((nuevo / "espectral" / "data.yaml").read_text(encoding="utf-8"))
    assert datos["path"] == (nuevo / "espectral").resolve().as_posix()


def test_bloqueo_impide_dos_escritores(tmp_path):
    (tmp_path / ".bloqueo").write_text('{"pid": %d}' % os.getppid())
    with pytest.raises(ErrorBloqueo):
        with Bloqueo(tmp_path):
            pass
    (tmp_path / ".bloqueo").write_text('{"pid": 999999}')  # proceso muerto: se recupera
    with Bloqueo(tmp_path):
        pass
