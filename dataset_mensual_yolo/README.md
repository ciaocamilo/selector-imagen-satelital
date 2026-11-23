# dataset_mensual_yolo

Dataset mensual automático de **nieve y hielo visible** del Nevado Santa Isabel (Sentinel-2, 2016–2026) para
**YOLO Segmentation**. Implementa `SDD_03_Dataset_Mensual_Automatico_YOLO.md`. Reutiliza como módulos la malla,
la descarga por bloques, la derivación de productos y la geometría YOLO del selector
(`buscar_descargar_sentinel2.ipynb`), que sigue funcionando por separado.

Para cada mes del calendario elige **una adquisición real como máximo**:

1. Inventaría el mes en Earth Engine y calcula las métricas en el servidor (20 m).
2. Ordena los candidatos por visibilidad del peor sector de las cumbres, después del AOI y por cercanía al centro del mes.
3. Descarga los candidatos en ese orden.
4. Verifica cada uno localmente y lo pseudoetiqueta con NDSI, sin clics.
5. Se queda con la primera escena exportable.

Al final publica dos datasets con las mismas claves, etiquetas y particiones:

- `rgb`: B4/B3/B2.
- `espectral`: NDSI/NDVI/B4.

## Requisitos

- Entorno conda `gee` (`pip install -r requirements.txt`).
- Credenciales de Earth Engine creadas una vez **fuera** del flujo:

  ```
  earthengine authenticate
  earthengine set_project <proyecto>
  ```

  También se puede usar `EE_PROJECT` o `[gee].proyecto`. El bucle nunca pide autenticación: sin credenciales, los meses quedan `error_reintentable` y se empaqueta lo que haya en caché.

## Ejecución

Desde esta carpeta:

```
conda activate gee
python -m mensual --config configuracion.toml                                   # rango completo 2016–2026
python -m mensual --config configuracion.toml --meses 2016-01,2024-01 --salida salida_prueba   # prueba corta
python -m mensual --config configuracion.toml --meses 2019-01..2019-12 --corte 2026-11-22
python -m mensual --config configuracion.toml --sin-red                         # solo caché
python -m mensual --config configuracion.toml --regresion-piloto                # regresión con la escena piloto
```

Equivalente en el cuaderno `generar_dataset_mensual.ipynb`:

1. Edita la **celda 1** (`CAMBIOS`, `SIN_RED`, `EJECUTAR`).
2. Ejecuta todas las celdas.
3. La celda 4 llama a la misma función `mensual.ejecutar` que la CLI.

El progreso se muestra por mes y al final aparecen el resumen y las rutas de los dos `data.yaml`.

## Reanudación

- Cada mes se guarda al terminar en `salida/estado/<aoi>/meses/<mes>.json`. Repetir el mismo comando continúa donde quedó y no repite consultas ni descargas: los inventarios de meses cerrados y las escenas quedan en caché.
- Los meses `error_reintentable` se reintentan (`--no-reintentar` lo evita). El mes en curso es provisional y se vuelve a consultar (`--no-refrescar` lo evita).
- Cambiar un umbral de `[etiquetas]` reutiliza las descargas y solo recalcula etiquetas. Cambiar `[particiones]` solo reempaqueta.
- Cada cambio de contenido publica una versión nueva (`datasets/vNNNN`) de forma atómica. Las anteriores se conservan; `datasets/ultima.json` apunta a la vigente.
- Un bloqueo impide dos ejecuciones simultáneas sobre la misma salida.

## Estados de `meses.csv`

`meses.csv` tiene una fila por mes planificado (132 en el rango completo).

| Estado | Significado |
|---|---|
| `exportada` | Escena elegida. La etiqueta es `positiva` (polígonos) o `negativa` (TXT vacío, cumbres observables al 99 %). |
| `etiqueta_ambigua` | Hubo escenas visibles, pero todas tienen componentes NDSI no resolubles automáticamente. |
| `sin_escena_viable` | Ninguna adquisición cumple la visibilidad en el AOI y en los tres sectores del núcleo (nubes). |
| `sin_qa` | Hay adquisiciones, pero sin máscara de calidad (SCL o Cloud Score+) y ninguna otra pasa. |
| `sin_adquisicion` | El inventario concluyó sin gránulos sobre el AOI. |
| `fuera_coleccion` | Mes anterior a L2A en modo `solo_l2a`. |
| `error_reintentable` | Red, cuota o autenticación. No demuestra ausencia de datos; se reintenta en la siguiente ejecución. |
| `futuro` | Mes posterior al corte; no se consulta Earth Engine. |
| `no_procesado` | Mes fuera del subconjunto `--meses` y sin estado previo en esta salida. |

Otras columnas:

- `conteo_motivos`: causas mixtas.
- `seleccion_incompleta=True`: un candidato mejor falló por error técnico; se reevalúa después.
- `candidatos.csv`: métricas, orden y resultado de cada candidato.
- `cache_escenas/.../etiquetas/<huella>/decision.json` y `evidencia.png`: decisión por componente (píxeles, NDSI, contacto con zonas ocultas, agua, motivo).

## Salida

```
salida/cache_escenas/<aoi>/<L2A|L1C>/<clave>/<receta>/   GeoTIFF, PNG, metadata.json, etiquetas/
salida/ejecuciones/<id>/{meses.csv, candidatos.csv, configuracion.toml, resumen_ejecucion.json}
salida/datasets/vNNNN/{manifiesto.csv, configuracion.toml, resumen.json}
salida/datasets/vNNNN/{rgb,espectral}/data.yaml
salida/datasets/vNNNN/{rgb,espectral}/{images,labels}/{train,val,test}/
```

Particiones: por defecto son cronológicas sobre el calendario (70/15/15 con floor, es decir 92/19/21 meses para 2016–2026). También se admiten cortes explícitos en `[particiones]`. `resumen.json` informa:

- las escenas reales por split, perfil y año;
- el estado del producto: `entrenable`, `entrenable_sin_prueba` o `no_entrenable`.

**Mover la carpeta.** Los YAML usan `path:` absoluto. Después de moverla, ejecuta:

```
python -m mensual --regenerar-yaml salida/datasets/vNNNN
```

## Ajustes

- **AOI** `[aoi]`: `[-75.397, 4.773, -75.348, 4.827]` (id `santa_isabel_54c3fecf1b`, malla 545×598 a 10 m, alineada como submalla de la escena piloto del selector). El núcleo por defecto `[-75.385, 4.775, -75.348, 4.827]` está recortado para quedar dentro de este AOI. La regresión piloto se evalúa con el AOI del selector, que es el que tiene la escena piloto.
- **Núcleo de cumbres** `[nucleo]`: rectángulo WGS84 dividido en tres franjas latitudinales. Revisa el encuadre con la celda 2 del cuaderno y ajústalo con cartografía de cumbres. Debe quedar dentro del AOI. La búsqueda de parches siempre abarca toda la escena.
- **Fuentes**:
  - `l2a_preferente`: L2A, y L1C + Cloud Score+ antes del 28/03/2017 o cuando el inventario L2A del mes está vacío.
  - `solo_l2a`: serie homogénea.
- **Etiquetas**: umbrales por perfil `[etiquetas.L2A]` / `[etiquetas.L1C]`. El parámetro `mascara_agua_geojson` permite corroborar lagunas en L1C.
- **`ambiguos_bloquean`**: con `true` (valor del SDD), un componente ambiguo potencialmente positivo descarta la escena. El TOML usa `false`: se exportan solo los componentes aceptados y los ambiguos quedan como fondo. Con `true`, la ejecución 2016–2026 descartó los 18 meses visibles; con `false` se exportaron los 18.
- **Visibilidad** `[preseleccion]`: `min_util_sector_pct = 80` (SDD: 90). Con 90 solo 18 meses tenían escena visible; con 80 se exportan 28.

## Pruebas sin red

```
python -m pytest -q
```

Las pruebas usan datos sintéticos y un proveedor falso: calendario, enrutamiento L2A/L1C, ranking, reglas de etiqueta, empaquetado, cargador de Ultralytics, interrupciones y bloqueo. La regresión con la escena piloto del selector se omite si esa escena no está presente.

Las pseudoetiquetas indican cumplimiento de reglas, no verificación glaciológica. Por eso val y test miden concordancia con NDSI.
