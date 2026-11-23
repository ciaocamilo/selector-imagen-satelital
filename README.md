# Selector de imágenes satelitales Sentinel-2

Herramientas en Python para buscar, revisar y descargar imágenes Sentinel-2 del **Nevado Santa Isabel, Colombia**, y preparar etiquetas de nieve y hielo visible en formato **YOLO Segmentation** mediante Google Earth Engine.

El proyecto ofrece dos flujos:

- **Selector interactivo:** consulta escenas por área y fecha, permite revisar su visibilidad, descargar las seleccionadas y anotar polígonos manualmente en un notebook.
- **Dataset mensual automático:** selecciona como máximo una adquisición real por mes y genera pseudoetiquetas basadas en NDSI. El periodo configurado es 2016–2026.

Ambos producen imágenes RGB (B4/B3/B2) y espectrales (NDSI/NDVI/B4) sobre una misma malla, para utilizar las mismas etiquetas en las dos representaciones.

## Estructura

| Ruta | Contenido |
|---|---|
| [buscar_descargar_sentinel2.ipynb](buscar_descargar_sentinel2.ipynb) | Selector, descarga y anotación interactiva. |
| [dataset_mensual_yolo/](dataset_mensual_yolo/) | Generación mensual por consola o notebook. |
| [configuracion.toml](dataset_mensual_yolo/configuracion.toml) | Área de interés, periodo, calidad, etiquetas y particiones. |
| [mensual/](dataset_mensual_yolo/mensual/) | Implementación del flujo automático. |
| [tests/](dataset_mensual_yolo/tests/) | Pruebas del flujo mensual. |
| `datos_sentinel2/` | Descargas y anotaciones del selector. |
| `dataset_mensual_yolo/salida/` | Caché, estados, informes y versiones del dataset. |
| `dataset_mensual_yolo/salida_prueba/` | Resultados de ejecuciones cortas. |

## Instalación

Se requiere Python 3.11 o superior, una cuenta con acceso a Earth Engine y un proyecto de Google Cloud habilitado para utilizarlo.

Desde la raíz del repositorio, con Conda disponible:

```bash
conda create -n gee python=3.11 -y
conda activate gee
python -m pip install -r dataset_mensual_yolo/requirements.txt
python -m pip install jupyterlab ipykernel geemap ipyleaflet ipywidgets
python -m ipykernel install --user --name gee --display-name "Python (gee)"
```

Autentica Earth Engine antes de ejecutar el flujo mensual y sustituye `TU_PROYECTO` por tu ID de proyecto:

```bash
earthengine authenticate
earthengine set_project TU_PROYECTO
```

El proyecto también puede indicarse mediante `EE_PROJECT`, el parámetro `proyecto_ee` del selector o `[gee].proyecto` en la configuración mensual. Las credenciales se mantienen fuera del repositorio.

## Selector interactivo

```bash
jupyter lab buscar_descargar_sentinel2.ipynb
```

Selecciona el kernel **Python (gee)** y sigue este flujo:

1. Ajusta `fecha_objetivo`, `ventana_dias` y el área de interés en la celda de parámetros.
2. Ejecuta las celdas de consulta y revisa la tabla de candidatos.
3. Inspecciona las escenas sobre el mapa de la sección 5.
4. Marca las escenas y pulsa **Descargar marcadas** en la sección 6; verifica los productos en la sección 7.
5. Usa la sección 9 para dibujar, guardar y aprobar las etiquetas YOLO. La anotación de descargas existentes puede realizarse sin conexión a Earth Engine.

Con `ids_seleccionados = []` no hay descargas automáticas al ejecutar el notebook; la selección se realiza mediante la interfaz.

## Dataset mensual automático

Revisa primero [configuracion.toml](dataset_mensual_yolo/configuracion.toml). Desde la raíz del repositorio:

```bash
conda activate gee
cd dataset_mensual_yolo

# Prueba con dos meses
python -m mensual --config configuracion.toml --meses 2016-01,2024-01 --salida salida_prueba

# Periodo completo configurado
python -m mensual --config configuracion.toml

# Procesamiento con los datos disponibles en caché
python -m mensual --config configuracion.toml --sin-red
```

También puedes abrir [generar_dataset_mensual.ipynb](dataset_mensual_yolo/generar_dataset_mensual.ipynb), editar `CAMBIOS`, `SIN_RED` y `EJECUTAR` en la primera celda y ejecutar el cuaderno.

El flujo ordena candidatos por visibilidad, descarga y verifica las escenas y conserva la primera exportable de cada mes. Guarda el estado para reanudar ejecuciones y publica datasets versionados con particiones cronológicas `train`, `val` y `test` (70/15/15 del calendario por defecto).

```text
salida/
├── cache_escenas/       # Productos y decisiones de etiquetado
├── estado/             # Inventarios y estado por mes
├── ejecuciones/        # Configuración, meses.csv y candidatos.csv
└── datasets/
    ├── ultima.json
    └── vNNNN/
        ├── rgb/        # data.yaml, images/ y labels/
        └── espectral/  # data.yaml, images/ y labels/
```

Los archivos `data.yaml` contienen rutas absolutas. Si mueves un dataset, regenera sus rutas desde `dataset_mensual_yolo/`:

```bash
python -m mensual --regenerar-yaml salida/datasets/vNNNN
```

Consulta el [README del módulo mensual](dataset_mensual_yolo/README.md) para los estados de procesamiento, los criterios de selección, la reanudación y los ajustes de etiquetado.

## Pruebas

Desde `dataset_mensual_yolo/`, con las dependencias instaladas:

```bash
python -m pytest -q
```

Las pruebas emplean datos sintéticos y un proveedor falso. La regresión de la escena piloto requiere sus archivos locales y se omite si no están disponibles.

## Rutas largas en Windows

Los productos generados incluyen identificadores de escena y de procesamiento en sus rutas. Para clonar el repositorio con soporte de rutas largas:

```bash
git -c core.longpaths=true clone https://github.com/ciaocamilo/selector-imagen-satelital.git
```

En una copia local existente, ejecuta `git config core.longpaths true`. Una carpeta de trabajo corta también reduce la longitud total de las rutas.

## Alcance de las etiquetas

La clase es **nieve y hielo visible en la adquisición**; no representa por sí sola un límite de glaciar permanente. Las pseudoetiquetas automáticas siguen reglas espectrales y no equivalen a una validación glaciológica independiente. Las métricas sobre esas etiquetas miden concordancia con dichas reglas.

La configuración actual permite exportar componentes aceptados aunque existan otros ambiguos (`ambiguos_bloquean = false`); estos últimos quedan como fondo. Revisa este criterio antes de utilizar los datos para entrenar o evaluar modelos. La falta de observaciones y la presencia de nubes no justifican por sí solas una etiqueta negativa.

Datos de origen: **Copernicus Sentinel-2 (ESA)**, consultados y procesados mediante **Google Earth Engine**.
