
# Reto: Conductas en Playground con Esqueletos 

## 📌 Descripción

Construir un sistema que, a partir de **ventanas de video (3–5 s)** y **esqueletos 2D**, prediga **etiquetas multi-etiqueta** a nivel de **escena** en un playground:

* **Riesgo de uso:** `wrong_way_climbing`, `blocking_slide_exit`, `falling`
* **Interacción peligrosa:** `pushing_collision`, `adult_child_aggressive`
* **Social positiva:** `cooperative_play`, `adult_assisting`
* **Base:** `normal_play`

**Modelo core (obligatorio):** encoder temporal por persona → **agregador por atención** (set de personas) → **cabeza multi-label**.
**Stretch (opcional):** integrar **MP-GCN** (grafo panorámico) o añadir nodos de **objetos** (tobogán/columpio/estructura).

---

## 🧭 Estructura sugerida del repo

```
project/
  README.md
  notebooks/
    00_query_roi_and_azure.ipynb   # detecciones→ROI→Azure, descarga/listado
    01_make_windows.ipynb          # generar windows.csv (opcional si ya lo haces en 00)
    02_vlm_pseudolabels.ipynb      # auto-etiquetado (opcional)
  data/
    windows.csv                    # ventanas (t_start–t_end + blob_url_sas)
    train.csv                      # etiquetas (humanas o pseudo) + conf_weight
    val.csv
    npy/                           # una .npy por ventana [T,K_max,V,C]
  src/
    data/                          # loaders [T,K_max,V,C] + máscara
    model/                         # PersonEncoder, SceneAggregator, Head
    train/                         # loop de entrenamiento
    eval/                          # métricas
    vlm/                           # agregador JSON→CSV (CORE-CLEAN)
  configs/
    classes.yaml                   # minimal5 / full8
```

> Puedes empezar **solo** con `notebooks/00_query_roi_and_azure.ipynb` (ya incluido en tu repo): produce `windows.csv` y descarga clips desde Azure.

---

## 🔐 Configuración (credenciales)

**No** subas contraseñas ni SAS al repo. Usa variables de entorno o un `config.json` ignorado en git.

Variables típicas:

```bash
# PostGIS
export POSTGIS_DB=crowdcounting
export POSTGIS_USER=...
export POSTGIS_PASS=...
export POSTGIS_HOST=...
export POSTGIS_PORT=5432

# Azure Blob (SAS solo lectura+lista)
export AZ_ACCOUNT_URL="https://<account>.blob.core.windows.net"
export AZ_CONTAINER="<container>"
export AZ_SAS="sv=...&sp=rl&sig=..."   # sp=rl (Read+List)
```

---

## 🧱 Dependencias mínimas

```text
pandas
numpy
geopandas
psycopg2-binary
shapely
azure-storage-blob
tqdm
# Entrenamiento:
torch
scikit-learn
pyyaml
matplotlib
```

---

## 🗃️ Datos de entrada

* **PostGIS** con detecciones **ya filtradas por ROI** (área de juegos).
  Tabla ejemplo:
  `person_observed(id, id_person, lat, long, timestamp, geom, camera_name, coordinate_x, coordinate_y, tracklet_id, ...)`

* **Azure Blob Storage** con los videos (container + **SAS de lectura**).

---

## 🚦 Pipeline reproducible

### 1) De detecciones a **segmentos con presencia** (ROI)

En `00_query_roi_and_azure.ipynb`:

1. Carga ROI (`playgroundROI.gpkg`, EPSG:4326).
2. Consulta PostGIS → detecciones **dentro** del ROI + *timestamps*.
3. **Colapsa por segundo** (`timestamp.floor('S')`), cuenta personas (`nunique(tracklet_id)`).
4. Mantén segundos con `persons ≥ 1`.
5. Une segundos contiguos en **segmentos** `[seg_start, seg_end)` (suma +1s al final).
6. Descarta segmentos cortos `< 3 s`.

> Esto puede hacerse con SQL (CTEs) o Pandas (ver notebook).

### 2) Segmentos → **ventanas deslizantes** → `windows.csv`

Dentro de cada segmento:

* Ventana `W = 5 s` (o `3 s`) con **hop** `= 2.5 s` (50%).
* **Salida:** `data/windows.csv` con columnas:

```csv
video_id,camera,t_start,t_end,blob_url_sas
C01_2025_09_10_120000,cam1,2025-09-10T12:00:00Z,2025-09-10T12:00:05Z,https://<acc>.blob.core.windows.net/<cont>/<path>.mp4?<SAS>
```

> `blob_url_sas` se arma con `AZ_ACCOUNT_URL`, `AZ_CONTAINER`, `blob_path` y `AZ_SAS`.

### 3) Ventanas → **esqueletos** `.npy`

Para cada ventana (`t_start–t_end`):

* Extrae **T frames** uniformes (p.ej., `T=64–96`).
* Usa `K_max = 4–6` (máx. personas por ventana), `V=17` (COCO-17), `C=2` (x,y) o `3` (x,y,score).
* **Padding** con ceros para personas ausentes; el loader infiere **máscara**.
* **Normalización:** *root-centered* (cadera al origen) + *scale-invariant* (altura/torso).
* Guarda **una** `.npy` por ventana: **`[T, K_max, V, C]`** (float32).

> Si ya tienes pose/track por frame, **reúsalos**; si no, usa un pose ligero (YOLOv8-pose, RTM/ViTPose-lite) y track *light*.

### 4) (Opcional) Auto-etiquetado con VLM → **CORE-CLEAN**

Si no hay labels humanas:

1. Pasa ventanas a un **VLM** (2 vistas: RGB y *stick-figure*; 2 offsets).
2. Pide **JSON estricto** con probabilidades \[0..1] por etiqueta.
3. Ensemblado (promedio) + **CORE-CLEAN** (alta confianza/baja varianza).
4. Genera `data/train.csv` y `data/val.csv`:

```csv
npy_path,wrong_way_climbing,blocking_slide_exit,falling,pushing_collision,adult_child_aggressive,cooperative_play,adult_assisting,normal_play,conf_weight
data/npy/clip_0001.npy,0,0,1,0,0,0,0,0,0.95
```

**Prompt sugerido (multi-label, JSON estricto):**

```text
[SYSTEM] Responde SOLO con un JSON válido.
[USER]
Analiza el clip. Devuelve {"scores":{etiqueta:prob,...}} con probabilidades [0..1] para:
["wrong_way_climbing","blocking_slide_exit","falling",
 "pushing_collision","adult_child_aggressive",
 "cooperative_play","adult_assisting","normal_play"]

Reglas:
- Si no hay tobogán visible: wrong_way_climbing y blocking_slide_exit ≤ 0.05.
- Si es dudoso: usa 0.0–0.2 (no inventes).
- No texto extra fuera del JSON.
```

### 5) Entrenamiento (Core)

**Arquitectura:**

* `PersonEncoder` (TemporalConv/LSTM pequeña) → `z_i ∈ R^d` por persona.
* `SceneAggregator` (attention pooling o DeepSets con máscara) → `z_scene`.
* `Head` (MLP multi-label) → logits → sigmoid.

**Pérdida:** `BCEWithLogits` + *class-weights* (clases raras ↑) + `sample_weight=conf_weight`.
**Hipers sugeridos:** `d=128`, `heads=4`, `layers=2`, `T=64`, `K_max=4–6`, batch `8–16`, LR `1e-3 → 1e-4`, dropout `0.1`.
**Métricas:** mAP, F1 macro; **Recall** en `falling`/`adult_child_aggressive`; **FPR ≤ 5%** en clases de riesgo.
**Calibración:** umbrales por clase (val) + **suavizado temporal** (EMA) para ventanas solapadas.

### 6) Stretch (opcional)

* **Objetos**: añadir centroides de `slide/swing` como “personas especiales” (ocupan slots en `K_max`).
* **MP-GCN**: migrar al *panoramic graph* (streams J/B/JM/BM + atención persona-tiempo).
* **Few-shot**: 50–100 ventanas por clase rara.
* **Auto-training**: re-entrenar la **head** con predicciones ≥ 0.90 no vistas.

---

## 🧩 Ontologías

`configs/classes.yaml` sugerido:

```yaml
full8:
  - wrong_way_climbing
  - blocking_slide_exit
  - falling
  - pushing_collision
  - adult_child_aggressive
  - cooperative_play
  - adult_assisting
  - normal_play

minimal5:
  - falling
  - wrong_way_climbing
  - pushing_collision
  - cooperative_play
  - normal_play
```

---

## ⚒️ Comandos útiles (Azure / AzCopy)

```bash
# Listar blobs
azcopy ls "$AZ_ACCOUNT_URL/$AZ_CONTAINER?$AZ_SAS"

# Descargar un prefijo
azcopy copy "$AZ_ACCOUNT_URL/$AZ_CONTAINER/playground/2025/*?$AZ_SAS" ./videos/ --recursive=true

# Sincronizar (reanudar descargas)
azcopy sync "$AZ_ACCOUNT_URL/$AZ_CONTAINER/playground/2025/?$AZ_SAS" ./videos/ --recursive=true
```

---

## 🗓️ Calendario (10 semanas, guía)

1. **Exploración & datos** (ROI→detecciones→segmentos).
2. **Ventanas** (`windows.csv`) y set-up de `.npy`.
3. **PersonEncoder + Aggregator + Head** (forward OK).
4. **VLM pseudo-labels → CORE-CLEAN** → `train/val.csv`.
5. **Entrenamiento baseline** + reporte preliminar.
6. **Calibración + suavizado**.
7. **Stretch:** objetos o **MP-GCN** (si hay tiempo).
8. **Few-shot** (opcional).
9. **Robustez & ablations** (por cámara, #personas; J vs J+B+JM+BM; con/sin objetos).
10. **Demo final** (Streamlit) + curvas PR/ROC + reporte.

---

## 🎯 Metas y rúbrica (sugeridas)

* **Semana 5:** F1 macro ≥ **0.45** (val).
* **Semana 8:** Recall `falling` ≥ **0.60** con **FPR ≤ 5%** (riesgos).
* **Semana 10:** demo fluida, **curvas PR/ROC**, ablations mínimas y discusión de limitaciones.

**Rúbrica:**

* Infra & datos (20%) — loaders reproducibles, normalización, notebook de exploración.
* Baseline & training (35%) — entrenamiento estable, métricas razonables, manejo de desbalance.
* Calibración & evaluación (25%) — umbrales por clase, FPR bajo, splits por cámara.
* Demo & reporte (20%) — app clara, visualizaciones, ablations y futuro.

---

## 🔗 Referencias

* **MP-GCN (Panoramic Graph):**
  Paper: *Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph* (arXiv:2407.19497)
  Repo: [https://github.com/mgiant/MP-GCN](https://github.com/mgiant/MP-GCN)

* **ST-GCN (clásico) y variantes:** para quien quiera profundizar en GCN de esqueletos.

---

## ❓FAQ

**¿Sin etiquetas humanas?**
Usa VLM → **CORE-CLEAN** para generar `train.csv` con `conf_weight`.

**¿GPU pequeña?**
`T=64`, `K_max=4`, `d=128`, batch `8–12`, ontología `minimal5`.

**¿Puedo aprobar sin objetos ni grafo?**
Sí. El **Core** (encoder por persona + agregador + head) es suficiente; el grafo es **stretch**.

---

> **Checklist día 1–2**
>
> 1. Ejecuta `00_query_roi_and_azure.ipynb` y genera **segmentos** → **ventanas** → `windows.csv`.
> 2. Crea `.npy` por ventana (`[T,K_max,V,C]`) normalizados.
> 3. (Opcional) VLM → CORE-CLEAN → `train/val.csv`.
> 4. Entrena el **baseline** (5–8 épocas) y reporta F1/Recall/FPR.

---
