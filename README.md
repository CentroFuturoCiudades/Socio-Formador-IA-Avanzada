# Reto — Escenas de playground usando **MP-GCN** 

## ¿Qué haremos y por qué?

Clasificar **una etiqueta por escena** en escenas de playground (`Transit`, `Social_People`, `Play_Object_Normal`, `Play_Object_Risk`, `Adult_Assisting`, `Negative_Contact`) usando **esqueletos 2D** y un **grafo panorámico** persona–objeto (**MP-GCN**).
MP-GCN modela **interacciones**: *intra-persona* (topología del cuerpo), *persona↔objeto* (manos↔columpio/lomas) y *inter-persona* (pelvis↔pelvis). Es **ligero**, **privado** y capta mejor **riesgo/uso del mobiliario** que un modelo por-persona con atención.

**Repo de referencia:** [MP-GCN](https://github.com/mgiant/MP-GCN) (ECCV’24, arXiv:2407.19497)

---

## Reglas prácticas 

* **FPS de proceso:** 12 → **T**≈60 (muestrea a **T=48** para el modelo)
* **K\_max personas por ventana:** 4
* **Forma de entrada (feeder estilo ST-GCN/MP-GCN):** `X ∈ [C, T, V', M]`

  * `V' = 17 + n_obj` (joints humanos + **centroides** de objetos por cámara)
  * **Streams:** `J` (joints), `B` (bones), `JM=ΔJ`, `BM=ΔB`
  * **Adyacencias:** `A0` (self), `A_intra` (humano + **obj↔manos**), `A_inter` (**pelvis↔pelvis**)

---

## Plan sugerido

### ✅ **Réplica el paper (3 semanas)**

**Objetivo:** entender MP-GCN y su *feeder* antes de tocar nuestro dato.

* Leer el **paper** (idea del grafo, 4 streams) y el **README** del repo.
* Clonar repo, crear ambiente e **instalar** dependencias.
* **Preparar** un dataset público (Volleyball / NBA; opcional **Kinetics-400** vía `pyskl`) como indica el repo.
* **Entrenar o inferir** con el *config* del repo y **reportar** Top-1/matriz de confusión.
* **Entender shapes** del *feeder* (`[C,T,V',M]`) e imprimirlos en un notebook.

**Entregables**

* Notebook “hello\_mpgcn.ipynb” (instalación + inferencia/entreno corto + shapes)

---

### ✅ Pipeline Playground (3 semanas)**

**Objetivo:** construir los **inputs panorámicos** a partir de nuestros videos.

* **Filtrar videos** (ROI) usando el notebook de la base de datos (PostGIS) y **seleccionar ≥100 escenas**:
  `video_id,camera,t_start,t_end,blob_path`

  > Puedes **prefiltrar** con el **VLM** y/o con el **# de detecciones** de la DB.
* **Esqueletos + tracking ligero** (YOLO/RTM-pose + ByteTrack/DeepSORT) en esas ventanas; **normalizar** por persona (cadera al origen, escala por torso).
  Guardar por ventana: **`[T,K_max,17,2]`**.
* **Objetos por cámara**: anotar manualmente **centroides** (0..1) de columpios/lomas en `configs/objects.yaml`.
* **Grafo panorámico**:

  * Expandir **`V → V' = 17 + n_obj`** (replicar centroides por frame/persona)
  * Añadir aristas **obj↔manos** (intra) y **pelvis↔pelvis** (inter)
  * Generar **`J/B/JM/BM`** y matrices **`A0/A_intra/A_inter`**
  * **Probar un forward** con el *feeder* del repo (batch pequeño)

**Entregables**

* `data/videos.csv` (≥100 filas)
* `data/npy/*.npy` o `.npz` por ventana (`[T,K_max,17,2]` normalizados)
* `configs/objects.yaml` (centroides por cámara)
* Script/función **`build_panoramic_graph`** y evidencia de **forward OK**

---

### ✅ **Etiquetado, entrenamiento y resultados (4 semanas)**

**Objetivo:** adaptar MP-GCN a playground y mostrar valor del grafo persona–objeto.

* **Etiquetado automático con VLM** → *scores por clase* → **argmax**; guardar `conf_weight = score_max` y aplicar **CORE-CLEAN** (p. ej., `score ≥ 0.8`).
  *(Si no usan VLM: curado humano ligero de 3–5 clases clave).*
* **Entrenamiento ligero**: congelar gran parte del backbone y entrenar cabezas/capas.


**Entregables**

* `train.csv` / `val.csv` (single-label; si VLM: con `conf_weight`)
* Checkpoint + **métricas**
* **Reporte** (2–4 páginas)

---



