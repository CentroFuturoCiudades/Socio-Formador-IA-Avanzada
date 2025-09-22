# Reto  — Escenas de playground con **MP-GCN** 

## 1) ¿Qué vamos a construir?

Un clasificador **single-label por ventana (3–5 s)** que decide entre escenas:
`Transit`, `Social_People`, `Play_Object_Normal`, `Play_Object_Risk`, `Adult_Assisting`, `Negative_Contact`.

**Por qué así:** con esqueletos 2D + nodos de objetos (columpios/lomas) es **ligero**, **privado** y captura **interacciones** (persona↔persona y persona↔objeto) que son justo las que importan para **riesgo** y **uso del equipamiento**.

---

## 2) Modelo base: **MP-GCN** (qué es y por qué)

**MP-GCN** representa cada ventana como **grafo espacio-temporal**:

* **Nodos:** joints humanos + **centroides** de objetos del playground (estáticos por cámara).
* **Aristas:**
  *Intra* (topología esqueleto), *Persona↔Objeto* (obj↔manos), *Inter-persona* (pelvis↔pelvis).
* **Cuatro “streams”** de entrada (como el paper): `J` (joints), `B` (bones), `JM=ΔJ`, `BM=ΔB`.
* **Entrada típica por muestra:** `X ∈ [C, T, V', M]`
  (`T`=frames, `V'`=17+j\_objetos, `M`=K\_max personas, `C`=canales).

**Ventaja clave:** el grafo **codifica relaciones** (contacto, bloqueo, crowding) *dentro* del backbone, sin heurísticas.

**Referencia oficial:**
Repo: `https://github.com/mgiant/MP-GCN` (ECCV’24) · Paper: arXiv:2407.19497

---

## 3) Estrategia general (por qué en este orden)

1. **Replicar** en dataset público (Volleyball / NBA; opcional Kinetics-400) → entiendes **feeder** y **streams** sin ruido.
2. **Preparar nuestro dato** (ROI→ventanas, esqueletos, objetos) → construyes el **input panorámico**.
3. **Etiquetar simple** (VLM que ya está desplegado **o** curado ligero) → obtienes supervisión inicial.
4. **Fine-tuning ligero** + **ablation mínima** → demuestras valor del grafo **persona-objeto**.

---

## 4) Plan de 10 semanas (resumido y flexible)

**S1 — Intro + setup + “hello world”**

* Leer la idea de MP-GCN (nodos/aristas/streams).
* Clonar repo y correr **inferencias/entrenamiento corto** en Volleyball/NBA (o preparar K400).

**S2 — Réplica pública con métricas**

* Entreno breve, ver **Top-1** y **matriz de confusión**.
* Entender shapes del *feeder* (`[C,T,V',M]`).

**S3 — Ventanas (Playground, ROI)**

* 5 s (hop 2.5 s) → `windows.csv` con `video_id,camera,t_start,t_end,blob_url`.
* Muestra visual de 20 ventanas.

**S4 — Esqueletos + tracking (subset)**

* Pose ligera (YOLO/RTM-pose) + ByteTrack/DeepSORT en **400–800** ventanas.
* Normalizar por persona (cadera al origen, escala por torso).
* Guardar **`[T,K_max,17,2]`** (sugerido **T=48**, **K\_max=4**, **FPS=12**).

**S5 — Objetos por cámara (manual, rápido)**

* YAML con **centroides normalizados (0..1)** de columpios/lomas visibles.
* (Son nodos estáticos; se replican en todos los frames).

**S6 — Builder panorámico (entrada a MP-GCN)**

* Expandir `V → V' = 17 + n_obj`.
* Añadir aristas: **obj↔manos** (intra), **pelvis↔pelvis** (inter).
* Generar `J/B/JM/BM` y matrices **A0/A\_intra/A\_inter**.
* Probar un **forward** con el *feeder* del repo.

**S7 — Etiquetado single-label (rápido)**

* **Opción A (recomendada):** VLM → scores por clase → **argmax**; guardar `conf_weight = score_max` y filtrar CORE-CLEAN (p. ej., score ≥ 0.8).
* **Opción B:** curado humano ligero (80–120 ventanas por clase clave).

**S8 — Fine-tuning ligero (Playground)**

* Congelar gran parte del backbone; entrenar **cabeza** (+1–2 bloques si hay tiempo).
* Loss CE con **class-weights**. Métricas iniciales.

**S9 — Ajustes + ablation mínima**

* Comparar: **sin objetos** / **sin inter-persona** / **panorámico completo**.
* Reportar mejora en `Play_Object_Risk` y `Negative_Contact`.

**S10 — Demo + reporte**

* Streamlit sencillo: video + sticks + nodos + etiqueta y confianza por ventana.
* Reporte breve: qué funciona y por qué; límites y siguiente paso.

---

## 5) Parámetros por defecto (para no atascarse)

* Cámaras: **2** (una de columpios, una de lomas).
* Ventana: **5 s**, **FPS de proceso = 12** ⇒ `T_raw≈60` → muestrear a **T=48**.
* `K_max=4`.
* Clases: 4–6 (las de arriba).
* Entrenamiento: **head-only** primero.

---

## 6) Entregables mínimos

* **Réplica pública:** notebook + métricas.
* `windows.csv` (Playground).
* `.npy/.npz` por ventana (`[T,K_max,17,2]` normalizados).
* `objects.yaml` (centroides por cámara).
* **Builder** panorámico (4 streams + A) con *forward* exitoso.
* `train.csv`/`val.csv` (VLM o curado).
* Checkpoint + **ablation mínima**.
* Demo (Streamlit) + reporte corto.

---

## 7) Notas rápidas

* El **VLM** sólo se usa para **generar etiquetas** (offline). El modelo final **NO** usa RGB: **sólo esqueletos + objetos**.
* Si el tracking es ruidoso: baja `K_max` a 3–4 y descarta ventanas con <70% de keypoints válidos.
* Si falta tiempo para objetos: empieza con **columpios** (asientos) y agrega lomas después.

> Meta didáctica: entender **por qué** el grafo **persona-objeto** mejora la detección de **riesgo** e **interacción social**, y dejar un pipeline **reproducible** que puedan extender.
