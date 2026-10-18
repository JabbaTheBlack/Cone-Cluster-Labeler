# Detection kísérletek — 2026. október

Branch: `feature/detection-enhancement` (2026-10-17-én ágaztatva a `feature/color-classification`-ből).
Feladat: bináris osztályozás, **kúp (1) / nem kúp (0)** LiDAR klaszterekre. A címke a `labeled_clusters.json` `is_cone` mezője; a szín itt nem számít.

Ez a dokumentum minden eredményt és a futtatás módját rögzíti úgy, hogy a modellek nélkül is reprodukálható legyen. A számok a `logs/detection/` alatti logokból származnak.

---

## 1. Adat

### 1.1 `Dataset/Processed` (eredeti)

| Rész | Klaszter | Kúp | Nem kúp | Intensity skála |
|---|---|---|---|---|
| `Detection/` (Skidpad, Acceleration, AutoX_7, AutoX_7_2) | 2371 | 1164 | 1207 | **0–1** |
| `Color/` (10 felvétel) | 4284 | 4177 | 107 | **2–282 (nyers)** |
| **Összesen** | **6655** | 5341 | 1314 | |

- A nem kúpok 69%-a (906) egyetlen trackből, a `Detection/Skidpad`-ből jön.
- A klaszterek nagyon kicsik: medián 9 pont, 90%-uk 32 pontnál kevesebb.

### 1.2 Az intensity szivárgás és a `Dataset/Processed_FrameNorm`

**Probléma.** A Detection klasztereket (2025-12-27, `Dataset.zip`) egy régebbi perception pipeline exportálta, amely az intensityt **frame-enként min–max normálta** 0–1-re. A frame-ek 98%-ában a maximum pontosan 1, 97%-ában a minimum pontosan 0. A Color klaszterek az `extract_clusters.py`-jal készültek a `/perception/landmarks` topicból, nyers intensityvel. Ezt nem ez a repó normálta, és a PCD-kből nem fordítható vissza, mert a frame-enkénti osztó elveszett.

Mivel a Color mappa 97,5%-a kúp, és a nem kúpok 92%-a a Detection mappában van, a nyers intensity gyakorlatilag kiolvassa a forrást, és ezzel a címkét is:

- Az intensity forrás-AUC-ja kúpokon: **1,000**.
- Az intensity_std osztály-AUC-ja: 0,924 a teljes adaton, de csak 0,755 a Detection mappán belül.

**Megoldás.** A Color klasztereket ugyanúgy frame-enként min–max normáltam, mint ahogy a Detection adatot a régi pipeline:

```
I_norm = (I − frame_min) / (frame_max − frame_min)
```

Itt a `frame_min` és a `frame_max` a mappa adott frame-jéhez tartozó **összes** PCD-ből számolódik (`scan_<id>_frame_<n>_cluster_*.pcd`), a címkézetlenekből is. A Detection fájlok változatlanul másolódnak. Az eredmény 667 frame és 4948 normált PCD.

```bash
python scripts/detection/frame_normalize.py Dataset/Processed Dataset/Processed_FrameNorm
```

A script bájtra pontosan a mostani `Dataset/Processed_FrameNorm`-ot állítja elő (ellenőrizve, `diff -rq`).

**Hatás:**
- A forrás-AUC kúpokon 1,000-ról kb. **0,63–0,67**-re esett.
- A teszten végzett, forráson belüli és trackenkénti keverés ugyanakkora F1-esést okoz, mint a teljes keverés. A modellek tehát a klaszterek saját intensity-mintázatát használják, nem a felvétel szintjét.

**Minden alábbi eredmény a `Dataset/Processed_FrameNorm`-on készült**, kivéve ahol ezt külön jelzem.

### 1.3 Train/test split: `balanced5track`

Az automatikus split (a `grouped_train_test_split_best_effort`) a 20%-os arányt részesítette előnyben a kúparánnyal szemben. Az eredmény egy 90/10-es teszt lett, mindössze 133 nem kúppal. Ehelyett kézzel választott, kiegyensúlyozott splitet használunk:

| | Minta | Nem kúp arány |
|---|---|---|
| **Test** (5 track): `Ekol_09_04_15`, `fireup_11_28_22`, `fireup_11_28_25`, `Acceleration`, `AutoX_7_2` | 1554 (23,35%) | 19,82% (308 db) |
| Train (9 track) | 5101 | 19,72% |

Preset a `train_detection.py`-ban: `--test-tracks balanced5track` (`SPLIT_PRESETS`). A korábbi neve „C split” volt.

### 1.4 Távolsági torzítás (még nincs kezelve)

**Nincs címkézett kúp 21,4 m-en túl.** A kúpok aránya a távolság szerint:

| Távolság | 0–5 m | 5–10 m | 10–15 m | 15–20 m | 20–30 m |
|---|---|---|---|---|---|
| Kúp arány | 96% | 92% | 77% | 12% | 0,5% |

A modellek ebből azt tanulhatják meg, hogy ami messze van, az nem kúp. **Javaslat** (még nincs megvalósítva): egy kemény 25 m-es határ a modell előtt. Ez 0 kúpot és 84 nem kúpot (6,4%) szűrne ki. Mellé egy távolság nélküli modell, távolsággal korrigált density-vel.

---

## 2. Feature-ök (RF és XGB)

`src/train/detection/features.py`:

- **`extract_features`** (7 feature, a régi baseline): height, width, depth, aspect_ratio, density, volume, distancefromlidar.
- **`extract_extended_features`** (13 feature): a júliusi 12 feature-os modell képletei (`3e1c34b1`) plusz a distancefromlidar.

| Feature | Képlet |
|---|---|
| height | `z_max − z_min` |
| width / depth | **`max(σx, σy)` / `min(σx, σy)`** (dinamikus, a `d6992c35` szerint) |
| aspect_ratio | `height / width` |
| volume | `height · width · depth` |
| density | `num_points / volume` |
| num_points | pontszám |
| compactness | a pontok középponttól mért távolságának `std / mean` aránya |
| intensity_mean / intensity_std | a pontok intensityjének átlaga és szórása |
| linearity / planarity | a kovariancia sajátértékeiből: `(λ1−λ2)/λ1`, `(λ2−λ3)/λ1` |
| distancefromlidar | a középpont távolsága az origótól |

**Javított hiba:** a 13 feature-os készlet először a `3e1c34b1`-ből a tengelyhez kötött x/y kiterjedést vette át width/depth-nek. Ez visszalépés volt a `d6992c35` dinamikus változatához képest, amit a 7 feature-os modell és a C++ oldal is használ. Ezeket a futásokat 2026-10-18-án töröltük. Az alábbi eredmények már a dinamikus definícióval készültek.

---

## 3. Iteratív feature selection (RF és XGB)

**Beállítás:**
- `Dataset/Processed_FrameNorm`, balanced5track split.
- Háromlépcsős GridSearch, `f1_macro` pontozással, 5-fold `GroupKFold` a train trackeken.
- **Kivágási szabály:** minden kör után a két leggyengébb, 1,5% alatti importance-ú feature esik ki (a modell saját importance-a szerint). 3% fölötti feature soha nem esik ki. Ha nincs 1,5% alatti feature, az út megáll.
- Az RF és az XGB külön úton halad.

```bash
python scripts/detection/auto_eliminate.py --model rf
python scripts/detection/auto_eliminate.py --model xgb
```

Ez egyenként ugyanaz, mint:

```bash
python src/train/detection/train_detection.py --dataset Dataset/Processed_FrameNorm --model rf \
  --features height,width,...  --test-tracks balanced5track --output-name rf_adaptive_balanced5track
```

**Eredmények** (teszt, küszöb 0,5; FP = nem kúp kúpnak jósolva, FN = kihagyott kúp):

| Modell | Sima F1 | Macro F1 | Nem kúp recall | Nem kúp precision | FP | FN | Kivágva utána |
|---|---|---|---|---|---|---|---|
| RF 13 | 0.9601 | 0.888 | 73.4% | 91.9% | 82 | 20 | compactness (0.20%), planarity (0.23%) |
| RF 11 | 0.9584 | 0.884 | 73.4% | 90.4% | 82 | 24 | num_points (0.37%), linearity (0.41%) |
| RF 9 | 0.9600 | 0.888 | 73.7% | 91.5% | 81 | 21 | intensity_mean (1.12%) |
| RF 8 | 0.9593 | 0.886 | 73.1% | 91.5% | 83 | 21 | — (megállt, minden > 3%) |
| XGB 13 | 0.9607 | 0.887 | 71.4% | 94.4% | 88 | 13 | depth (0.66%), compactness (0.67%) |
| XGB 11 | 0.9599 | 0.884 | 70.8% | 94.4% | 90 | 13 | volume (0.54%), linearity (0.61%) |
| **XGB 9** | **0.9625** | **0.894** | 73.4% | 94.2% | 82 | 14 | num_points (0.29%), planarity (0.44%) |
| XGB 7 | 0.9607 | 0.886 | 70.8% | 95.2% | 90 | 11 | — (megállt) |

**Végső készletek:**
- RF 8: height, width, depth, aspect_ratio, density, intensity_std, volume, distancefromlidar.
- XGB 7: height, width, aspect_ratio, density, intensity_mean, intensity_std, distancefromlidar.
- XGB 9 (a legjobb): height, width, aspect_ratio, num_points, density, intensity_mean, intensity_std, planarity, distancefromlidar. Paraméterek: 50 fa, `max_depth=2`, `learning_rate=0.3`.

**Baseline** (régi 7 feature, nyers `Dataset/Processed`, balanced5track):

| Modell | Sima F1 | Macro F1 | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|
| RF 7 | 0.9551 | 0.866 | 65.9% | 105 | 11 |
| XGB 7 | 0.9532 | 0.861 | 65.3% | 107 | 14 |

**Megállapítások:**
- **A görbe lapos:** kb. 6 feature elhagyható veszteség nélkül. Mindkét modellből a compactness, a planarity, a linearity és a num_points esik ki elsőként.
- **A beépített importance és a valós hatás eltér.** A permutációs teszt szerint az aspect_ratio a döntő feature. A beépített importance túlbecsüli a density-t (XGB: az első fa gyökérvágásának gainje) és a distancefromlidar-t (RF).
- **Az intensity szerepe kicsi** (kb. 4–7% importance), de valódi.

Ábrák:
- `figures/detection/adaptive_elimination_curve.png`
- `figures/detection/adaptive_final_importance.png`
- körönként: `figures/detection/<rf|xgb>_adaptive_balanced5track/<N>_features/importance_summary.png`

---

## 4. PointNet

### 4.1 Változások a `pointnet_model.py`-ban

A modell felépítése Mező-Kerekes Márk kódja. A tanítást én módosítottam:

- **Early stopping validáción.** Korábban a teszthalmaz döntött, ami szivárgás volt. A validáció a train trackekből osztálykiegyensúlyozottan választódik (`_balanced_group_split`): **AutoX_7 + Ekol_09_04_16**, 437 minta, ebből 74 nem kúp.
- **Valódi augmentáció:** minden epoch új pontmintát és új zajt kap.
- **Mintavétel:** ha egy klaszterben legfeljebb `num_points` pont van, mind bekerül, és a hiányzó helyek ismétléssel töltődnek. Mivel a max-pooling nem érzékeny az ismétlésre, a kiértékelés így determinisztikus. A 4.4-es ablation még a régi mintavétellel futott.
- **`arch` opció:**
  - `"modified"`: az eredeti (4×4-es T-Net az x, y, z, intensity csatornákon, egy 64-es réteg).
  - `"paper"`: Qi és társai, 2017. A 3×3-as T-Net csak x, y, z, az intensity a transzformáció után csatlakozik, 64-64 réteg, és 0,001-es ortogonalitás-regularizáció a feature T-Net-en.
- **Log-mezők:** ugyanazok, mint az RF-nél és az XGB-nél.
- **CLI:** `--arch`, `--no-intensity`, `--seed`, `--no-validation`, `--epochs`, `--output-name`.

**Környezet:** `torch 2.14.1+cu126` a venv-ben, ami a HDD-n van. A telepítéshez a `PIP_CACHE_DIR` és a `TMPDIR` a `/media/praksz/Delta/FRT/` alá mutatott, mert a rendszer SSD-n csak kb. 22 GB szabad hely van. GPU: RTX 3080.

```bash
TMPDIR=/media/praksz/Delta/FRT/.tmp PIP_CACHE_DIR=/media/praksz/Delta/FRT/.pip-cache \
  cluster_labeler_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 4.2 Első futás (módosított, seed 42)

```bash
python src/train/detection/train_detection.py --dataset Dataset/Processed_FrameNorm --model pointnet \
  --test-tracks balanced5track --output-name pointnet_balanced5track
```

| Változat | Sima F1 | Macro F1 | Nem kúp recall / precision | FP | FN |
|---|---|---|---|---|---|
| validációval (4664 tanító minta), 21 epoch, a legjobb a 6. | 0.9566 | 0.867 | 64.0% / 99.0% | 111 | 2 |
| teljes train (5101), 6 epoch, `--no-validation --epochs 6` | 0.9550 | 0.862 | 63.3% / 98.0% | 113 | 4 |

### 4.3 Küszöbhangolás

A küszöböt **a validáción** választjuk (a legjobb macro F1), és utána alkalmazzuk a teszten. Ugyanazon a modellen:
- küszöb 0,50: F1 0.9566, macro 0.867 (FP 111, FN 2);
- **küszöb 0,55: F1 0.9627, macro 0.891** (FP 90, FN 6).

```bash
python scripts/detection/pointnet_threshold_eval.py OUT.png "Cím" "címke=output_name" ...
```

### 4.4 Intensity ablation (módosított szerkezet, 5 seed, régi mintavétel)

```bash
python src/train/detection/train_detection.py ... --model pointnet --seed S [--no-intensity] \
  --output-name pointnet_ablation/{xyzi|xyz}_seedS
```

| Bemenet | Sima F1 (hangolt küszöb) | Macro F1 (hangolt) | FP | FN |
|---|---|---|---|---|
| x, y, z, intensity | 0.9636 ± 0.0038 | 0.897 ± 0.013 | 79.6 | 13.6 |
| x, y, z | 0.9578 ± 0.0027 | 0.874 ± 0.011 | 101.0 | 8.2 |

**Az intensity segít** (+0,023 macro F1). A PointNet az intensityt klaszterenként 0–1-re normálja, ami minden frame-enkénti skálázást kiolt. Nála ezért **a nyers és a frame-normált adat ugyanazt a bemenetet adja**, vagyis a szivárgás nem érinti, és az autón sem kell frame-normálás.

### 4.5 Hiperparaméter-keresés (cikk szerinti szerkezet)

```bash
python scripts/detection/pointnet_search.py --configs 20 --seeds 3 --final-seeds 5
```

- **Fix:** `arch="paper"`, intensityvel, 0,001-es regularizáció.
- **Keresési tér:** learning rate {3e-4, 1e-3, 3e-3}, lr_step {0, 20}, dropout {0,3; 0,5}, weight decay {0, 1e-4, 1e-3}, num_points {16, 32, 64}, türelem {15, 30}, loss {CE, focal}.
- **Menet:** 20 kombináció, az első a cikk alapbeállítása; kombinációnként 3 seed. A pontszám a validációs macro F1, hangolt küszöbbel.

**Eredmény:** egyik kombináció sem volt jobb a cikk alapbeállításánál a validáción.
- A cikk alapbeállítása: 0.9765 ± 0.0020.
- A második helyezett: 0.9750 ± 0.0091 (dropout 0,5, weight decay 1e-3, 16 pont, focal).

A validációs macro F1 szinte minden kombinációnál 0,96 és 0,98 között van. 74 nem kúp mellett ez gyenge szelekciós jel. A teljes lista: `logs/detection/pointnet_search/search_results.json`.

> Megjegyzés: a script a „hangolt” változatnak az alapbeállítástól eltérő legjobbat választja. Ezért a lenti `paper_tuned` valójában a második helyezett.

### 4.6 Végső összevetés (5 seed, új mintavétel, a küszöb a validáción hangolva)

| PointNet | Sima F1 | Macro F1 | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|
| **Módosított (Márk)** | **0.9665 ± 0.0023** | **0.907 ± 0.007** | 77.3% | 69.8 | 15.6 |
| Módosított, küszöb 0,5 | 0.9635 ± 0.0008 | 0.895 ± 0.002 | 72.1% | 86.0 | 7.8 |
| Cikk szerinti, alap | 0.9580 ± 0.0042 | 0.876 ± 0.014 | 67.9% | 98.8 | 9.6 |
| Cikk szerinti, „hangolt” | 0.9580 ± 0.0022 | 0.877 ± 0.007 | 68.5% | 97.0 | 11.2 |

A választott küszöbök a módosított modellnél: 0.53, 0.57, 0.65, 0.53, 0.58.

**Csatorna-importance** (macro F1 esés, ha a csatornát a klaszterek között összekeverjük, seed 0):

| Csatorna | Módosított | Cikk szerinti |
|---|---|---|
| x | 0.116 | 0.123 |
| y | 0.100 | 0.118 |
| z | **0.154** | 0.048 |
| intensity | **0.067** | **0.005** |

A cikk szerinti modell gyakorlatilag nem használja az intensityt. A módosított a 4×4-es T-Net-ben a geometriával együtt dolgozza fel, és a z is fontosabb neki. **Hipotézis:** a fényvisszaverő csík magassága a jel.

Ábrák:
- `figures/detection/pointnet_search/final_comparison.png`
- `figures/detection/pointnet_ablation/intensity_ablation.png`
- `figures/detection/<output_name>/channel_importance.png`
- `figures/detection/pointnet_balanced5track/threshold_tuning.png`

---

## 5. Összesítés

| Modell (balanced5track teszt) | Sima F1 | Macro F1 | Megjegyzés |
|---|---|---|---|
| **PointNet módosított**, hangolt küszöb | **0.9665** | **0.907** | 5 seed átlaga |
| XGB 9 (adaptív) | 0.9625 | 0.894 | küszöb 0,5 |
| RF 9 / RF 13 (adaptív) | 0.960 | 0.888 | küszöb 0,5 |
| PointNet cikk szerinti | 0.958 | 0.876 | 5 seed átlaga |
| RF 7 baseline | 0.9551 | 0.866 | nyers adat, küszöb 0,5 |

**Fenntartás:** a fa modellek küszöbe még nincs hangolva. A korrekt összevetéshez ugyanígy, a validáción kellene hangolni.

## 6. Nyitott kérdések

- Távolság: kemény 25 m-es határ, vagy távolság nélküli modell korrigált density-vel.
- Küszöbhangolás az RF és az XGB modelleken.
- Melyik modell menjen a C++ oldalra. A PointNethez nem kell intensity-normálás; a fa modellekhez a bővített feature-ök és a frame-normálás kell.
- PointNet pontszám-csatorna: a klaszter eredeti pontszáma most elvész.

## 7. Fájlok

| Mi | Hol |
|---|---|
| Tanító kód | `src/train/detection/{features,train_detection,rf_model,xgb_model,pointnet_model}.py` |
| Scriptek | `scripts/detection/{frame_normalize,auto_eliminate,plot_importance,pointnet_search,pointnet_threshold_eval,pointnet_channel_importance}.py` |
| Logok (JSON soronként) | `logs/detection/<output_name>/training_log.txt`, plusz `threshold_tuning.json`, `permutation_importance.json` |
| Kivágás összesítő | `logs/detection/auto_eliminate/` |
| PointNet keresés | `logs/detection/pointnet_search/{search_results.json,best_config.json,search_progress.txt}` |
| **Kiválasztott PointNet** | `models/detection/pointnet_search/final_modified_seed0/` (+ a logja és ábrái). Az 5 módosított seed közül a validációs sima F1 szerint a legjobb (0,9918 @0,5; 0,9932 @0,53); teszten @0,53: F1 0,9668, macro F1 0,906 |
| A többi PointNet futás | A logjaik és ábráik nincsenek a gitben, minden számuk itt van: `docs/detection/pointnet_runs_summary.md` |
| Modellek | A gitben a PointNetekből csak a kiválasztott van; a többi újratanítható |

---

# Függelék — a teljes munkamenet (2026-10-17 – 2026-10-18)

Ez a rész a beszélgetésben elhangzott minden lényeges információt rögzíti, időrendben. A korábbi futások egy részét azóta töröltük vagy felülírtuk. A számaik itt maradnak meg, de **nem összevethetők** a fenti eredményekkel: más split, más adat, illetve tengelyhez kötött width/depth.

## A. Kiindulás

- A color munka (blue/orange kísérletek, profile_xgb, docs) commit nélkül, **git stash-ben** van: `color-classification WIP: blue/orange experiments, profile_xgb retrain, docs (2026-10-17)`. Visszaállítás: `git switch feature/color-classification && git stash pop --index` (előtte `git stash list`).
- **A régi RF tanító** (`rf_model.py`) értelmezése:
  - 7 geometriai feature, StandardScaler (a C++ export miatt);
  - háromlépcsős GridSearch (coarse 640 kombináció, medium, fine), `f1_macro` szerint, `GroupKFold(5)`, `class_weight='balanced_subsample'`;
  - mentés `.pkl` és C++-hoz `.bin` formátumban (scaler mean/std, fánként a csomópontok).
- **Talált hibák a régi kódban:**
  1. `max_depth=None` esetén a medium lépés összeomlik. **Javítva.**
  2. A medium lépés a mélységet 18-nál levágja (nincs javítva).
  3. A végső CV `class_weight` nélküli modellt mér, bináris F1-gyel, a GridSearch pedig `f1_macro`-val (nincs javítva).
  4. A végső CV a teljes X-en fut, a tesztet is beleértve (nincs javítva).
  5. A régi logsorok más rácsdefinícióval készültek.
  6. A régi 7 feature-os változatban a width/depth szórás, a height kiterjedés.
- **A régi RF logok** (8 futás) mind 4435 mintán készültek. Ez = Detection 2371 + Color 2064, vagyis a Ground Filter Reclassification mappa nélkül, ami 2026-10-05-én került be a `47fe1f37` commitban (2220 minta, ebből 2213 kúp). Az eredmények: F1 0,947–0,971, CV F1 kb. 0,968.

## B. Futások a régi, automatikus split-tel (nyers `Dataset/Processed`, 6655 minta)

A teszt itt 1331 minta, ebből 1198 kúp és 133 nem kúp (90/10). A teszttrackek: AutoX_7, fireup_11_28_22, Ekol_09_04_15, Ekol_09_04_16.

| Modell | Sima F1 | Macro F1 | Nem kúp recall | Konfúziós mátrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|
| RF 7 (régi feature-ök) | 0.9785 | ≈0.882 | 71.4% | [[95, 38], [14, 1184]] |
| RF 13 (tengelyhez kötött) | 0.9767 | 0.858 | 61.0% | [[81, 52], [5, 1193]] |
| RF 11 | 0.9779 | 0.867 | 63.2% | [[84, 49], [5, 1193]] |
| RF 9 | 0.9758 | 0.854 | 61.0% | [[81, 52], [7, 1191]] |
| RF 8 (volume nélkül) | 0.9795 | 0.879 | 66.2% | [[88, 45], [5, 1193]] |
| XGB 13 | 0.9806 | 0.892 | 72.2% | [[96, 37], [10, 1188]] |
| XGB 11 | 0.9760 | 0.870 | 70.7% | [[94, 39], [19, 1179]] |
| XGB 9 | 0.9729 | 0.840 | 60.2% | [[80, 53], [13, 1185]] |
| XGB 8 | 0.9713 | 0.829 | 57.9% | [[77, 56], [14, 1184]] |

- Az RF 7 a teljes adaton: `max_depth=16`, 270 fa, CV F1 0,978 ± 0,015. Felülírta a gitben lévő `models/detection/random_forest` modellt.
- A GridSearch belső F1 macro értéke és a teszt **ellentmondott egymásnak**: a legmagasabb belső pontszámú, sekély modellek (RF 9, XGB 9 és 8, `max_depth` 2–6) a teszten gyengék voltak.
- Ezért **a split kiegyensúlyozatlan és túl kicsi**. Megvizsgáltuk a LOTO-t (leave-one-track-out) is, időbecsléssel: teljes nested LOTO kb. 25–30 óra, LOTO mint GridSearch-CV kb. 6–7 óra, LOTO fix paraméterekkel kb. 5 perc. **Időhiány miatt elvetettük.** Helyette a kiegyensúlyozott, 15–25%-os splitet választottuk (a „C” jelöltet, ma `balanced5track`). Más jelöltek: „A” (1650 minta, 334 nem kúp) és „B” (1638 minta, 323 nem kúp). Mindkettő a tesztbe tette mindhárom kisebb Detection tracket, így a train nem kúpjainak kb. 92%-a Skidpad lett volna.

## C. Futások a balanced5track spliten, nyers adaton, tengelyhez kötött width/depth (azóta törölve)

| Modell | Sima F1 | Macro F1 | Nem kúp recall |
|---|---|---|---|
| RF 13 | 0.9541 | 0.867 | 67.9% |
| RF 11 | 0.9555 | 0.873 | 70.1% |
| RF 9 | 0.9579 | 0.879 | 70.5% |
| XGB 13 | 0.9512 | 0.866 | 72.1% |
| XGB 11 | 0.9506 | 0.862 | 69.8% |
| XGB 9 | 0.9513 | 0.866 | 71.4% |
| XGB 8 (volume nélkül) | 0.9508 | 0.865 | 71.8% |
| XGB 7 (volume és depth nélkül) | 0.9456 | 0.848 | 67.5% |

Itt derült ki az **intensity szivárgás** (1.2). A modellek kb. 40%-ban az intensityre támaszkodtak.

## D. Frame-normált adaton, tengelyhez kötött width/depth (azóta törölve)

| Modell | Sima F1 | Macro F1 | Intensity importance |
|---|---|---|---|
| XGB 13 | 0.9637 | 0.897 | 8.0% |
| RF 13 | 0.9600 | 0.883 | 9.8% |
| XGB 8 | 0.9587 | 0.881 | 5.5% |
| RF 9 | 0.9578 | 0.875 | 7.0% |
| XGB 9 (4 feature egyszerre kivágva) | 0.9568 | 0.875 | 4.5% |

- **Permutációs teszt** (mindkét intensity oszlop keverve a teszten): a macro F1 esés RF 13-nál 0,028, XGB 13-nál 0,047. **Ugyanekkora** esés jön ki forráson belüli és trackenkénti keveréssel is. A modellek tehát a valódi, klaszterenkénti intensity-jelet használják.
- **A felhasználó döntési szabálya:** ha a nagy modellek sokat használják az intensityt, kivágjuk, és marad a tiszta geometria; ha keveset, maradhat. **Eredmény:** kicsi és valódi, ezért maradt.
- Négy feature egyszerre kivágása (compactness, planarity, linearity, num_points) rontott (XGB −0,022 macro). Az alacsony importance tehát nem jelenti, hogy egy feature nélkülözhető.
- **Permutációs importance** (RF 9 és XGB 9, ezen az adaton): az **aspect_ratio** adja a pozitív F1-esések 58%-át, illetve 47%-át. A width az RF-ben 14,7% beépített importance-t kapott, de a permutációs hatása 0. Az XGB density 49,6%-os gainje egyetlen gyökérvágásból jön (1073-as gain az első fában).

## E. A width/depth definíció története (git)

| Időszak | Commit | width / depth |
|---|---|---|
| 2025-12 | `8048e822` | x/y kiterjedés |
| 2026-01-17 – 07-28 | `d6992c35` („adaptive width and depth”, Mező-Kerekes Márk) | `max/min(σx, σy)` |
| 2026-07-29 22:32 | `cff9be2b` (7 feature, 2371 minta, F1 0,924) | `max/min(σx, σy)` |
| 2026-07-29 22:42 | `3e1c34b1` (12 feature) | visszaállt: x/y kiterjedés |
| 2026-08-31 | `f836b974` | a 12 feature-os, kiterjedéses változat volt a fő modell |
| 2026-09-25 óta | `features.py` (`extract_features`) | `max/min(σx, σy)` |

- Az augusztusi 12 feature-os kód nem tartalmazta a distancefromlidar-t.
- A `test/models/cone_detector_7features.bin` 2025-12-30-án készült, amikor a committolt `train.py` még kiterjedéses volt. A teszt script viszont januártól dinamikus width-del értékeli ki, így lehetséges az eltérés.
- A felhasználó által mutatott régi importance-ábra (width 0,335, aspect 0,175 stb.) a `d6992c35` és `cff9be2b` közötti 7 feature-os korszakból való.
- **Ellenőrzés:** a kúpoknál a tengelyhez kötött x és y kiterjedés minden irányszögben kb. 0,10–0,12 m, és az aspect_ratio 1,8–2,4, tehát a kúpoknál a tengelyfüggés kicsi. Elnyúlt tárgyaknál (fal, kerítés) számít: egy y irányú fal kis x-kiterjedést és ezért kúpszerű aspect_ratio-t kapott.
- **A density kiugró értékei:** a medián kb. 3400, a maximum 1,2 millió, mert 32 klaszter szinte lapos (térfogat < 1e-4 m³).

## F. Döntések, szabályok

- **Feature-kivágás:**
  - eredetileg: körönként 2 feature 1% alatt; ha nincs elég, pótlás az 1–3%-os sávból; 3% fölött kérdezni kell;
  - később: automatikusan, 1,5%-os határral; a felhasználó egy körben 4 feature egyszerre kivágását is kérte.
- **Az RF és az XGB külön úton halad** („más set-ekben mások lesznek fontosak”).
- **Metrikák:** mindig a sima és a macro F1 is szerepel. A rangsor a sima F1 szerint, de a macro F1 is ott van mellette.
- **Minden tanítás után feature importance ábra** készül. PointNetnél csatorna-importance.
- **A split neve:** „C split”, ma `balanced5track`.
- **A tengelyhez kötött futások** kimenetét a felhasználó kérésére töröltük: `rf/xgb_feature_selection`, `*_splitC`, `*_framenorm_splitC`.
- **PointNet:**
  - a cikk szerinti szerkezet a viszonyítási alap, és csak a tanítási paramétereket hangoljuk, a kisebb keresés szerint;
  - Márk változatának neve: `"modified"`.
- **Commitok:** a felhasználó maga commitol, én csak a stage-be teszek. A modellekből csak a reprezentatív PointNetek kerültek a gitbe.

## G. Technikai megjegyzések

- **RF szálkezelés:** a GridSearch és a modell is `n_jobs=-1` volt, ami szálversenyt okozott. A javítás után a modell `n_jobs=1`, és kb. 2,7-szer gyorsabb (32 perc → 12 perc). Az eredmény nem változik.
- **Az XGB gyorsabb** (kb. 12 ms/fit az RF kb. 130 ms-ával szemben). Ennek oka a sekély fa (2–4), a hisztogram alapú vágáskeresés (`tree_method='hist'`) és a bootstrap hiánya. A `hist` nem csalás: csak a train foldból készít legfeljebb 256 küszöbjelöltet. Mérve: hist kb. 330 ms, F1 0,9542; exact kb. 530 ms, F1 0,9523.
- **A torch-ot a HDD-re telepítettük**, lásd 4.1. Az ONNX exporthoz hiányzik az `onnxscript`, ezért kimarad, de a `.pth` modell mentődik.
- **2026-10-18-án áramszünet volt:** a `/tmp` alatti scriptek elvesztek, és az RF 9 (adaptív) megszakadt. Utána folytattam. A scriptek azóta a `scripts/detection/` alatt vannak.
- **Egy hibás várakozás:** a `pgrep -f train_detection.py` a várakozó shell saját parancssorát is megtalálja, ezért végtelen ciklusba kerül. A PID-re kell várni.
- **A GPU-s PointNet tanítás nem bitre determinisztikus** (cuDNN). Ugyanaz a seed más epochnál állhat meg. Mindig több seed átlagát kell nézni.

## H. Magyarázatok (rövid fogalomtár)

- **Macro F1:** az F1 osztályonként (kúp és nem kúp) külön, utána egyszerű átlag. A sima F1 csak a kúp osztály F1-e, és 90%-os kúparány mellett a nem kúp gyengeséget elrejti.
- **Gini impurity:** `1 − Σ p_k²`. Az RF fái ez alapján vágnak, a legnagyobb súlyozott csökkenés szerint. Az F1 macro csak a hiperparaméterek kiválasztását irányítja. Alternatívák:
  - sklearn: entropy / log_loss;
  - más fáknál: gain ratio, twoing, chi-négyzet, Hellinger-távolság (osztályarány-független);
  - az XGB a veszteségfüggvény gradienseiből számolt gaint használja (log loss, `reg_lambda`, `gamma`); lehetne focal loss is.
- **A beépített importance:**
  - RF: az összes vágás Gini-csökkenése, 100%-ra normálva;
  - XGB: a vágások átlagos gainje.

  A kettő nem összemérhető, és mindkettő torzíthat. A permutációs importance a valós hatást méri.
- **PointNet:**
  - minden pontot ugyanaz a kis háló dolgoz fel, majd a pontok vektorait max-pooling vonja össze, így az eredmény független a pontok sorrendjétől;
  - a T-Net-ek tanult igazító transzformációk;
  - kb. 3,5 M paraméter;
  - a max-pooling nem érzékeny az ismétlésre, ezért **a klaszter pontszáma elvész**: egy 9 pontos klaszter, 32-re töltve, ugyanaz, mint 9 pontként.
