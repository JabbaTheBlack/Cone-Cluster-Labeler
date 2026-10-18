# PointNet futások összefoglalója (nem végleges futások)

Ez a fájl az összes PointNet futás eredményét rögzíti, a kiválasztott modell kivételével. A logjaikat és ábráikat nem commitoljuk: a modellek a `scripts/detection/pointnet_search.py`, a `src/train/detection/train_detection.py` és a lenti parancsok alapján újrataníthatók. Adat: `Dataset/Processed_FrameNorm`, split: `balanced5track` (teszt: 1246 kúp, 308 nem kúp). Validáció: AutoX_7 + Ekol_09_04_16 (437 minta, 74 nem kúp). FP = nem kúp kúpnak jósolva, FN = kihagyott kúp. A háttér és a magyarázatok: `docs/detection/experiments_2026-10.md`.

**Kiválasztott modell:** `pointnet_search/final_modified_seed0` (módosított szerkezet, intensityvel). Azért ezt választottuk, mert az 5 seed közül ennek a legjobb a **validációs sima F1**-e: 0,9918 0,5-ös küszöbbel, 0,9932 a hangolt 0,53-as küszöbbel. A validációs macro F1 is ezt rangsorolja elsőnek (0,9791). Teszten, 0,53-as küszöbbel: F1 0,9668, macro F1 0,906. A logja, ábrái és modellje commitolva vannak.

## 1. Első futások (módosított szerkezet, seed 42, régi mintavétel)

| Futás | Tanító minták | Epochok (legjobb) | Sima F1 @0,5 | Macro F1 @0,5 | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|
| `pointnet_balanced5track` (validációval) | 4664 | 21 (6) | 0.9566 | 0.8668 | 64.0% | 111 | 2 |
| `pointnet_balanced5track_full` (teljes train, `--no-validation --epochs 6`) | 5101 | 6 (6) | 0.9550 | 0.8621 | 63.3% | 113 | 4 |

Küszöbhangolás (`pointnet_balanced5track`): a validáción a legjobb küszöb 0.55 (val macro F1: 0.9478 0,5-nél, 0.9675 a legjobb küszöbnél). A teszten ezzel F1 0.9627, macro F1 0.8911, FP 90, FN 6; 0,5-nél F1 0.9566, macro F1 0.8668, FP 111, FN 2.

## 2. Intensity ablation (módosított szerkezet, régi mintavétel)

Parancs: `train_detection.py --dataset Dataset/Processed_FrameNorm --model pointnet --test-tracks balanced5track --seed S [--no-intensity] --output-name pointnet_ablation/{xyzi|xyz}_seedS`

### x, y, z, intensity

| Seed | Epochok (legjobb) | F1 @0,5 | Macro @0,5 | Küszöb | F1 @küszöb | Macro @küszöb | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 48 (33) | 0.9645 | 0.8991 | 0.59 | 0.9675 | 0.9092 | 76.9% | 71 | 12 |
| seed 1 | 18 (3) | 0.9604 | 0.8898 | 0.46 | 0.9573 | 0.8749 | 68.2% | 98 | 12 |
| seed 2 | 39 (24) | 0.9627 | 0.8915 | 0.54 | 0.9668 | 0.9057 | 75.0% | 77 | 8 |
| seed 3 | 42 (27) | 0.9615 | 0.8889 | 0.50 | 0.9615 | 0.8889 | 71.4% | 88 | 11 |
| seed 4 | 24 (9) | 0.9551 | 0.8612 | 0.65 | 0.9648 | 0.9053 | 79.2% | 64 | 25 |

### x, y, z

| Seed | Epochok (legjobb) | F1 @0,5 | Macro @0,5 | Küszöb | F1 @küszöb | Macro @küszöb | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 44 (29) | 0.9553 | 0.8648 | 0.50 | 0.9553 | 0.8648 | 64.6% | 109 | 7 |
| seed 1 | 33 (18) | 0.9536 | 0.8560 | 0.57 | 0.9621 | 0.8927 | 73.4% | 82 | 15 |
| seed 2 | 19 (4) | 0.9596 | 0.8821 | 0.49 | 0.9582 | 0.8757 | 67.2% | 101 | 7 |
| seed 3 | 32 (17) | 0.9546 | 0.8616 | 0.50 | 0.9546 | 0.8616 | 63.6% | 112 | 6 |
| seed 4 | 32 (17) | 0.9586 | 0.8766 | 0.50 | 0.9586 | 0.8766 | 67.2% | 101 | 6 |

Átlag (hangolt küszöb): intensityvel F1 0,9636 ± 0,0038, macro F1 0,897 ± 0,013; intensity nélkül F1 0,9578 ± 0,0027, macro F1 0,874 ± 0,011.

## 3. Hiperparaméter-keresés (cikk szerinti szerkezet, 20 kombináció × 3 seed)

Parancs: `python scripts/detection/pointnet_search.py --configs 20 --seeds 3 --final-seeds 5`. A pontszám a validációs macro F1, hangolt küszöbbel. Fix: `arch=paper`, intensityvel, 0,001-es regularizáció. A 0. kombináció a cikk alapbeállítása.

| # | Val macro F1 (átlag ± szórás) | Seedenként | learning rate | lr_step | dropout | weight decay | pontok | türelem | loss | legjobb epochok |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (cikk) | 0.9765 ± 0.0020 | 0.975, 0.979, 0.975 | 0.001 | 20 | 0.3 | 0.0 | 32 | 15 | ce | 3, 24, 32 |
| 2 | 0.9750 ± 0.0091 | 0.983, 0.962, 0.979 | 0.001 | 20 | 0.5 | 0.001 | 16 | 15 | focal | 37, 20, 15 |
| 17 | 0.9738 ± 0.0078 | 0.963, 0.979, 0.979 | 0.0003 | 20 | 0.5 | 0.0 | 16 | 15 | ce | 14, 17, 30 |
| 3 | 0.9738 ± 0.0020 | 0.975, 0.971, 0.975 | 0.0003 | 0 | 0.5 | 0.001 | 64 | 15 | focal | 57, 36, 40 |
| 15 | 0.9737 ± 0.0019 | 0.975, 0.971, 0.975 | 0.001 | 0 | 0.5 | 0.001 | 64 | 30 | ce | 50, 17, 78 |
| 9 | 0.9724 ± 0.0037 | 0.967, 0.975, 0.975 | 0.0003 | 20 | 0.5 | 0.001 | 16 | 30 | ce | 5, 52, 51 |
| 19 | 0.9710 ± 0.0034 | 0.967, 0.975, 0.971 | 0.001 | 0 | 0.5 | 0.0 | 16 | 15 | ce | 4, 31, 8 |
| 13 | 0.9698 ± 0.0038 | 0.967, 0.975, 0.967 | 0.0003 | 0 | 0.5 | 0.001 | 64 | 30 | ce | 23, 10, 86 |
| 1 | 0.9697 ± 0.0021 | 0.967, 0.972, 0.971 | 0.001 | 20 | 0.3 | 0.0001 | 64 | 30 | focal | 23, 24, 22 |
| 10 | 0.9681 ± 0.0055 | 0.967, 0.962, 0.975 | 0.003 | 20 | 0.5 | 0.0 | 64 | 30 | focal | 23, 31, 43 |
| 7 | 0.9681 ± 0.0019 | 0.971, 0.967, 0.967 | 0.001 | 0 | 0.5 | 0.001 | 16 | 15 | ce | 37, 24, 12 |
| 14 | 0.9679 ± 0.0042 | 0.962, 0.971, 0.971 | 0.0003 | 20 | 0.5 | 0.0001 | 32 | 30 | ce | 4, 52, 8 |
| 18 | 0.9662 ± 0.0038 | 0.962, 0.966, 0.971 | 0.0003 | 0 | 0.3 | 0.0 | 64 | 15 | ce | 9, 7, 5 |
| 4 | 0.9649 ± 0.0018 | 0.966, 0.966, 0.962 | 0.0003 | 0 | 0.5 | 0.0001 | 64 | 15 | focal | 22, 52, 12 |
| 5 | 0.9642 ± 0.0041 | 0.967, 0.967, 0.958 | 0.001 | 20 | 0.3 | 0.001 | 32 | 30 | focal | 78, 95, 44 |
| 12 | 0.9640 ± 0.0073 | 0.967, 0.954, 0.971 | 0.003 | 20 | 0.5 | 0.0 | 16 | 15 | ce | 4, 14, 32 |
| 16 | 0.9613 ± 0.0070 | 0.955, 0.971, 0.958 | 0.0003 | 0 | 0.5 | 0.0 | 64 | 15 | focal | 21, 36, 17 |
| 11 | 0.9612 ± 0.0053 | 0.967, 0.954, 0.963 | 0.003 | 0 | 0.5 | 0.0 | 16 | 15 | ce | 4, 14, 22 |
| 8 | 0.9609 ± 0.0071 | 0.955, 0.971, 0.957 | 0.0003 | 0 | 0.5 | 0.0 | 16 | 30 | focal | 23, 52, 47 |
| 6 | 0.9591 ± 0.0159 | 0.957, 0.940, 0.979 | 0.0003 | 0 | 0.3 | 0.001 | 32 | 15 | focal | 64, 6, 72 |

Egyik kombináció sem volt jobb a cikk alapbeállításánál. A script a „hangolt” változatnak az alapbeállítástól eltérő legjobbat választotta, ez a 2. kombináció.

## 4. Végső kör (5 seed, új mintavétel)

Parancs: ugyanaz a script (`pointnet_search.py`). Kimenet: `pointnet_search/final_<név>_seed<k>`.

### Módosított (Márk) — a kiválasztott modell családja

| Seed | Epochok (legjobb) | F1 @0,5 | Macro @0,5 | Küszöb | F1 @küszöb | Macro @küszöb | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 34 (19) | 0.9635 | 0.8937 | 0.53 | 0.9668 | 0.9057 | 75.0% | 77 | 8 |
| seed 1 | 29 (14) | 0.9639 | 0.8950 | 0.57 | 0.9667 | 0.9070 | 76.6% | 72 | 13 |
| seed 2 | 33 (18) | 0.9646 | 0.8976 | 0.65 | 0.9699 | 0.9195 | 81.8% | 56 | 20 |
| seed 3 | 31 (16) | 0.9622 | 0.8921 | 0.53 | 0.9626 | 0.8973 | 76.3% | 73 | 22 |
| seed 4 | 18 (3) | 0.9634 | 0.8944 | 0.58 | 0.9662 | 0.9063 | 76.9% | 71 | 15 |

Validációs sima F1 seedenként (@0,5 / @hangolt küszöb): seed 0 0,9918 / 0,9932 (**kiválasztva**), seed 1 0,9904 / 0,9904, seed 2 0,9890 / 0,9903, seed 3 0,9877 / 0,9904, seed 4 0,9904 / 0,9918. Validációs macro F1 (hangolt): 0,9791, 0,9711, 0,9720, 0,9714, 0,9753.

### Cikk szerinti, alapbeállítás

| Seed | Epochok (legjobb) | F1 @0,5 | Macro @0,5 | Küszöb | F1 @küszöb | Macro @küszöb | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 18 (3) | 0.9568 | 0.8699 | 0.43 | 0.9557 | 0.8653 | 64.3% | 110 | 5 |
| seed 1 | 18 (3) | 0.9518 | 0.8545 | 0.68 | 0.9536 | 0.8668 | 68.5% | 97 | 22 |
| seed 2 | 33 (18) | 0.9615 | 0.8882 | 0.59 | 0.9649 | 0.9000 | 74.0% | 80 | 10 |
| seed 3 | 42 (27) | 0.9632 | 0.9059 | 0.06 | 0.9550 | 0.8626 | 63.6% | 112 | 5 |
| seed 4 | 45 (30) | 0.9667 | 0.9070 | 0.22 | 0.9609 | 0.8846 | 69.2% | 95 | 6 |

### Cikk szerinti, „hangolt” (a keresés 2. kombinációja)

| Seed | Epochok (legjobb) | F1 @0,5 | Macro @0,5 | Küszöb | F1 @küszöb | Macro @küszöb | Nem kúp recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 52 (37) | 0.9609 | 0.8843 | 0.47 | 0.9583 | 0.8749 | 66.6% | 103 | 5 |
| seed 1 | 35 (20) | 0.9600 | 0.8834 | 0.49 | 0.9612 | 0.8863 | 69.8% | 93 | 7 |
| seed 2 | 30 (15) | 0.9546 | 0.8667 | 0.50 | 0.9546 | 0.8667 | 66.9% | 102 | 15 |
| seed 3 | 42 (27) | 0.9558 | 0.8700 | 0.51 | 0.9569 | 0.8743 | 68.5% | 97 | 14 |
| seed 4 | 75 (60) | 0.9598 | 0.8857 | 0.48 | 0.9591 | 0.8825 | 70.8% | 90 | 15 |

Átlagok (hangolt küszöb): módosított F1 0,9665 ± 0,0023, macro F1 0,907 ± 0,007; cikk szerinti alap 0,9580 ± 0,0042 / 0,876 ± 0,014; cikk szerinti „hangolt” 0,9580 ± 0,0022 / 0,877 ± 0,007.

## 5. Csatorna-importance

Macro F1 esés a teszten, ha a csatornát a klaszterek között összekeverjük (20 ismétlés átlaga).

| Modell | x | y | z | intensity | xy együtt |
|---|---|---|---|---|---|
| `pointnet_balanced5track` | 0.141 | 0.096 | 0.082 | 0.032 | 0.285 |
| `pointnet_balanced5track_full` | 0.123 | 0.074 | 0.086 | 0.038 | 0.265 |
| `pointnet_ablation/xyzi_seed0` | 0.164 | 0.087 | 0.115 | 0.033 | 0.265 |
| `pointnet_ablation/xyz_seed0` | 0.094 | 0.113 | 0.091 | – | 0.254 |
| `pointnet_search/final_paper_default_seed0` | 0.123 | 0.118 | 0.048 | 0.005 | 0.294 |

A kiválasztott modell (`final_modified_seed0`) csatorna-importance-e: x 0,116, y 0,100, z 0,154, intensity 0,067. Ez csak elemzés, a választásban nem szerepelt, és a fájlja (`permutation_importance.json`) nincs commitolva.

