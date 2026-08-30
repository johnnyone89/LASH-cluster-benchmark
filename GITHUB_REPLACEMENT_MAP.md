# GitHub replacement map — LASH second-round aligned repository

This patch was built from the user-supplied `LASH-cluster-benchmark-main.zip`, not from the stale public GitHub crawler snapshot.

## Why these changes are necessary

The uploaded repository is already well organized into `data/`, `notebooks/`, and `src/`, so the modular structure is preserved.

However, the uploaded first-revision repository still contains three protocol mismatches with the code that generated the reported first- and second-round experiments:

1. `README.md` and Notebooks 00/01 describe 3 HPO seeds and 30–60 trials per repeat, while the executed manuscript protocol used **two TPE seeds (42, 142) and 3–5 finite COMPLETE trials per repeat**, followed by final refits with seeds 42/142/242.
2. Notebook 02 calls the generic retuned-ablation workflow, while the reported matched mechanism analysis **freezes each dataset's selected full-model setting** and treats the sequential-only prediction as the primary estimand.
3. The executed 2026-08-23/29 workflows used three additional source modules (`lash_deadline48.py`, `lash_hardware_optimized.py`, `lash_per_dataset_hpo.py`) that were not present in the uploaded repository. These files are copied byte-for-byte from the actual executed source bundle.

The patch does not add generated results. Users regenerate results locally under `outputs/`.

---

## Replace these files

Copy the following files from this package over the same paths in GitHub:

- `.gitignore`
- `README.md`
- `RUN_ORDER.txt`
- `requirements.txt`
- `notebooks/00_Protocol_and_Data_Audit.ipynb`
- `notebooks/01_LASH_and_Fair_Benchmark.ipynb`
- `notebooks/02_Matched_Ablation_Router_YoY.ipynb`
- `notebooks/03_Posthoc_Statistics_XAI_Visualization.ipynb`

### What each replacement fixes

- `README.md`
  - keeps the existing modular repository organization;
  - corrects the paper HPO protocol to 2 TPE seeds + 3–5 COMPLETE trials;
  - explains that outputs are regenerated locally rather than committed;
  - adds the second-round confirmatory notebook to the run order.

- `00_Protocol_and_Data_Audit.ipynb`
  - replaces the generic 30–60-trial budget audit with the exact dataset-local 3–5-COMPLETE-trial protocol manifest.

- `01_LASH_and_Fair_Benchmark.ipynb`
  - switches the public runner to the exact `reviewer_20260823_per_dataset_hpo_v1` workflow used for the manuscript;
  - writes a frozen run contract before fitting;
  - keeps the four-dataset, historical-only, direct 24-step benchmark.

- `02_Matched_Ablation_Router_YoY.ipynb`
  - uses the frozen full-model setting for the reported matched mechanism tests;
  - keeps YoY analysis restricted to the two multiyear cluster datasets;
  - adds the same structural/feature evaluation for BDG without introducing invalid YoY tests.

- `03_Posthoc_Statistics_XAI_Visualization.ipynb`
  - uses the first-revision priority inferential workflow;
  - documents the 1,000-replicate first-revision analysis separately from the second-round 5,000-replicate extension;
  - applies the same XAI diagnostic contract to all four datasets, with the correct Ridge-only interpretation for BDG.

- `requirements.txt`
  - adds dependencies already imported by the repository code (SciPy, statsmodels, OpenPyXL, boosting libraries, SHAP, hardware-audit packages);
  - PyTorch remains a separate installation so users can select the correct CPU/CUDA build.

- `.gitignore`
  - actually excludes generated `outputs/`, Optuna databases, caches, and notebook checkpoints.

---

## Add these files

Add the following new files without deleting the existing three `lash_revision_*` modules:

- `src/lash_deadline48.py`
- `src/lash_hardware_optimized.py`
- `src/lash_per_dataset_hpo.py`
- `notebooks/04_SecondRound_Confirmatory.ipynb`

The three source files are the exact modules used by the reported 2026-08-23/29 runs.

`04_SecondRound_Confirmatory.ipynb` is a repository-friendly version of the executed second-round notebook. It:

- reads frozen settings from `outputs/` created by Notebook 01;
- uses the same ten confirmatory seeds;
- uses 5,000 hierarchical bootstrap replicates;
- freezes the 1% revision-stage relative-NMAE interpretation threshold before new refits;
- performs validation-only reduced-architecture selection and writes the architecture lock before test evaluation;
- performs retained-component drop-one tests;
- audits the BDG claim ceiling;
- runs the common constrained storage-dispatch sensitivity;
- writes all new files under `outputs/second_round_confirmatory/`;
- does not embed or publish generated result files.

---

## Leave these files unchanged

Do **not** replace or rewrite:

- `data/Cluster_1_Harmonized.csv`
- `data/Cluster_2_Harmonized.csv`
- `data/BDG_Edu_Harmonized.csv`
- `data/BDG_Dorm_Harmonized.csv`
- `data/README_DATA.md`
- `src/lash_revision_core.py`
- `src/lash_revision_analysis.py`
- `src/lash_revision_ablation.py`
- `outputs/.gitkeep`

The three existing `lash_revision_*` Python modules are already byte-for-byte identical to the source bundle embedded in the actual second-round run.

---

## Exact source hashes used by the second-round run

- `lash_revision_core.py`  
  `ce918efe1b1598f0d2a810f4400979ba51ce94b11a6dbac0aae00e85039c088b`

- `lash_revision_analysis.py`  
  `28591f97ffe7417546afcf6664c020c151c398747663d6b7659e119247a85633`

- `lash_revision_ablation.py`  
  `0908dac343661e19c97df0f05bd9ff0fd2cef1a297ced37ccaec82f0a45097b9`

- `lash_deadline48.py`  
  `8125a385ff40d60b49faa4fa83b4b6b5d7f33d89afd52fd65e264cfbd399b166`

- `lash_hardware_optimized.py`  
  `71f60dd830a9ab6079653d9e5e44b8e7ad3edb4d1c0fbbf642de121091112f1a`

- `lash_per_dataset_hpo.py`  
  `e116eb1fdc439cc4ea821e8aa06658205453892a3586329c1ef78f02647d1652`

---

## Recommended GitHub update sequence

1. Replace the eight files listed under **Replace these files**.
2. Add the four files listed under **Add these files**.
3. Do not upload any local `outputs/` result directory.
4. Run Notebook 00 once as a quick audit.
5. For a full reproduction, follow `RUN_ORDER.txt`.
6. Commit with a concise message such as:
   `Align public reproduction code with revised dataset-local HPO and second-round confirmation`

