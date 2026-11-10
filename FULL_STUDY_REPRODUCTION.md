# Reproducing the reported analyses

## 1. What the archive provides

The paper's mathematical theory and exact numerical construction, five human-activity datasets, one chemical sensor dataset and a separate 106-participant EEG analysis constitute the research results reported in the manuscript and Supporting Information. These analyses were completed in the course of the study. This submission provides all 43 archived analysis and figure-source scripts in `Scripts/` and a directly runnable mathematical demonstration with the small input datasets and expected output in `Demo/`.

The compact reviewer demonstration verifies the exact three-outcome example, 16 primal–dual certificate records, 0–1 classification-risk comparisons and finite-sample thresholds. Its purpose is to give reviewers a short, executable route through a central theoretical and decision result. The study's full empirical results and figure source data are reported and archived separately in the manuscript, Supporting Information and Source Data.

## 2. Original study analysis entry points

| Workflow | Relevant code | Inputs and outputs |
|---|---|---|
| Exact theory and decision example | `Scripts/35_R528_VERIFY_EXACT_PURE_PYTHON.py`; `Scripts/20_R535_FINITE_SAMPLE_CERTIFICATE_PYTHON_STDLIB.py` | `Demo/INPUT/certificates_16.json`, `Demo/INPUT/example_matrices.json`; checked outputs in `Demo/OUTPUT/` |
| Six-source retrospective analysis | `Scripts/01_R546_REPLAY_SIX_SOURCE_ONE_COMMAND.py` | Original entry-point syntax: `python <script> <PaperA_R544_COMPLETE.zip> <output_directory>`; the author archive stores a nested historical input hierarchy with UCI HAR, GasDrift and derived features for other sources |
| EEG participant-level replay | `Scripts/37_R544_FULL_R_REPLAY_CORRECTED.R` | `Rscript Scripts/37_R544_FULL_R_REPLAY_CORRECTED.R /path/to/working_root`; reads `R544_REPRO_INPUT/` and creates `R544_REPRO_OUTPUT/`. Source inputs include `EEG_106_SUBJECTS_ALL_EXTRACTED_FEATURES.csv.gz`, subject maps, portfolio/information tables and risk-comparison tables |
| Figure production | `Scripts/38_prepare_source_data.py`–`Scripts/42_make_figures.py` | Source Data tables and the associated panel-to-result mapping, with NumPy/pandas/Matplotlib/PyMuPDF as required |
| Other analysis and validation steps | The remaining scripts in `Scripts/` | Source-specific raw or derived input files, evaluation partitions and stored outputs as described in their respective code |

The source files retain the original path and computation details, including historical nested-directory references. Keep the expected input arrangement when reproducing a specific workflow. `SOURCE_MANIFEST.json` identifies each source by original archive path and SHA-256.

## 3. Public source datasets

The original seven sources are WISDM, Daily and Sports Activities, MEx, MHEALTH, UCI HAR, Gas Sensor Array Drift at Different Concentrations, and PhysioNet EEG Motor Movement/Imagery. Use the provider URLs and DOIs in the paper's Data Availability statement. Original sensor measurements and the derived feature tables needed for individual scripts are distinct inputs.

The compact ZIP carries the mathematics demonstration rather than duplicating the complete public datasets. Editors or reviewers seeking a deeper replay of an individual empirical comparison can request the corresponding available feature-level inputs, analysis-output maps, preserved intermediate results and run guidance from the corresponding author. Sharing of third-party data follows its original licence and repository conditions.

## 4. Software and timing

The mathematical demonstration uses the Python 3 standard library and was tested with Python 3.13.5 on Linux x86-64. Five demo runs had a median wall time of 3.2904 seconds. For additional Python packages and compatible imported versions, see `requirements-full-reviewer.txt`. The original R source suggests R >= 4.2. The installed reviewer-example software versions are identified in `Demo/TESTED_ENVIRONMENT.json`; historical versions from the original research environment were not completely archived.

## 5. Quick reproduction

```bash
python Demo/run_all_demos.py
```

All published mathematical check values for this example are listed in `README.md` and stored in `Demo/EXPECTED/`. For a separate three-state probability experiment:

```bash
python Demo/run_custom_matrices.py --input path/to/your_matrices.json --output path/to/your_risks.json
```

If additional full-data verification is requested during peer review, the corresponding author can identify the matching research script and provide available detailed reproduction instructions and intermediate materials for that analysis.
