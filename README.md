# Information guarantees for scientific conclusions from measurements
## Research code and reviewer demonstration

This archive provides the 43 original analysis and figure-source scripts used during this study, together with a compact, directly executable demonstration of the paper's central mathematical construction. The original source scripts are unchanged. Their original archive paths and SHA-256 identifiers are recorded in `SOURCE_MANIFEST.json`.

**Start here:** install Python 3, extract the archive, and run `python Demo/run_all_demos.py` (or double-click `Run_Demo_Windows.bat` on Windows). The demo verifies 16 exact primal–dual certificates, the local-versus-complete ordering reversal, exact 0–1 classification-risk differences, and finite-sample certification thresholds. The expected outputs are included for comparison.

The full study's results, including the analyses of the seven publicly available sensing datasets, were obtained during the research and are reported in the manuscript and Supporting Information. All 43 analysis and figure scripts are supplied here for review. The compact demonstration offers a quick entry point; the original data sources are cited in the manuscript, and additional available intermediate outputs, source-specific instructions, or detailed reproduction materials can be requested from the corresponding author for a particular analysis, subject to original dataset terms.

## 1. System requirements and tested software versions

| Component | Requirements and tested environment |
|---|---|
| **Runnable demonstration** | Python 3 (standard library only); tested on CPython 3.13.5 under Linux 6.18.57 x86-64, glibc 2.41 |
| **Full Python analysis sources** | Python with NumPy, pandas, SciPy, scikit-learn, statsmodels, Matplotlib and PyMuPDF; example version-pinned reviewer environment in `requirements-full-reviewer.txt` |
| **R analysis source** | R, with Rscript >= 4.2 as suggested in the archived R script header |
| **Non-standard hardware** | None for the included mathematical demo. A normal CPU-only desktop can run it; no sensor, EEG headset or GPU is needed |
| **Full empirical inputs** | Public datasets identified by provider and DOI in the paper's Data Availability statement, plus the relevant archived derived feature tables when a specific workflow requires them |

The demo's system and software-version record is in `Demo/TESTED_ENVIRONMENT.json`. The pinned Python package versions in `requirements-full-reviewer.txt` describe a prepared reviewer environment; the original historical software-version inventory was not retained. The empirical scripts are supplied in their original form to preserve their numerical procedures.

## 2. Installation

**Windows:**

1. Download and extract `Code_Scripts.zip` into a writable folder, keeping the included directory structure.
2. Install Python 3 if needed and enable `py` or `python` on PATH. Python 3.13 is a tested-version choice.
3. Double-click `Run_Demo_Windows.bat`, or run `py -3 Demo/run_all_demos.py` from the extracted root.
4. Open `Demo/OUTPUT/test_report.json` to inspect the results.

**Linux/macOS:**

```bash
python3 Demo/run_all_demos.py
```

For the **compact demonstration**, no third-party packages, network access, authentication credentials or database downloads are required. With Python available, unpacking and preparing this small archive took under one minute on the reference machine; installing Python itself varies by machine.

For empirical and figure scripts, a convenient example Python environment can be prepared with:

```bash
python3 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-full-reviewer.txt
```

Specific research scripts use the input files and paths identified in `FULL_STUDY_REPRODUCTION.md`.

## 3. Small demonstration dataset and results

`Demo/INPUT/certificates_16.json` contains the 16 exact rational LP primal/dual certificate records used in the study. `Demo/INPUT/example_matrices.json` contains the three-state and three-observation probability matrices from the mathematical construction. These compact JSON inputs are supplied with the corresponding verification scripts.

**One-command demonstration:**

```bash
python Demo/run_all_demos.py
```

Expected numerical outputs include:

- `exact_primal_dual_certificates_passed = 16`, with `all_pass = true`;
- `A_full = 3/86`, `A_local = 1/430`, and `B_full = B_local = 1/100`;
- risk changes `3/86` and `-1/150`, with regret `134/3225` (about 4.155 percentage points);
- sufficient per-state-row observation counts of **33,700** and **987,818** (and **327,258** for the intermediate criterion);
- `Panel alignment self-test: PASS`.

The program writes its checked outputs into `Demo/OUTPUT/`. Reference outputs are in `Demo/EXPECTED/`; mismatches cause a non-zero exit status. Five complete demo runs on the specified Linux/Python environment took 3.0842–3.3585 seconds, with a median of **3.2904 seconds**. These are measured demonstration times.

## 4. Running the demonstration on your own data

Prepare a JSON file modelled on `Demo/INPUT/example_matrices.json`. For each experiment, provide three conditional probability rows for each matrix `P` and `Q`, each containing three non-negative observation probabilities that sum to one. Specify a three-element prior distribution. Inputs may be rational strings such as `"1/3"` or decimal strings.

```bash
python Demo/run_custom_matrices.py --input path/to/your_matrices.json --output path/to/your_risks.json
```

The program computes the optimal 0–1 Bayes classification risks of `P` and `Q` and their difference. When experiments `A` and `B` are supplied, it also compares their risk changes. It validates dimensions and probability row totals. To run the same comparison alongside the published mathematical checks, use:

```bash
python Demo/run_all_demos.py --custom-input path/to/your_matrices.json --output Demo/MY_OUTPUT
```

Other decision functions and empirical activity, chemical sensor or EEG measurements are handled by the corresponding research scripts in `Scripts/`; see the workflow map in `FULL_STUDY_REPRODUCTION.md`.

## 5. Full-study analysis and reviewer requests

The study includes previously completed empirical evaluations of five activity datasets, one chemical sensing dataset and an independent EEG dataset. Their principal outputs are reported in the manuscript, Supporting Information and associated Source Data. The preserved analysis sources in `Scripts/` include preprocessing, information comparisons, sensor selection, participant-level inference, temporal evaluation and figure generation. The data provider identifiers and access links appear in the paper's Data Availability statement.

For the mathematical construction and finite-sample certificates, this archive supplies all demonstration inputs directly. For review of a particular full-data workflow, readers may request the corresponding available derived-input mappings, intermediate results, detailed run instructions or clarification from the corresponding author. Third-party datasets remain accessible through their original providers and subject to their terms. The archive itself already contains the full set of 43 preserved research scripts.

## 6. Licence and public code archive

The study authors' original analysis, figure-generation and demonstration source code is provided under the **MIT License**; see `LICENSE` in this archive. Third-party software packages and the seven source datasets retain their own terms. This licence is for the original software and does not relicense dataset contents or third-party dependencies.

The exact manuscript-associated code is supplied in this ZIP for editorial and peer review. The DOI-minting public archival record is prepared for deposit but has **not yet been created**; once a permanent code URL/DOI exists, the manuscript Code availability statement and the software checklist must be updated with the real identifier. `CITATION.cff` records the software authors, version and licence; it deliberately does not invent a DOI.

## 7. File map

| Location | Contents |
|---|---|
| `Scripts/` | 43 original analysis and figure-source files, unchanged |
| `SOURCE_MANIFEST.json` | Original source paths, sizes and SHA-256 identifiers |
| `Demo/INPUT/` | 16 exact certificates and three-state example matrices |
| `Demo/EXPECTED/` | Expected numerical results |
| `Demo/run_all_demos.py` | Single-command reviewer demonstration |
| `Demo/run_custom_matrices.py` | Example interface for user-supplied probability matrices |
| `Demo/TESTED_ENVIRONMENT.json` | Tested environment and measured run times |
| `requirements-full-reviewer.txt` | Example pinned dependencies for broader Python analysis |
| `FULL_STUDY_REPRODUCTION.md` | Map from paper analyses to scripts and data inputs |
| `Run_Demo_Windows.bat` | Windows demonstration launcher |

**Suggested reviewer starting point:** `python Demo/run_all_demos.py`.
