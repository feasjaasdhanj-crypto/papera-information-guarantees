# Code and Software Submission Checklist: submission materials

The accompanying reviewer demonstration, source code and README provide the following items required for submission. The completed research analyses are documented in the manuscript and Supporting Information; the runnable compact demonstration offers a convenient initial check.

| Checklist item | Provided material |
|---|---|
| Source code | 43 original scripts in `Scripts/`, with SHA-256 provenance manifest |
| Small demonstration dataset | `Demo/INPUT/certificates_16.json` and `Demo/INPUT/example_matrices.json` |
| Software dependencies and operating system versions | `README.md` §1 and `requirements-full-reviewer.txt` |
| Tested software version | `Demo/TESTED_ENVIRONMENT.json`: Python 3.13.5 on Linux x86-64 |
| Non-standard hardware | `README.md` §1: no special hardware for the executable demo |
| Installation instructions | `README.md` §2; Windows launcher included |
| Typical installation time | `README.md` §2: under one minute to extract/prepare with Python preinstalled on reference machine |
| Instructions to run demo on data | `README.md` §3; `Demo/run_all_demos.py` |
| Expected output | `Demo/EXPECTED/`, `Demo/OUTPUT/` |
| Expected runtime | `README.md` §3; median 3.2904 seconds, five tests |
| How to run on user data | `README.md` §4; `Demo/run_custom_matrices.py` |
| Optional reproduction instructions | `FULL_STUDY_REPRODUCTION.md` |

**Reviewer access arrangement:** The full set of 43 preserved analysis scripts is already present in this ZIP. The executable demo reproduces the selected mathematical construction and its central quantitative results. Original source datasets are cited in the manuscript. For a specific additional empirical replay, editors and reviewers may contact the corresponding author to obtain available input mappings, intermediate result files or tailored run instructions under the respective dataset-use conditions.

The example pinned dependencies are provided for reviewer convenience; the original historical versions were not completely archived. The original research code now carries an MIT License in LICENSE. The public repository identifier has not yet been created and must be added after an actual deposit.
