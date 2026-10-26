# XAI case study: diabetes risk screening

Final project for *Explainable AI* (Máster en IA). End-to-end case study: a model that predicts
diabetes/prediabetes from health-survey data, explained with several XAI techniques, whose
explanations are **evaluated, criticised and then used** to improve the model and to
analyse fairness.

- **Task:** binary classification, `Diabetes_binary` (diagnosed diabetes or prediabetes, self-reported).
- **Data:** [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
  (UCI id 891, BRFSS survey): 253,680 respondents, 21 features (clinical, lifestyle, access to care, socio-demographic).
  Downloaded by `src/data.py`; nothing is committed.
- **Stakeholder:** a public-health screening programme that must decide whom to send for a confirmatory test,
  and the clinician who has to trust (or question) each flag.
- **Main model:** LightGBM (regularised, early stopping), compared with a logistic-regression baseline.

## Key results

| | |
|---|---|
| Model (test, duplicates removed) | LightGBM ROC-AUC 0.818, PR-AUC 0.442 (logistic regression 0.807 / 0.404); no overfitting (train 0.825), well calibrated |
| Operating point | threshold chosen on validation for recall ≥ 80 %: recall 0.80, precision 0.31, 39 % of people flagged. Implies a missed case is valued ≈ 7 confirmatory tests (notebook 10) |
| Global explanations | `GenHlth`, `Age`, `BMI`, `HighBP`, `HighChol` dominate; permutation, SHAP and logistic coefficients agree (Spearman 0.94-0.96); lifestyle variables barely matter |
| Local explanations | SHAP vs LIME agree little (top-5 Jaccard 0.47) although LIME is stable across seeds; contrafactuals: mostly "treat blood pressure / cholesterol" or lose BMI |
| Sanity checks | SHAP collapses with shuffled labels; MoRF deletion drops AUC far faster than random; SHAP stable across backgrounds (0.93) and seeds (0.98) |
| Actionable use | `CholCheck`, `HvyAlcoholConsump`, `AnyHealthcare` reflect *detection bias*. Removing them costs 0.003 AUC and changes behaviour for unscreened groups (recall 26 % → 71 % in people with no cholesterol test) at the price of more false alarms: a trade-off, not a free win |
| Fairness | large recall gap by age (42 % for 18-44 vs 86 % for 65+), moderate by income, small by sex; calibration good in all groups; group thresholds close the gaps at a cost |
| Plausibility | learnt directions agree with CDC risk factors for 5 of 6 features (smoking is reversed) |

## Repository structure

```
src/           data.py (download), preprocess.py (cleaning + split), evaluation.py, xai.py (grouped permutation, ALE), counterfactuals.py
notebooks/     01-10, executed, with outputs
figures/       plots used in the report
data/          downloaded data (git-ignored)
models/        trained models (git-ignored, created by notebooks 02 and 06)
run_all.sh     runs everything from scratch
```

| Notebook | Content |
|---|---|
| `01_eda` | data quality, target balance, correlations, first warning signs |
| `02_models` | splits, baselines, LightGBM, metrics, overfitting check, screening threshold |
| `03_global_xai` | permutation importance (individual and grouped), SHAP, PDP/ICE/ALE, method comparison |
| `04_local_xai` | SHAP waterfalls, LIME vs SHAP, counterfactuals |
| `05_sanity_checks` | label randomisation, MoRF/LeRF deletion, stability |
| `06_actionable_model_improvement` | remove artefact features, before/after |
| `07_fairness` | subgroup errors and explanations, mitigation |
| `08_sensitivity_duplicates` | what if duplicates are kept |
| `09_plausibility` | fidelity vs plausibility against CDC risk factors |
| `10_cost_threshold` | cost-benefit of the threshold, decision curve |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact tested versions (Python 3.13); requirements.txt has minimum versions
./run_all.sh                              # downloads the data and executes notebooks 01-10 in order (~30-40 min on CPU)
```

Run the notebooks **in order**: `02` creates the models used by `03-07`, `09`, `10`, and `06` creates the reduced model used by `07`, `09`, `10`.
Seeds are fixed (split seed 42, SHAP samples with fixed generators).

## Main design decisions

- Exact duplicate rows are dropped before splitting (9.5 %; they are different respondents with identical answers, so this is a conservative choice; see notebook 08).
- Split 70/15/15 stratified; validation picks hyper-parameters, early stopping and the threshold; **test is only used for final numbers and explanations**.
- No resampling or class weights: the 15 % positive rate is handled with threshold selection and PR-AUC/recall/calibration.
- The decision threshold is a recall target (≥ 80 %), justified in notebook 10.

## Limitations

- The label is *diagnosed, self-reported* diabetes: undiagnosed cases look healthy, which is what the artefact features exploit. Interpreting `CholCheck` as detection bias is a plausible hypothesis, not something the data prove.
- One train/validation/test split; some subgroups have few positives (e.g. young people).
- SHAP uses a 100-row background; LIME and SHAP are not directly comparable (probability vs log-odds).
- Counterfactuals describe what changes the **model output**, not a person's real risk; the explanation of a faithful model can still mislead if read causally.

## References and credits

- Dataset: CDC Behavioral Risk Factor Surveillance System, via the UCI Machine Learning Repository (id 891).
- Libraries: LightGBM, scikit-learn, [SHAP](https://github.com/shap/shap) (Lundberg & Lee, 2017), [LIME](https://github.com/marcotcr/lime) (Ribeiro et al., 2016), pandas, NumPy, SciPy, matplotlib, seaborn, `ucimlrepo`.
- Evaluation ideas: label-randomisation sanity checks (Adebayo et al., 2018); deletion / ROAR / ROAD-style evaluation (Hooker et al., 2019; Rong et al., 2022); decision curve analysis (Vickers & Elkin, 2006); course material of *Explainable AI*.
- Medical reference for plausibility: CDC Prediabetes Risk Test scoring and CDC *Diabetes and Prediabetes* fact sheet (links in notebook 09).
- Counterfactual search is a small exhaustive implementation (`src/counterfactuals.py`) written for this project.
- Code was developed with the assistance of a generative AI tool (Claude, Anthropic); the author reviewed the code and wrote the analysis.
