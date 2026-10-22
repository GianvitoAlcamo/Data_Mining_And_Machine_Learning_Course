# Classification of Human Body Postures through Wearable Sensors
**Data Mining & Machine Learning — MSc in Artificial Intelligence and Data Engineering**  
*University of Pisa, A.Y. 2025–2026*  
**Authors:** Gianvito Alcamo, Matteo Billi Ciani

---

## 1. The Problem

The goal is to automatically classify **human body postures and movements** from wearable accelerometer data. The dataset (UCI PUC-Rio, ~165,000 samples, 4 subjects) contains raw tri-axial acceleration readings from 4 body-worn sensors (waist, left thigh, right ankle, right upper arm), sampled at 8 Hz.

The **5 target classes** are:
- Static postures: `sitting`, `standing`, `walking`
- Transitional movements: `sitting-down`, `standing-up`

This is a **multi-class classification** task evaluated with weighted F1-score. The main challenge is generalization to unseen users: transitional classes are short and ambiguous, and the dataset covers only 4 subjects.

---

## 2. Approach

The pipeline separates dataset optimization from model selection:

1. **Exploratory Data Analysis** — raw signal visualization, correlation analysis, outlier detection, class balance inspection.
2. **Feature Engineering** — several dataset variants were compared using a Decision Tree as a proxy model:

   | Variant | Avg F1 |
   |---|---|
   | Baseline (raw) | 0.378 |
   | Augmented (+ magnitude, roll, pitch, yaw) | 0.429 |
   | **Windowed (8-sample windows, stride 7)** | **0.674** |
   | SMOTE oversampling | 0.618 |
   | Synthetic users | 0.599 |
   | PCA | 0.415 |

   The **windowed representation** (mean, variance, min, max per window) was selected as best. Orientation-based features and PCA did not improve generalization under LOSO. Demographic variables (age, weight, height) were removed after SHAP analysis showed them to be among the least informative.

3. **Model Comparison** — five classifiers were compared under the same nested LOSO cross-validation: Decision Tree, Random Forest, AdaBoost, k-NN, Naive Bayes.

4. **Validation — Nested LOSO Cross-Validation** — outer loop: Leave-One-Subject-Out (one user held out at a time); inner loop: stratified k-fold for hyperparameter tuning. All preprocessing steps (scaling, SMOTE, discretization) were fitted only on training folds to prevent data leakage.

---

## 3. Key Design Choices

- **Windowed features over raw signals.** Grouping 8 consecutive samples and computing summary statistics (mean, variance, min, max) per window increased F1 from 0.38 to 0.67. The window is labelled with the majority class only when ≥60% of samples agree.
- **LOSO over standard k-fold.** Standard k-fold allows samples from the same subject in both train and test, leading to over-optimistic results. LOSO simulates real deployment on an unseen user and provides a more honest estimate.
- **Random Forest as final model.** RF achieved the highest and most consistent LOSO F1-scores. A Bayesian probability-of-superiority analysis confirmed RF is better than NB (96.0%), k-NN (94.7%), DT (86.8%), and AdaBoost (86.6%) — formal Wilcoxon significance was not reachable due to the very small sample size (n=4 folds).
- **SHAP for explainability.** Global SHAP values showed sensors on the waist and left thigh as most informative. Local SHAP analysis confirmed the model's tendency to favor majority classes for ambiguous transition windows.

---

## 4. Results and Limitations

**Random Forest** achieved the best overall weighted F1-score under LOSO. Static and cyclic classes (`sitting`, `standing`, `walking`) are reliably recognized. The main source of error is confusion between `sitting-down` and `standing-up`: window statistics capture movement intensity but not the direction of the transition.

Subject `katia` was the hardest held-out user — removing her from training slightly improved generalization, indicating she is the most distinctive subject in the dataset.

**Main limitations:**
- Only 4 subjects — cross-subject generalization is inherently limited.
- Low sampling frequency (8 Hz) reduces temporal resolution.
- Window-based features lose the direction of transitional movements.

**Future directions:** more subjects, sequential models (e.g. RNNs, HMMs) to encode temporal context, federated learning to improve generalization without sharing raw personal data.

---

## 5. Repository Structure

```
├── Code/
│   ├── 1_DATA_EXPLORATION/         # EDA: signal visualization, correlation, outliers
│   ├── 2_from_BASELINE_to_BEST/    # Dataset variants comparison, SHAP feature selection
│   ├── 3_MODELS_COMPARISON/        # Nested LOSO CV, hyperparameter tuning
│   └── 4_RESULTS_ANALYSIS/         # Final RF evaluation, error analysis, SHAP
├── Report.pdf                      # Full technical report
└── Slides.pdf                      # Presentation slides
```