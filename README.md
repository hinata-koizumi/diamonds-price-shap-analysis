# Diamond Price SHAP Analysis

This project performs a comprehensive price analysis on the `diamonds` dataset using machine learning models, specifically Random Forest Regressor. It leverages SHAP (SHapley Additive exPlanations) to uncover the market structures behind diamond pricing.

## Overview

The analysis includes:

- **Model prediction**: Accurate price prediction using Random Forest ($R^2 > 0.98$).
- **Feature importance analysis**: Interpretability using MDI and SHAP values.
- **Hypothesis verification**:
    - **Magic Number**: Quantifying the premium for 1.00ct vs 0.99ct.
    - **Quality Multiplier**: Analyzing how Clarity/Color amplify the base price.
    - **Cut Bias**: Explaining why "Cut" appears less important than expected.
- **Robustness checks**: Validating findings using Carat Group Splitting to prevent data leakage.

## Requirements

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## Usage

Run the main analysis script:

```bash
python main.py
```

This will:

1.  Load data from the `data/` directory.
2.  Train models and perform all experiments (Baseline, No-Carat, Interactions, Magic Number).
3.  Generate visualizations and save them to the `results/` directory.

## Project Structure

```
diamonds-price-shap-analysis/
├── data/                    # Dataset files (diamonds.csv)
├── results/                 # Generated plots and outputs
├── main.py                  # Main analysis script
├── report.md                # Analysis report (Japanese)
├── report_en.md             # Analysis report (English)
└── README.md                # Project documentation
```

## Documentation

Detailed analysis reports are available in both Japanese and English:

- **[日本語レポート (Japanese Report)](report.md)** - 詳細な分析結果と考察
- **[English Report](report_en.md)** - Detailed analysis results and discussion

Both reports include:

- Experimental setup and methodology
- Model performance comparisons
- Feature importance analysis (MDI & SHAP)
- Detailed validation of "Magic Number" and "Quality Multiplier" hypotheses
- Conclusions and future work

## Results

All generated visualizations are saved in the `results/` directory:

- `feature_importance.png`: Baseline feature importance
- `deep_dive_no_carat_importance.png`: Importance when Carat is removed (proxy analysis)
- `deep_dive_interaction_carat_clarity.png`: SHAP interaction plot (Weight vs Quality)
- `deep_dive_price_per_carat_importance.png`: Feature importance for Unit Price model
- `cut_distribution.png`: Distribution of Cut grades explaining selection bias

## Key Findings

- **Best Model**: Random Forest achieved high accuracy ($R^2 \approx 0.98$) and robust generalization ($R^2 \approx 0.96$ in Group Split).
- **Dominant Feature**: Carat and dimensional variables (x, y, z) are the primary price drivers.
- **Magic Number**: A 1.00ct diamond commands a **~25.5% premium** over a physically scaled 0.99ct stone.
- **Quality Multiplier**: Clarity and Color act as multipliers that amplify the base price determined by size.
- **Cut Ambiguity**: Cut grade serves as a "minimum requirement" rather than a differentiation factor, due to high market standards.

## License

This project is for research and educational purposes.
