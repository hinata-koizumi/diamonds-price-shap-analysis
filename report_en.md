# 1. Introduction
In this report, we constructed a regression model to predict diamond prices using the `diamonds` dataset and analyzed the variables contributing to price determination and model performance evaluation. For this analysis, we used the `diamonds` data included in R's `ggplot2` package, obtained via `seaborn-data` in the Python environment [1].

Specifically, we built a price prediction model using a Random Forest Regressor and interpreted the market valuation structure behind price formation using Feature Importance and SHAP values (average contribution of each feature to the predicted value). To ensure the robustness of the evaluation, we conducted a "Carat Group Split" validation in addition to the standard random split, evaluating performance while avoiding overfitting to specific carat bands.

Furthermore, we quantitatively verified the "price gap between 0.99 carats and 1.00 carats," widely known in the market, through a simulation with consistent physical dimensions.

# 2. Experimental Setup
## 2.1 Dataset

**Table 1: Classification and Role of Features in the diamonds Dataset**

| Category | Variable | Type | Role/Note |
| --- | --- | --- | --- |
| Size | carat, x, y, z | Numeric | Primary factors representing size. Strong correlation between carat and x, y, z. |
| Quality | clarity, color | Categorical | Quality evaluation. Interpreted as amplification factors for the base price. |
| Processing | cut | Categorical | Cut quality. Often presumed to be above a certain standard. |

## 2.2 Preprocessing and Splitting Methods
Training and evaluation were conducted using the following two splitting methods.

**Table 2: Comparison of Data Splitting Methods for Training/Evaluation**

| Method | Criteria | Purpose | Characteristics/Notes |
| --- | --- | --- | --- |
| Random Split | Randomly split all data | Basic model performance evaluation | Prone to information leakage of size bands, leading to optimistic evaluations. |
| Carat Group Split | Grouping Carat by 0.05ct intervals | Generalization performance verification of price structure | Prevents identical carat bands from spanning training/evaluation, suppressing overestimation due to memorization. |

## 2.3 Model Used
We used the Random Forest Regressor for this experiment.

# 3. Experimental Results
## 3.1 Model Performance Evaluation
The performance of the constructed Random Forest regression model is shown below.

**Table 3: Price Prediction Performance Metrics of Random Forest Regression Model**

| Metric | Value |
| --- | --- |
| $R^2$ Score | 0.9816 |
| MAE | $266.57 |
| Mean Error Rate | Approx. 6.8% |

In addition to the random split, the model maintained an average $R^2 = 0.9676$ in the Carat Group Split validation.
This result indicates that the model is learning a price determination logic based on size and quality, rather than simply memorizing individual data points.

# 4. Comparison and Discussion of Feature Importance
## 4.1 Baseline Analysis: Feature Importance
Checking the feature importance of the Random Forest showed that Carat (63.3%) and y (25.5%) accounted for about 90% of the total.

[![Feature Importance by Random Forest (Baseline)](results/feature_importance.png)](results/feature_importance.png)
**Figure 1: Feature Importance by Random Forest (Baseline)**

This result indicates that price is primarily and strongly defined by size.
The high importance of `y` can be interpreted not as the importance of width itself, but as functioning as a proxy variable that, in combination with `carat`, expressively represents the "size perception" from multiple angles.

## 4.2 Detailed Analysis by SHAP
### 4.2.1 Verification of Size Redundancy
Re-training with `carat` excluded from the explanatory variables resulted in a performance of $R^2 = 0.9806$, a decrease of only 0.001.

[![Feature Importance after Excluding Carat](results/deep_dive_no_carat_importance.png)](results/deep_dive_no_carat_importance.png)
**Figure 2: Feature Importance after Excluding Carat**

This result shows that since size information is redundantly included in x, y, and z, the dimensional features substituted for the missing `carat`. At the same time, this is consistent with the possibility that the market emphasizes "visual size perception" rather than weight itself.

### 4.2.2 Amplification Effect of Quality
We analyzed the interaction between `carat` and `clarity` using SHAP Interaction values.

[![SHAP Interaction Plot: Carat × Clarity](results/deep_dive_interaction_carat_clarity.png)](results/deep_dive_interaction_carat_clarity.png)
**Figure 3: SHAP Interaction Plot: Carat × Clarity**

It was consistently observed that for the same Carat, individuals with better Clarity had a larger SHAP value impact on price for an increase in weight. This suggests that quality functions not merely as an additive element, but as a "multiplier" for the base price.

### 4.2.3 Re-evaluation of Quality Factors via Unit Price Model
When learning with the target variable changed to `price / carat`, the importance of `clarity` and `color` rose to a total of about 25%.

[![Feature Importance of Unit Price Model](results/deep_dive_price_per_carat_importance.png)](results/deep_dive_price_per_carat_importance.png)
**Figure 4: Feature Importance of Unit Price Model**

While size appears dominant for the total price, it was confirmed that quality factors strongly influence the "unit price" perspective.

### 4.2.4 Verification of Threshold Effect
With other features fixed to representative values and x, y, z scaled to be physically consistent with carat, we compared the predicted prices for 0.99ct and 1.00ct.

**Table 4: Comparison of Predicted Prices at 0.99ct and 1.00ct (Magic Number Verification)**

| Weight | Predicted Price | Note |
| --- | --- | --- |
| 0.99 ct | $4,835 | Physically scaled down |
| 1.00 ct | $6,067 | Market median value |
| Difference | +$1,232 (+25.5%) | |

A price difference of about 25% occurred for a weight difference of only 1%, quantitatively confirming the existence of a premium for the discrete label of 1 carat.

### 4.2.5 Why Cut Doesn't Work
Investigating the background of the low importance of Cut, we found that about 87% of the data was concentrated in "Very Good" or higher.

[![Distribution of Cut Grades](results/cut_distribution.png)](results/cut_distribution.png)
**Figure 5: Distribution of Cut Grades**

From this result, it is highly likely that Cut functions not as a differentiation factor, but as a "minimum requirement" for participating in the market.

# 5. Robustness of Evaluation Protocol
Even in the Carat Group Split validation, the model maintained an average $R^2 = 0.9676$.
This indicates that the price formation structures, such as the Magic Number and Quality Multiplier, are likely general trends independent of specific size bands.

# 6. Conclusion
As a result of this analysis, the diamond price formation can be organized into a three-layer structure. First, as a prerequisite, Cut is required to be above a certain standard. On top of that, size and the so-called "Magic Number" determine the baseline price standard, and finally, quality factors such as Clarity and Color play the role of amplifying that base price. Thus, it is suggested that diamond price formation is not merely a reflection of physical characteristics but a result of the complex interaction of market psychology and quality evaluation.

# 7. Limitations and Future Issues
Since this analysis depends on the specific distribution characteristics of the `diamonds` dataset, care must be taken when directly extrapolating the findings to the entire actual market. Also, there is so-called collinearity among size-related variables such as carat and dimensions, and as a result, there is a constraint that the assignment of feature importance may vary depending on the model and representation method. Future tasks include conducting comparative analyses with Boosting-based models or linear models to verify the consistency and robustness of the interpretation results.

# 8. References
[1] seaborn-data/diamonds: Diamonds dataset (version 1.0.0) (https://github.com/seaborn-data/diamonds)
