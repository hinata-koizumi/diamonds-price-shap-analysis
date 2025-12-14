
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_results_dir():
    if not os.path.exists('results'):
        os.makedirs('results')

def load_and_preprocess():
    print("Loading and preprocessing data...")
    df = pd.read_csv('data/diamonds.csv')
    
    # Ordinal Encoding
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    
    df['cut_enc'] = df['cut'].map(cut_mapping)
    df['color_enc'] = df['color'].map(color_mapping)
    df['clarity_enc'] = df['clarity'].map(clarity_mapping)
    return df

def train_base_model(df):
    print("\n--- 3.1 Model Performance Evaluation (Baseline) ---")
    features = ['carat', 'cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using 100 trees as per original script
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    r2 = rf.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mean_price = y_test.mean()
    error_rate = (mae / mean_price) * 100
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")
    print(f"Mean Error Rate: {error_rate:.1f}%")
    
    # Feature Importance Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(features)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title("Random Forest Feature Importance (MDI)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    return rf, X_train, X_test, y_train, y_test

def robust_validation(df):
    print("\n--- 5. Robustness Check (Group K-Fold by Carat Bins) ---")
    features = ['carat', 'cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z']
    target = 'price'
    
    # 0.05 step bins
    df['carat_bin'] = (df['carat'] / 0.05).astype(int)
    groups = df['carat_bin'].values
    X = df[features]
    y = df[target]
    
    gkf = GroupKFold(n_splits=5)
    r2_scores = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Reduced estimators for speed in robust check
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        score = r2_score(y_test, rf.predict(X_test))
        r2_scores.append(score)
    
    mean_r2 = np.mean(r2_scores)
    print(f"Mean GroupKFold R2: {mean_r2:.4f}")

def exp_no_carat(df):
    print("\n--- 4.2.1 Verification of Size Redundancy (No Carat) ---")
    features_no_carat = ['cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z']
    target = 'price'
    
    X = df[features_no_carat]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"R2 Score (No Carat): {score:.4f}")
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(features_no_carat)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title(f"Feature Importance without Carat")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig('results/deep_dive_no_carat_importance.png')
    plt.close()

def exp_interactions(rf, X_train):
    print("\n--- 4.2.2 Amplification Effect of Quality (Carat x Clarity) ---")
    # Using small sample for SHAP speed
    X_shap = X_train.sample(500, random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure()
    shap.dependence_plot("carat", shap_values, X_shap, interaction_index="clarity_enc", show=False)
    plt.title("SHAP Interaction: Carat vs Clarity")
    plt.tight_layout()
    plt.savefig('results/deep_dive_interaction_carat_clarity.png')
    plt.close()

def exp_unit_price(df):
    print("\n--- 4.2.3 Unit Price Model ---")
    df['price_per_carat'] = df['price'] / df['carat']
    features = ['carat', 'cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z']
    
    X = df[features]
    y = df['price_per_carat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(features)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title("Feature Importance (Target: Price/Carat)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig('results/deep_dive_price_per_carat_importance.png')
    plt.close()

def exp_magic_number(rf, df):
    print("\n--- 4.2.4 Verification of Threshold Effect (0.99 vs 1.00) ---")
    features = ['carat', 'cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z']
    
    # Use global median for baseline features (Color, Clarity, etc.)
    base_values = df[features].median()
    
    # Force 1.00ct and typical dimensions for 1.00ct
    base_values['carat'] = 1.00
    # Median dimensions for 1.00-1.05ct to be realistic
    median_1ct = df[(df['carat'] >= 1.00) & (df['carat'] <= 1.05)][['x', 'y', 'z']].median()
    base_values['x'] = median_1ct['x']
    base_values['y'] = median_1ct['y']
    base_values['z'] = median_1ct['z']
    
    # Simulation
    # Case 1: 1.00 ct
    row_1 = base_values.copy()
    row_1['carat'] = 1.00
    pred_1 = rf.predict(pd.DataFrame([row_1]))[0]
    
    # Case 2: 0.99 ct (Physically scaled)
    row_99 = base_values.copy()
    row_99['carat'] = 0.99
    # Scale x, y, z by cube root of weight ratio
    scale_factor = (0.99 / 1.00) ** (1/3)
    row_99['x'] = row_1['x'] * scale_factor
    row_99['y'] = row_1['y'] * scale_factor
    row_99['z'] = row_1['z'] * scale_factor
    pred_99 = rf.predict(pd.DataFrame([row_99]))[0]
    
    print(f"Weight 0.99 ct: ${pred_99:.0f}")
    print(f"Weight 1.00 ct: ${pred_1:.0f}")
    print(f"Difference: +${pred_1 - pred_99:.0f} (+{(pred_1 - pred_99)/pred_99*100:.1f}%)")

def exp_cut_distribution(df):
    print("\n--- 4.2.5 Cut Distribution ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='cut', data=df, order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    plt.title("Distribution of Cut Grades")
    plt.tight_layout()
    plt.savefig('results/cut_distribution.png')
    plt.close() 

def main():
    create_results_dir()
    df = load_and_preprocess()
    
    # 3.1 & 4.1 Baseline Model
    # 3.1 & 4.1 Baseline Model
    rf_base, X_train, X_test, y_train, y_test = train_base_model(df)
    
    # 5. Robustness
    robust_validation(df)
    
    # 4.2.1 No Carat
    exp_no_carat(df)
    
    # 4.2.2 Interactions
    exp_interactions(rf_base, X_train)
    
    # 4.2.3 Unit Price
    exp_unit_price(df)
    
    # 4.2.4 Magic Number
    exp_magic_number(rf_base, df)
    
    # 4.2.5 Cut Distribution
    exp_cut_distribution(df)
    
    print("\nAnalysis Complete. All artifacts saved in 'results/'.")

if __name__ == "__main__":
    main()
