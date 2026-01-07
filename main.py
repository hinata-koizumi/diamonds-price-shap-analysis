import os
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

# 再現性のためにシードを設定
np.random.seed(42)

# --- 定数 ---
RESULTS_DIR = 'results'
DATA_PATH = 'data/diamonds.csv'

CUT_MAPPING = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
COLOR_MAPPING = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
CLARITY_MAPPING = {
    'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7
}

BASE_FEATURES = [
    'carat', 'cut_enc', 'color_enc', 'clarity_enc', 'depth', 'table', 'x', 'y', 'z'
]
TARGET_COLUMN = 'price'


def create_results_dir() -> None:
    """結果保存用ディレクトリが存在しない場合は作成する。"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def load_and_preprocess() -> pd.DataFrame:
    """データセットを読み込み、順序エンコーディングを行う。"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)

    df['cut_enc'] = df['cut'].map(CUT_MAPPING)
    df['color_enc'] = df['color'].map(COLOR_MAPPING)
    df['clarity_enc'] = df['clarity'].map(CLARITY_MAPPING)
    return df


def plot_feature_importance(model: RandomForestRegressor, features: List[str],
                            title: str, filename: str) -> None:
    """特徴量重要度をプロットし保存する。"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(features)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def train_base_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ベースラインのランダムフォレストモデルを学習し、性能を評価する。"""
    print("\n--- 3.1 Model Performance Evaluation (Baseline) ---")
    
    X = df[BASE_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    plot_feature_importance(
        rf, BASE_FEATURES,
        "Random Forest Feature Importance (MDI)",
        "feature_importance.png"
    )

    return rf, X_train, X_test, y_train, y_test


def robust_validation(df: pd.DataFrame) -> None:
    """カラットのビンに基づくGroupKFoldを用いたロバスト性検証を行う。"""
    print("\n--- 5. Robustness Check (Group K-Fold by Carat Bins) ---")
    
    # 0.05 carat bins
    df['carat_bin'] = (df['carat'] / 0.05).astype(int)
    groups = df['carat_bin'].values
    X = df[BASE_FEATURES]
    y = df[TARGET_COLUMN]

    gkf = GroupKFold(n_splits=5)
    r2_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 速度向上のため推定器数を削減
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        score = r2_score(y_test, rf.predict(X_test))
        r2_scores.append(score)

    mean_r2 = np.mean(r2_scores)
    print(f"Mean GroupKFold R2: {mean_r2:.4f}")


def exp_no_carat(df: pd.DataFrame) -> None:
    """実験: 'carat'特徴量を除外してモデルを学習する。"""
    print("\n--- 4.2.1 Verification of Size Redundancy (No Carat) ---")
    features_no_carat = [f for f in BASE_FEATURES if f != 'carat']

    X = df[features_no_carat]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"R2 Score (No Carat): {score:.4f}")

    plot_feature_importance(
        rf, features_no_carat,
        "Feature Importance without Carat",
        "deep_dive_no_carat_importance.png"
    )


def exp_interactions(rf: RandomForestRegressor, X_train: pd.DataFrame) -> None:
    """実験: SHAPを用いてCaratとClarityの相互作用を分析する。"""
    print("\n--- 4.2.2 Amplification Effect of Quality (Carat x Clarity) ---")
    # 速度向上のため少数のサンプルを使用
    X_shap = X_train.sample(500, random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)

    plt.figure()
    shap.dependence_plot(
        "carat", shap_values, X_shap,
        interaction_index="clarity_enc", show=False
    )
    plt.title("SHAP Interaction: Carat vs Clarity")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'deep_dive_interaction_carat_clarity.png'))
    plt.close()


def exp_unit_price(df: pd.DataFrame) -> None:
    """実験: カラット単価に対する特徴量重要度を分析する。"""
    print("\n--- 4.2.3 Unit Price Model ---")
    df['price_per_carat'] = df['price'] / df['carat']
    
    X = df[BASE_FEATURES]
    y = df['price_per_carat']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    plot_feature_importance(
        rf, BASE_FEATURES,
        "Feature Importance (Target: Price/Carat)",
        "deep_dive_price_per_carat_importance.png"
    )


def exp_magic_number(rf: RandomForestRegressor, df: pd.DataFrame) -> None:
    """実験: 閾値効果（0.99 vs 1.00カラット）を検証する。"""
    print("\n--- 4.2.4 Verification of Threshold Effect (0.99 vs 1.00) ---")
    
    # 基本特徴量に中央値を使用
    base_values = df[BASE_FEATURES].median()

    # 1.00-1.05カラットのダイヤモンドの中央値寸法を取得
    median_1ct = df[(df['carat'] >= 1.00) & (df['carat'] <= 1.05)][['x', 'y', 'z']].median()
    base_values['x'] = median_1ct['x']
    base_values['y'] = median_1ct['y']
    base_values['z'] = median_1ct['z']

    # ケース1: 1.00カラット
    row_1 = base_values.copy()
    row_1['carat'] = 1.00
    pred_1 = rf.predict(pd.DataFrame([row_1]))[0]

    # ケース2: 0.99カラット（物理的にスケーリング）
    row_99 = base_values.copy()
    row_99['carat'] = 0.99
    scale_factor = (0.99 / 1.00) ** (1/3)
    row_99['x'] = row_1['x'] * scale_factor
    row_99['y'] = row_1['y'] * scale_factor
    row_99['z'] = row_1['z'] * scale_factor
    pred_99 = rf.predict(pd.DataFrame([row_99]))[0]

    print(f"Weight 0.99 ct: ${pred_99:.0f}")
    print(f"Weight 1.00 ct: ${pred_1:.0f}")
    diff_pct = (pred_1 - pred_99) / pred_99 * 100
    print(f"Difference: +${pred_1 - pred_99:.0f} (+{diff_pct:.1f}%)")


def exp_cut_distribution(df: pd.DataFrame) -> None:
    """実験: Cutグレードの分布をプロットする。"""
    print("\n--- 4.2.5 Cut Distribution ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x='cut', data=df,
        order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    )
    plt.title("Distribution of Cut Grades")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cut_distribution.png'))
    plt.close()


def main() -> None:
    create_results_dir()
    df = load_and_preprocess()

    # 3.1 & 4.1 ベースラインモデル
    rf_base, X_train, _, _, _ = train_base_model(df)

    # 5. ロバスト性検証
    robust_validation(df)

    # 4.2.1 カラット除外
    exp_no_carat(df)

    # 4.2.2 相互作用
    exp_interactions(rf_base, X_train)

    # 4.2.3 カラット単価
    exp_unit_price(df)

    # 4.2.4 マジックナンバー
    exp_magic_number(rf_base, df)

    # 4.2.5 カットの分布
    exp_cut_distribution(df)

    print("\nAnalysis Complete. All artifacts saved in 'results/'.")


if __name__ == "__main__":
    main()
