from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    Train an XGBoost classifier for AML detection.
    """

    # ---------------------------------------------------------
    # Calculate class imbalance
    # ---------------------------------------------------------

    normal_count = (y_train == 0).sum()
    laundering_count = (y_train == 1).sum()

    scale_pos_weight = normal_count / laundering_count

    print("\nClass distribution:")
    print(f"Normal:      {normal_count:,}")
    print(f"Laundering:  {laundering_count:,}")

    print(
        f"\nscale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        random_state=42,

        n_jobs=-1
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("✓ Training complete.")

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_path = MODEL_DIR / "xgboost_aml_model.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to:")
    print(model_path)

    return model