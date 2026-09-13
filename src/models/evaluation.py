import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Prediction probability
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # Default threshold
    # ---------------------------------------------------------

    threshold = 0.5

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ---------------------------------------------------------
    # PR-AUC
    # ---------------------------------------------------------

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(
        f"\nPR-AUC: {pr_auc:.4f}"
    )

    # ---------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "probabilities": probabilities,
        "predictions": predictions
    }