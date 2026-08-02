
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from loguru import logger


# ─── Paths ────────────────────────────────────────────────────────────────────

PROCESSED_CSV    = "data/ansur_processed.csv"
COMPARISON_CSV   = "artifacts/size_models_comparison.csv"
BEST_MODEL_PATH  = "artifacts/best_size_model.pkl"
BEST_MODEL_NAME  = "artifacts/size_model_name.txt"

FEATURE_COLS = ["shoulder_cm", "arm_cm", "height_cm", "weight_kg"]
LABEL_COL    = "size_label"
LABEL_ORDER  = ["XS", "S", "M", "L", "XL", "XXL"]


# ─── Step 8 — Load data and split ─────────────────────────────────────────────

def load_and_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the preprocessed (already-scaled) dataset and perform an
    80/20 stratified train/test split. Stratification guarantees
    every size label appears proportionally in both train and test —
    critical here since XS has only 41 total samples.
    """
    df = pd.read_csv(PROCESSED_CSV)
    logger.info(f"[Train] Loaded {len(df)} rows from {PROCESSED_CSV}")

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    logger.info(f"[Train] Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    train_dist = pd.Series(y_train).value_counts().reindex(LABEL_ORDER, fill_value=0)
    test_dist  = pd.Series(y_test).value_counts().reindex(LABEL_ORDER, fill_value=0)
    logger.info(f"[Train] Train distribution:\n{train_dist}")
    logger.info(f"[Train] Test distribution:\n{test_dist}")

    return X_train, X_test, y_train, y_test


# ─── Step 9 — Define all models ───────────────────────────────────────────────

def get_models() -> dict:
    """
    All models use class_weight='balanced' where supported, since the
    dataset is imbalanced (L: 1,314 samples vs XS: 41 samples).
    Balanced weighting penalises misclassifying minority classes more
    heavily during training, preventing the model from just always
    predicting the majority classes (M/L).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",   # closer neighbours matter more
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,      # needed for predict_proba at inference
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
    }


# ─── Step 10 — Train and evaluate each model ──────────────────────────────────

def train_and_evaluate(
    models: dict,
    X_train, y_train, X_test, y_test,
) -> pd.DataFrame:
    """
    Train every model on the identical split, print a full
    classification report per model, and collect summary metrics
    for the final comparison table.
    """
    results = []

    for name, model in models.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"[Train] Training: {name}")
        logger.info(f"{'='*70}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc        = accuracy_score(y_test, y_pred)
        f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(
            y_test, y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        )
        logger.info(f"\n[Report] {name}\n{report}")

        cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
        cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
        logger.info(f"[Confusion Matrix] {name}\n{cm_df}")

        results.append({
            "model":        name,
            "accuracy":     round(acc, 4),
            "f1_macro":     round(f1_macro, 4),
            "f1_weighted":  round(f1_weighted, 4),
        })

        logger.success(
            f"[Result] {name}: accuracy={acc:.4f}, "
            f"f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}"
        )

    comparison_df = pd.DataFrame(results).sort_values(
        "f1_macro", ascending=False
    ).reset_index(drop=True)

    return comparison_df


# ─── Step 11 — Select and save the best model ────────────────────────────────

def select_best_model(
    models: dict,
    comparison_df: pd.DataFrame,
    X_train, y_train,
) -> tuple[str, object]:
    """
    Selection criterion: highest macro F1 score.

    Macro F1 averages the F1 score across all 6 classes EQUALLY —
    so a model that nails M/L but completely fails on XS/XXL gets
    penalised, even if its raw accuracy looks high. This is the
    correct metric for an imbalanced 6-class problem.
    """
    best_name = comparison_df.iloc[0]["model"]
    best_model = models[best_name]

    logger.success(
        f"\n[Selection] 🏆 Best model: {best_name} "
        f"(macro F1 = {comparison_df.iloc[0]['f1_macro']:.4f})"
    )

    return best_name, best_model


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    Path("artifacts").mkdir(exist_ok=True)

    # Step 8
    X_train, X_test, y_train, y_test = load_and_split()

    # Step 9
    models = get_models()

    # Step 10
    comparison_df = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    logger.info(f"\n{'='*70}")
    logger.info("[Comparison] Final model comparison (sorted by macro F1)")
    logger.info(f"{'='*70}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    comparison_df.to_csv(COMPARISON_CSV, index=False)
    logger.success(f"[Train] Saved comparison table → {COMPARISON_CSV}")

    # Step 11
    best_name, best_model = select_best_model(models, comparison_df, X_train, y_train)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.success(f"[Train] Saved best model → {BEST_MODEL_PATH}")

    with open(BEST_MODEL_NAME, "w") as f:
        f.write(best_name)
    logger.success(f"[Train] Saved model name → {BEST_MODEL_NAME}")

    logger.info(
        "\n[Train] Next step: integrate best model with FastAPI\n"
        "    Update src/models/size_fit_model.py and src/api/main.py"
    )


if __name__ == "__main__":
    main()