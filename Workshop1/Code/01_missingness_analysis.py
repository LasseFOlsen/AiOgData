from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Data" / "Raw" / "Sales_with_NaNs_v1.3.csv"
OUTPUT_DIR = BASE_DIR / "Output" / "Task1" / "01_missingness_analysis"
RANDOM_STATE = 42


def SaveFigure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def GetColumnTypes(data, excluded_column=None):
    columns = [column for column in data.columns if column != excluded_column]
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(data[column])]
    categorical_columns = [column for column in columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def BuildPreprocessor(features):
    numeric_columns, categorical_columns = GetColumnTypes(features)
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(DATA_PATH)
missing_columns = data.columns[data.isna().any()].tolist()

# Først laver vi et hurtigt overblik over hvor der mangler værdier.
missing_summary = (
    pd.DataFrame(
        {
            "feature": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_percent": data.isna().mean().values * 100,
        }
    )
    .sort_values("missing_percent", ascending=False)
    .reset_index(drop=True)
)
missing_summary.to_csv(OUTPUT_DIR / "missing_summary.csv", index=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=missing_summary, x="missing_percent", y="feature", color="#3f7f93")
plt.title("Missing values per feature")
plt.xlabel("Missing percent")
plt.ylabel("Feature")
SaveFigure(OUTPUT_DIR / "missing_percent_by_feature.png")

missing_correlation = data.isna().astype(int).corr()
missing_correlation.to_csv(OUTPUT_DIR / "missingness_indicator_correlation.csv")
plt.figure(figsize=(8, 6))
sns.heatmap(missing_correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between missingness indicators")
SaveFigure(OUTPUT_DIR / "missingness_indicator_correlation.png")

# Her sammenligner vi rækker hvor en værdi findes med rækker hvor den mangler.
numeric_rows = []
categorical_rows = []

for missing_column in missing_columns:
    is_missing = data[missing_column].isna()
    numeric_columns, categorical_columns = GetColumnTypes(data, missing_column)

    for column in numeric_columns:
        observed = data.loc[~is_missing, column].dropna()
        missing = data.loc[is_missing, column].dropna()
        if observed.empty or missing.empty:
            continue

        _, p_value = mannwhitneyu(observed, missing, alternative="two-sided")
        numeric_rows.append(
            {
                "missing_feature": missing_column,
                "comparison_feature": column,
                "observed_mean": observed.mean(),
                "missing_mean": missing.mean(),
                "mean_difference": missing.mean() - observed.mean(),
                "mann_whitney_p_value": p_value,
            }
        )

    for column in categorical_columns:
        table = pd.crosstab(
            is_missing.map({False: "observed", True: "missing"}),
            data[column].fillna("Missing"),
        )
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        _, p_value, _, _ = chi2_contingency(table)
        proportions = table.div(table.sum(axis=1), axis=0)
        for category in table.columns:
            categorical_rows.append(
                {
                    "missing_feature": missing_column,
                    "comparison_feature": column,
                    "category": category,
                    "observed_share": proportions.loc["observed", category],
                    "missing_share": proportions.loc["missing", category],
                    "share_difference": proportions.loc["missing", category] - proportions.loc["observed", category],
                    "chi_square_p_value": p_value,
                }
            )

pd.DataFrame(numeric_rows).to_csv(OUTPUT_DIR / "observed_vs_missing_numeric.csv", index=False)
pd.DataFrame(categorical_rows).to_csv(OUTPUT_DIR / "observed_vs_missing_categorical.csv", index=False)

# Til sidst tester vi, om missingness kan forudsiges fra resten af data.
model_rows = []
all_indicators = data.isna().astype(int).add_prefix("is_missing_")

for missing_column in missing_columns:
    target = data[missing_column].isna().astype(int)
    feature_sets = {
        "observed_features": data.drop(columns=[missing_column]),
        "other_missingness_indicators": all_indicators.drop(columns=[f"is_missing_{missing_column}"]),
    }

    for feature_set_name, features in feature_sets.items():
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        models = {
            "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": Pipeline(
                [
                    ("preprocess", BuildPreprocessor(features)),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            "lightgbm": Pipeline(
                [
                    ("preprocess", BuildPreprocessor(features)),
                    ("model", LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1)),
                ]
            ),
        }

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)[:, 1]
            model_rows.append(
                {
                    "missing_feature": missing_column,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, predictions),
                    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "average_precision": average_precision_score(y_test, probabilities),
                }
            )

model_results = pd.DataFrame(model_rows)
model_results.to_csv(OUTPUT_DIR / "missingness_model_results.csv", index=False)
best_auc = model_results.groupby("missing_feature")["roc_auc"].max().reset_index()

notes = [
    "Missingness analysis",
    "",
    f"Rows: {len(data)}",
    f"Columns: {len(data.columns)}",
    f"Total missing cells: {int(data.isna().sum().sum())}",
    "",
    "Missing percent per feature:",
    missing_summary.to_string(index=False),
    "",
    "Best ROC AUC per missing feature:",
    best_auc.to_string(index=False),
]
(OUTPUT_DIR / "missingness_analysis_notes.txt").write_text("\n".join(notes), encoding="utf-8")
