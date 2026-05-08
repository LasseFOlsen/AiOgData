from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "Data" / "Raw" / "Sales_with_NaNs_v1.3.csv"
PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
OUTPUT_DIR = BASE_DIR / "Output" / "Task1" / "02_imputation_evaluation"
RANDOM_STATE = 42
N_SPLITS = 5


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBM.* was fitted with feature names",
    category=UserWarning,
)


def GetColumnTypes(data):
    numeric_columns = [
        column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical_columns = [column for column in data.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def SaveFigure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def ImputeMedianMode(data):
    numeric_columns, categorical_columns = GetColumnTypes(data)
    imputed = data.copy()

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        imputed[numeric_columns] = numeric_imputer.fit_transform(imputed[numeric_columns])

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        imputed[categorical_columns] = categorical_imputer.fit_transform(
            imputed[categorical_columns]
        )

    return imputed


def ImputeKnn(data, n_neighbors=5):
    numeric_columns, categorical_columns = GetColumnTypes(data)
    encoded = data.copy()
    mappings = {}

    for column in categorical_columns:
        categories = sorted(data[column].dropna().unique())
        category_to_code = {category: index for index, category in enumerate(categories)}
        code_to_category = {index: category for category, index in category_to_code.items()}
        encoded[column] = data[column].map(category_to_code)
        mappings[column] = code_to_category

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(encoded),
        columns=encoded.columns,
        index=encoded.index,
    )
    imputed_scaled = pd.DataFrame(
        KNNImputer(n_neighbors=n_neighbors, weights="distance").fit_transform(scaled),
        columns=encoded.columns,
        index=encoded.index,
    )
    imputed_encoded = pd.DataFrame(
        scaler.inverse_transform(imputed_scaled),
        columns=encoded.columns,
        index=encoded.index,
    )

    imputed = imputed_encoded.copy()
    for column in categorical_columns:
        valid_codes = list(mappings[column].keys())
        imputed[column] = (
            imputed[column]
            .round()
            .clip(min(valid_codes), max(valid_codes))
            .astype(int)
            .map(mappings[column])
        )

    imputed[numeric_columns] = imputed[numeric_columns].astype(float)
    return imputed[data.columns]


def BuildLightgbmPreprocessor(features):
    numeric_columns, categorical_columns = GetColumnTypes(features)
    transformers = []

    if numeric_columns:
        transformers.append(("numeric", "passthrough", numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def ImputeLightgbm(data):
    numeric_columns, categorical_columns = GetColumnTypes(data)
    imputed = ImputeMedianMode(data)
    complete_features = ImputeMedianMode(data)

    for target_column in data.columns[data.isna().any()]:
        observed_rows = data[target_column].notna()
        missing_rows = data[target_column].isna()
        features = complete_features.drop(columns=[target_column])

        # Vi træner kun på kendte værdier og udfylder kun de tomme celler.
        if target_column in numeric_columns:
            model = Pipeline(
                [
                    ("preprocess", BuildLightgbmPreprocessor(features)),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=150,
                            learning_rate=0.05,
                            random_state=RANDOM_STATE,
                            verbosity=-1,
                        ),
                    ),
                ]
            )
            model.fit(features.loc[observed_rows], data.loc[observed_rows, target_column])
            imputed.loc[missing_rows, target_column] = model.predict(features.loc[missing_rows])
        else:
            categories = sorted(data.loc[observed_rows, target_column].unique())
            category_to_code = {category: index for index, category in enumerate(categories)}
            code_to_category = {index: category for category, index in category_to_code.items()}
            target = data.loc[observed_rows, target_column].map(category_to_code)
            model = Pipeline(
                [
                    ("preprocess", BuildLightgbmPreprocessor(features)),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=150,
                            learning_rate=0.05,
                            random_state=RANDOM_STATE,
                            verbosity=-1,
                        ),
                    ),
                ]
            )
            model.fit(features.loc[observed_rows], target)
            predicted_codes = model.predict(features.loc[missing_rows])
            imputed.loc[missing_rows, target_column] = [
                code_to_category[int(code)] for code in predicted_codes
            ]

    return imputed[data.columns]


PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

original = pd.read_csv(RAW_DATA_PATH)

# De tre metoder sammenlignes på præcis samme datasæt.
imputed_versions = {
    "median_mode": ImputeMedianMode(original),
    "knn": ImputeKnn(original),
    "lightgbm": ImputeLightgbm(original),
}

for method_name, imputed in imputed_versions.items():
    imputed.to_csv(PROCESSED_DIR / f"Sales_with_NaNs_v1.3_imputed_{method_name}.csv", index=False)

numeric_columns, categorical_columns = GetColumnTypes(original)
validation_rows = []

# Vi skjuler kendte værdier, så vi kan måle hvor tæt imputationen rammer.
for column in original.columns:
    observed_index = original.index[original[column].notna()].to_numpy()
    observed_values = original.loc[observed_index, column]
    if pd.api.types.is_numeric_dtype(observed_values):
        stratification_labels = pd.qcut(observed_values, q=N_SPLITS, labels=False, duplicates="drop")
    else:
        stratification_labels = observed_values.astype(str)

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold_number, (_, validation_position) in enumerate(
        splitter.split(observed_index, stratification_labels),
        start=1,
    ):
        masked_index = observed_index[validation_position]
        masked = original.copy()
        true_values = original.loc[masked_index, column].copy()
        masked.loc[masked_index, column] = np.nan
        validation_imputations = {
            "median_mode": ImputeMedianMode(masked),
            "knn": ImputeKnn(masked),
            "lightgbm": ImputeLightgbm(masked),
        }

        for method_name, imputed in validation_imputations.items():
            predicted_values = imputed.loc[masked_index, column]
            if column in numeric_columns:
                validation_rows.append(
                    {
                        "feature": column,
                        "method": method_name,
                        "feature_type": "numeric",
                        "fold": fold_number,
                        "mae": mean_absolute_error(true_values, predicted_values),
                        "rmse": mean_squared_error(true_values, predicted_values) ** 0.5,
                        "accuracy": np.nan,
                        "masked_values": len(masked_index),
                    }
                )
            else:
                validation_rows.append(
                    {
                        "feature": column,
                        "method": method_name,
                        "feature_type": "categorical",
                        "fold": fold_number,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "accuracy": accuracy_score(true_values, predicted_values),
                        "masked_values": len(masked_index),
                    }
                )

validation_results = pd.DataFrame(validation_rows)

# Her tjekker vi om fordelingen ændrer sig efter imputation.
numeric_rows = []
for column in numeric_columns:
    observed = original[column].dropna()
    numeric_rows.append(
        {
            "feature": column,
            "method": "observed_only",
            "mean": observed.mean(),
            "median": observed.median(),
            "std": observed.std(),
            "missing_count": original[column].isna().sum(),
        }
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(observed, kde=True, stat="density", label="Observed", color="#4c6f91")
    for method_name, imputed in imputed_versions.items():
        values = imputed[column]
        numeric_rows.append(
            {
                "feature": column,
                "method": method_name,
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "missing_count": imputed[column].isna().sum(),
            }
        )
        sns.histplot(values, kde=True, stat="density", label=method_name, alpha=0.35)

    plt.title(f"Distribution before and after imputation: {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    SaveFigure(OUTPUT_DIR / f"{column}_histogram_imputation_comparison.png")

numeric_distribution = pd.DataFrame(numeric_rows)

categorical_rows = []
for column in categorical_columns:
    observed_props = original[column].dropna().value_counts(normalize=True)
    for category, proportion in observed_props.items():
        categorical_rows.append(
            {
                "feature": column,
                "method": "observed_only",
                "category": category,
                "proportion": proportion,
                "missing_count": original[column].isna().sum(),
            }
        )

    for method_name, imputed in imputed_versions.items():
        proportions = imputed[column].value_counts(normalize=True)
        for category, proportion in proportions.items():
            categorical_rows.append(
                {
                    "feature": column,
                    "method": method_name,
                    "category": category,
                    "proportion": proportion,
                    "missing_count": imputed[column].isna().sum(),
                }
            )

    plot_data = pd.DataFrame(categorical_rows)
    plot_data = plot_data[plot_data["feature"] == column]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="category", y="proportion", hue="method")
    plt.title(f"Category proportions before and after imputation: {column}")
    plt.xlabel(column)
    plt.ylabel("Proportion")
    SaveFigure(OUTPUT_DIR / f"{column}_category_proportion_comparison.png")

categorical_distribution = pd.DataFrame(categorical_rows)

relationship_rows = []
observed_corr = original[numeric_columns].corr()
for method_name, imputed in imputed_versions.items():
    imputed_corr = imputed[numeric_columns].corr()
    difference = imputed_corr - observed_corr
    for row_column in numeric_columns:
        for col_column in numeric_columns:
            if row_column >= col_column:
                continue
            relationship_rows.append(
                {
                    "method": method_name,
                    "feature_1": row_column,
                    "feature_2": col_column,
                    "observed_correlation": observed_corr.loc[row_column, col_column],
                    "imputed_correlation": imputed_corr.loc[row_column, col_column],
                    "absolute_change": abs(difference.loc[row_column, col_column]),
                }
            )

relationship_results = pd.DataFrame(relationship_rows)

numeric_summary = (
    validation_results[validation_results["feature_type"] == "numeric"]
    .groupby("method")[["mae", "rmse"]]
    .mean()
    .reset_index()
)
categorical_summary = (
    validation_results[validation_results["feature_type"] == "categorical"]
    .groupby("method")["accuracy"]
    .mean()
    .reset_index()
)
relationship_summary = (
    relationship_results.groupby("method")["absolute_change"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_change": "mean_absolute_correlation_change"})
)
numeric_changes = numeric_distribution[numeric_distribution["method"] != "observed_only"].copy()
observed_numeric = numeric_distribution[numeric_distribution["method"] == "observed_only"][
    ["feature", "mean", "median", "std"]
].rename(columns={"mean": "observed_mean", "median": "observed_median", "std": "observed_std"})
numeric_changes = numeric_changes.merge(observed_numeric, on="feature")
numeric_changes["absolute_mean_change"] = (
    numeric_changes["mean"] - numeric_changes["observed_mean"]
).abs()
distribution_summary = (
    numeric_changes.groupby("method")["absolute_mean_change"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_mean_change": "mean_absolute_mean_change"})
)
method_summary = (
    numeric_summary.merge(categorical_summary, on="method")
    .merge(relationship_summary, on="method")
    .merge(distribution_summary, on="method")
)

validation_results.to_csv(OUTPUT_DIR / "masked_value_validation_results.csv", index=False)
numeric_distribution.to_csv(OUTPUT_DIR / "numeric_distribution_comparison.csv", index=False)
categorical_distribution.to_csv(OUTPUT_DIR / "categorical_distribution_comparison.csv", index=False)
relationship_results.to_csv(OUTPUT_DIR / "numeric_correlation_preservation.csv", index=False)
method_summary.to_csv(OUTPUT_DIR / "imputation_method_summary.csv", index=False)

best_numeric = method_summary.sort_values("rmse").iloc[0]
best_categorical = method_summary.sort_values("accuracy", ascending=False).iloc[0]
lines = [
    "Imputation method evaluation",
    "",
    "Methods compared:",
    "- median_mode: median for numeric features and mode for categorical features.",
    "- knn: KNN imputation after numeric scaling and ordinal encoding of categorical features.",
    "- lightgbm: supervised feature-by-feature imputation using the feature with missing values as the target.",
    "",
    "Masked-value validation uses sklearn StratifiedKFold with 5 folds. In each fold, known observed values are hidden, imputed, and compared to the true hidden values.",
    "All imputation methods replace only NaN values. Observed non-missing values are kept unchanged.",
    f"Best average numeric RMSE: {best_numeric['method']} ({best_numeric['rmse']:.3f}).",
    f"Best average categorical accuracy: {best_categorical['method']} ({best_categorical['accuracy']:.3f}).",
    "",
    "Outputs:",
    "- masked_value_validation_results.csv",
    "- numeric_distribution_comparison.csv",
    "- categorical_distribution_comparison.csv",
    "- numeric_correlation_preservation.csv",
    "- imputation_method_summary.csv",
    "",
    "Per-feature validation results:",
    validation_results.to_string(index=False),
]
(OUTPUT_DIR / "imputation_evaluation_notes.txt").write_text("\n".join(lines), encoding="utf-8")
