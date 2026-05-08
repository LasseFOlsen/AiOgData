from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "Data" / "Raw"
output_path = base_path / "Output" / "DataVisualisation"
plot_path = base_path / "Plots"
output_path.mkdir(parents=True, exist_ok=True)
plot_path.mkdir(parents=True, exist_ok=True)

train_data = pd.read_csv(data_path / "DailyDelhiClimateTrain.csv", parse_dates=["date"])
test_data = pd.read_csv(data_path / "DailyDelhiClimateTest.csv", parse_dates=["date"])

variables = ["meantemp", "humidity", "wind_speed", "meanpressure"]
screening_window = 14
zoom_start = "2015-04-01"
zoom_end = "2015-06-30"

# Grundinfo bruges til at dokumentere datasættet i rapporten.
dataset_summary = pd.DataFrame(
    [
        {
            "split": "train",
            "rows": len(train_data),
            "start_date": train_data["date"].min().date(),
            "end_date": train_data["date"].max().date(),
            "missing_values": int(train_data.isna().sum().sum()),
            "most_common_interval_days": int(train_data["date"].diff().dt.days.dropna().mode().iloc[0]),
        },
        {
            "split": "test",
            "rows": len(test_data),
            "start_date": test_data["date"].min().date(),
            "end_date": test_data["date"].max().date(),
            "missing_values": int(test_data.isna().sum().sum()),
            "most_common_interval_days": int(test_data["date"].diff().dt.days.dropna().mode().iloc[0]),
        },
    ]
)
dataset_summary.to_csv(output_path / "dataset_summary.csv", index=False)

screening_rows = []
for variable in variables:
    rolling_mean = train_data[variable].rolling(screening_window, center=True).mean()
    residual = train_data[variable] - rolling_mean
    daily_change = train_data[variable].diff()

    screening_rows.append(
        {
            "variable": variable,
            "original_std": train_data[variable].std(),
            "mean_absolute_daily_change": daily_change.abs().mean(),
            "daily_change_to_std_ratio": daily_change.abs().mean() / train_data[variable].std(),
            "residual_std_14_day": residual.std(),
            "residual_to_std_ratio": residual.std() / train_data[variable].std(),
            "minimum": train_data[variable].min(),
            "maximum": train_data[variable].max(),
        }
    )

pd.DataFrame(screening_rows).to_csv(output_path / "time_series_screening_metrics.csv", index=False)

fig, axes = plt.subplots(len(variables), 1, figsize=(12, 9), sharex=True)
for axis, variable in zip(axes, variables):
    rolling_mean = train_data[variable].rolling(screening_window, center=True).mean()
    axis.plot(train_data["date"], train_data[variable], label="original", linewidth=0.7, alpha=0.6)
    axis.plot(train_data["date"], rolling_mean, label="14-day rolling mean", linewidth=1.5)
    axis.set_ylabel(variable)
    axis.grid(True, alpha=0.3)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("date")
fig.suptitle("Training data: numeric time-series screening")
fig.tight_layout()
fig.savefig(plot_path / "01_all_numeric_time_series_screening.png", dpi=180)
plt.close(fig)

zoom_data = train_data[(train_data["date"] >= zoom_start) & (train_data["date"] <= zoom_end)]

fig, axes = plt.subplots(len(variables), 1, figsize=(12, 9), sharex=True)
for axis, variable in zip(axes, variables):
    rolling_mean = zoom_data[variable].rolling(screening_window, center=True).mean()
    axis.plot(zoom_data["date"], zoom_data[variable], marker="o", markersize=2.5, linewidth=0.8, label="original")
    axis.plot(zoom_data["date"], rolling_mean, linewidth=1.6, label="14-day rolling mean")
    axis.set_ylabel(variable)
    axis.grid(True, alpha=0.3)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("date")
fig.suptitle("Training data: 90-day zoom for numeric time-series screening")
fig.tight_layout()
fig.savefig(plot_path / "02_all_numeric_time_series_zoom_screening.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(len(variables), 1, figsize=(12, 9), sharex=True)
for axis, variable in zip(axes, variables):
    rolling_mean = train_data[variable].rolling(screening_window, center=True).mean()
    residual = train_data[variable] - rolling_mean
    axis.axhline(0, color="black", linewidth=0.8)
    axis.plot(train_data["date"], residual, linewidth=0.7)
    axis.set_ylabel(variable)
    axis.grid(True, alpha=0.3)
axes[-1].set_xlabel("date")
fig.suptitle("Training data: residuals from 14-day rolling mean")
fig.tight_layout()
fig.savefig(plot_path / "03_all_numeric_time_series_residuals.png", dpi=180)
plt.close(fig)

print(f"Saved data visualisation outputs to: {output_path}")
