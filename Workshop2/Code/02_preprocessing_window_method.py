from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "Data" / "Raw"
output_path = base_path / "Output" / "Preprocessing"
plot_path = base_path / "Plots"
output_path.mkdir(parents=True, exist_ok=True)
plot_path.mkdir(parents=True, exist_ok=True)

train_data = pd.read_csv(data_path / "DailyDelhiClimateTrain.csv", parse_dates=["date"])
test_data = pd.read_csv(data_path / "DailyDelhiClimateTest.csv", parse_dates=["date"])

signal = "wind_speed"
window_sizes = [7, 14, 30]
zoom_start = "2015-04-01"
zoom_end = "2015-06-30"

train_processed = train_data.copy()
test_processed = test_data.copy()

for window_size in window_sizes:
    train_processed[f"{signal}_ma_{window_size}_day"] = train_processed[signal].rolling(
        window_size, center=True
    ).mean()
    test_processed[f"{signal}_ma_{window_size}_day"] = test_processed[signal].rolling(
        window_size, center=True
    ).mean()

train_processed.to_csv(output_path / "train_wind_speed_moving_average.csv", index=False)
test_processed.to_csv(output_path / "test_wind_speed_moving_average.csv", index=False)

metric_rows = []
for split_name, data in [("train", train_processed), ("test", test_processed)]:
    original_daily_change = data[signal].diff().abs().mean()

    for window_size in window_sizes:
        column = f"{signal}_ma_{window_size}_day"
        valid_data = data[[signal, column]].dropna()
        smoothed_daily_change = valid_data[column].diff().abs().mean()
        residual = valid_data[signal] - valid_data[column]

        metric_rows.append(
            {
                "split": split_name,
                "window_days": window_size,
                "valid_points": len(valid_data),
                "original_mean_abs_daily_change": original_daily_change,
                "smoothed_mean_abs_daily_change": smoothed_daily_change,
                "daily_change_reduction_percent": (1 - smoothed_daily_change / original_daily_change) * 100,
                "residual_std": residual.std(),
                "max_absolute_residual": residual.abs().max(),
            }
        )

pd.DataFrame(metric_rows).to_csv(output_path / "moving_average_metrics.csv", index=False)

fig, axis = plt.subplots(figsize=(12, 5))
axis.plot(train_processed["date"], train_processed[signal], label="original", linewidth=0.7, alpha=0.45)
for window_size in window_sizes:
    axis.plot(
        train_processed["date"],
        train_processed[f"{signal}_ma_{window_size}_day"],
        label=f"{window_size}-day moving average",
        linewidth=1.8,
    )
axis.set_title("Training wind speed before and after moving average preprocessing")
axis.set_xlabel("date")
axis.set_ylabel(signal)
axis.grid(True, alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(plot_path / "01_train_wind_speed_all_windows_full.png", dpi=180)
plt.close(fig)

zoom_data = train_processed[
    (train_processed["date"] >= zoom_start) & (train_processed["date"] <= zoom_end)
]

fig, axis = plt.subplots(figsize=(12, 5))
axis.plot(zoom_data["date"], zoom_data[signal], marker="o", markersize=3, label="original", alpha=0.6)
for window_size in window_sizes:
    axis.plot(
        zoom_data["date"],
        zoom_data[f"{signal}_ma_{window_size}_day"],
        label=f"{window_size}-day moving average",
        linewidth=1.8,
    )
axis.set_title("Training wind speed zoom: moving average windows")
axis.set_xlabel("date")
axis.set_ylabel(signal)
axis.grid(True, alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(plot_path / "02_train_wind_speed_all_windows_zoom.png", dpi=180)
plt.close(fig)

fig, axis = plt.subplots(figsize=(12, 5))
axis.axhline(0, color="black", linewidth=1)
for window_size in window_sizes:
    residual = train_processed[signal] - train_processed[f"{signal}_ma_{window_size}_day"]
    axis.plot(train_processed["date"], residual, label=f"{window_size}-day residual", linewidth=0.8)
axis.set_title("Training wind speed residuals for moving average windows")
axis.set_xlabel("date")
axis.set_ylabel("original minus moving average")
axis.grid(True, alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(plot_path / "03_train_wind_speed_residuals_all_windows.png", dpi=180)
plt.close(fig)

fig, axis = plt.subplots(figsize=(12, 4.5))
axis.plot(test_processed["date"], test_processed[signal], marker="o", markersize=3, label="original", alpha=0.65)
for window_size in window_sizes:
    axis.plot(
        test_processed["date"],
        test_processed[f"{signal}_ma_{window_size}_day"],
        label=f"{window_size}-day moving average",
        linewidth=1.8,
    )
axis.set_title("Test wind speed before and after moving average preprocessing")
axis.set_xlabel("date")
axis.set_ylabel(signal)
axis.grid(True, alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(plot_path / "04_test_wind_speed_all_windows.png", dpi=180)
plt.close(fig)

print(f"Saved preprocessing outputs to: {output_path}")
