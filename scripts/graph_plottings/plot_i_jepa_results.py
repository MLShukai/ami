"""Expected log structure.

i_jepa_tensorboard_logs
├── large
│   └── target_validation_reconstruction
│       ├── multiruns_2024-09-17_11-59-58_0_tensorboard_i_jepa_latent_visualization_target.csv
│       ├── multiruns_2024-09-17_11-59-58_1_tensorboard_i_jepa_latent_visualization_target.csv
│       └── ... .csv
└── small
    └── target_validation_reconstruction
        ├── multiruns_2024-09-13_16-33-46_0_tensorboard_i_jepa_latent_visualization_target.csv
        ├── multiruns_2024-09-13_16-33-46_1_tensorboard_i_jepa_latent_visualization_target.csv
        └── ... .csv
"""
from pathlib import Path

import japanize_matplotlib  # pip install japanize-matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

MODEL_SIZES = ["small", "large"]
FIG_SIZE = (4, 4)
MAX_UPTIME = 24  # 最大経過時間（時間）

COLORS = {
    "small": "#008000",  # green
    "large": "#0000FF",  # blue
}

SIZE_NAMES = {"small": "スモールモデル", "large": "ラージモデル"}


def read_csv_files(base_path: Path, size: str) -> list[pd.DataFrame]:
    path = base_path / size / "target_validation_reconstruction"
    return [pd.read_csv(file) for file in path.glob("*.csv")]


def process_data(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # 全てのDataFrameから 'Value', 'Step' 列を抽出
    values = [df["Value"].values for df in dfs]
    steps = [df["Step"].values for df in dfs]

    # 最小の長さに合わせて切り詰める
    min_length = min(len(v) for v in values)
    values = [v[:min_length] for v in values]
    steps = [s[:min_length] for s in steps]

    # NumPyの配列に変換
    values_array = np.array(values)
    steps_array = np.array(steps)

    # 行ごとの平均と標準偏差を計算
    mean = np.mean(values_array, axis=0)
    std = np.std(values_array, axis=0)
    steps_array = np.mean(steps_array, axis=0)

    return pd.DataFrame({"Step": steps_array, "Mean": mean, "Std": std})


def calculate_ema(data: pd.Series, span: int = 20) -> pd.Series:
    return data.ewm(span=span, adjust=False).mean()


def plot_metric(base_path: Path, title: str, y_label: str, y_lims: tuple[float, float], ema_span: int = 20) -> None:
    plt.figure(figsize=FIG_SIZE)

    for size in MODEL_SIZES:
        dfs = read_csv_files(base_path, size)
        df = process_data(dfs)
        color = COLORS[size]

        # 元のデータをプロット
        x = np.linspace(0, MAX_UPTIME, len(df["Step"]))
        plt.plot(x, df["Mean"], color=color, alpha=0.2, linewidth=1)
        plt.fill_between(x, df["Mean"] - df["Std"], df["Mean"] + df["Std"], color=color, alpha=0.1)

        # EMAを計算してプロット
        ema = calculate_ema(df["Mean"], span=ema_span)
        plt.plot(x, ema, color=color, label=f"{SIZE_NAMES[size]}", linewidth=2, alpha=0.8)

    plt.title(f"{title}", fontsize=18)
    plt.xlabel("経過時間（時間）", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.ylim(y_lims)
    plt.xlim(0, MAX_UPTIME)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout(pad=0.1)
    plt.savefig(base_path / f"{title}.svg")


def main() -> None:
    base_path = PROJECT_ROOT / "data/i_jepa_tensorboard_logs"
    ema_span = 10

    plot_metric(base_path, "I-JEPAの復元誤差", "誤差", y_lims=(0.0, 0.6), ema_span=ema_span)


if __name__ == "__main__":
    main()
