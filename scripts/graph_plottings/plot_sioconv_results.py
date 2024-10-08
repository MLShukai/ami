"""Expected log structure.

sioconv_tensorboard_logs
├── large
│   └── observation_loss
│       ├── multiruns_2024-09-16_15-27-32_0_tensorboard_forward_dynamics.csv
│       ├── multiruns_2024-09-17_08-10-13_0_tensorboard_forward_dynamics.csv
│       └── multiruns_2024-09-18_08-01-48_0_tensorboard_forward_dynamics.csv
├── permutation
│   └── observation_loss
│       └── multiruns_2024-09-18_21-21-58_0_tensorboard_forward_dynamics.csv
├── small
│   └── observation_loss
│       ├── multiruns_2024-09-18_07-56-17_0_tensorboard_forward_dynamics.csv
│       ├── multiruns_2024-09-18_07-56-17_1_tensorboard_forward_dynamics.csv
│       └── multiruns_2024-09-18_07-56-17_2_tensorboard_forward_dynamics.csv
└── with_ijepa
    └── observation_loss
        ├── multiruns_2024-09-16_15-28-35_0_tensorboard_forward_dynamics.csv
        ├── runs_2024-09-17_10-28-17_tensorboard_forward_dynamics.csv
        └── runs_2024-09-18_03-37-03_tensorboard_forward_dynamics.csv
"""
from pathlib import Path

import japanize_matplotlib  # pip install japanize-matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

MODEL_TYPES = ["large", "small", "permutation", "with_ijepa"]
FIG_SIZE = (6, 6)
MAX_UPTIME = 24  # 最大経過時間（時間）

COLORS = {
    "large": "#0000FF",  # blue
    "small": "#008000",  # green
    "permutation": "#FFA500",  # orange
    "with_ijepa": "#FF0000",  # red
}

MODEL_NAMES = {
    "large": "ラージモデル",
    "small": "スモールモデル",
    "permutation": "系列破壊（ラージモデル）",
    "with_ijepa": "I-JEPAと同時学習（ラージモデル）",
}


def read_csv_files(base_path: Path, model_type: str) -> list[pd.DataFrame]:
    path = base_path / model_type / "observation_loss"
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

    for model_type in MODEL_TYPES:
        dfs = read_csv_files(base_path, model_type)
        if not dfs:
            print(f"No data found for {model_type}")
            continue
        df = process_data(dfs)
        color = COLORS[model_type]

        # 元のデータをプロット
        x = np.linspace(0, MAX_UPTIME, len(df["Step"]))
        plt.plot(x, df["Mean"], color=color, alpha=0.1, linewidth=1)
        plt.fill_between(x, df["Mean"] - df["Std"], df["Mean"] + df["Std"], color=color, alpha=0.05)

        # EMAを計算してプロット
        ema = calculate_ema(df["Mean"], span=ema_span)
        plt.plot(x, ema, color=color, label=f"{MODEL_NAMES[model_type]}", linewidth=2, alpha=0.8)

    plt.title(f"{title}", fontsize=18)
    plt.xlabel("経過時間（時間）")
    plt.ylabel(y_label)
    plt.ylim(y_lims)
    plt.xlim(0, MAX_UPTIME)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(base_path / f"{title}.png")


def main() -> None:
    base_path = PROJECT_ROOT / "data/sioconv_tensorboard_logs"
    ema_span = 10

    plot_metric(base_path, "SIOConvの観測損失", "損失", y_lims=(0.0, 2.5), ema_span=ema_span)


if __name__ == "__main__":
    main()
