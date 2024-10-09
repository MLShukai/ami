import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

FIG_SIZE = (7, 4)
COLORS = ["#008000", "#0000FF", "#FFA500", "#FF0000"]
PATTERNS = ["Simple World (Small)", "Simple World (Large)", "Noisy World (Small)", "Noisy World (Large)"]
PATTERN_LABELS = ["シンプルなワールド，スモールモデル", "シンプルなワールド，ラージモデル", "ノイズ付きワールド，スモールモデル", "ノイズ付きワールド，ラージモデル"]
ACTION_COLUMNS = [f"Act{i}" for i in range(1, 9)]
X_TICK_LABELS = ["静止", "前後移動\nのみ", "左右移動\nのみ", "左右回転\nのみ", "左右回転\nなし", "左右移動\nなし", "前後移動\nなし", "前後左右\n移動回転"]
BAR_WIDTH = 0.2


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df["ActualSum"] = df[ACTION_COLUMNS].sum(axis=1)

    for col in ACTION_COLUMNS:
        df[f"{col}_prop"] = df[col] / df["ActualSum"]

    return df


def plot_histogram(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x = np.arange(len(ACTION_COLUMNS))

    for i, pattern in enumerate(PATTERNS):
        values = df[df["Pattern"] == pattern][[f"{act}_prop" for act in ACTION_COLUMNS]].values[0] * 100
        ax.bar(x + i * BAR_WIDTH, values, BAR_WIDTH, label=PATTERN_LABELS[i], color=COLORS[i])

        # for j, v in enumerate(values):
        #     if v > 0:
        #         ax.text(x[j] + i*width, v, f'{v:.1f}%', ha='center', va='bottom', rotation=90, fontsize=8)

    ax.set_ylabel("割合 (%)")
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x + BAR_WIDTH * 1.5)
    ax.set_xticklabels(X_TICK_LABELS)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "data" / "ami_action_convergence_histogram.png")
    plt.close()


def main() -> None:
    file_path = PROJECT_ROOT / "data" / "AMI行動収束パターン集計_VConf24.csv"
    df = pd.read_csv(file_path)
    processed_df = process_data(df)
    plot_histogram(processed_df, "行動収束パターン分布")


if __name__ == "__main__":
    main()
