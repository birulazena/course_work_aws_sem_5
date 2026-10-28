import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = "benchmark_results.csv"
    if not os.path.exists(csv_file):
        print(f"Ошибка: файл {csv_file} не найден. Сначала запустите C++ бенчмарк.")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    modes = df["Mode"].unique()
    sizes = sorted(df["MatrixSize"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Цветовая палитра и маркеры для 5 режимов
    style_map = {
        "Single-Core (SMT OFF)": {"color": "#1f77b4", "marker": "o", "ls": "--"},
        "Single-Core (SMT ON)":  {"color": "#aec7e8", "marker": "s", "ls": ":"},
        "Multi-Core (SMT OFF)":  {"color": "#2ca02c", "marker": "^", "ls": "-."},
        "Multi-Core (SMT ON)":   {"color": "#1b9e77", "marker": "D", "ls": "-"},
        "iGPU Compute (OpenCL)": {"color": "#d95f02", "marker": "*", "ls": "-"}
    }

    # 1. График производительности (GFLOPS)
    for mode in modes:
        sub_df = df[df["Mode"] == mode]
        style = style_map.get(mode, {"color": "gray", "marker": "x", "ls": "-"})
        ax1.plot(
            sub_df["MatrixSize"],
            sub_df["GFLOPS"],
            label=mode,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2,
            markersize=6
        )

    ax1.set_title("Производительность вычислений (GFLOPS)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Размер матрицы N x N", fontsize=11)
    ax1.set_ylabel("GFLOPS (выше — лучше)", fontsize=11)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # 2. График времени выполнения (мс, лог-шкала)
    for mode in modes:
        sub_df = df[df["Mode"] == mode]
        style = style_map.get(mode, {"color": "gray", "marker": "x", "ls": "-"})
        ax2.plot(
            sub_df["MatrixSize"],
            sub_df["Time_ms"],
            label=mode,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2,
            markersize=6
        )

    ax2.set_title("Время выполнения (Log Scale)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Размер матрицы N x N", fontsize=11)
    ax2.set_ylabel("Время (мс, ниже — лучше)", fontsize=11)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(sizes)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

    plt.tight_layout()
    output_png = "benchmark_results.png"
    plt.savefig(output_png, dpi=300)
    print(f"График успешно сохранен в: {output_png}")

if __name__ == "__main__":
    main()