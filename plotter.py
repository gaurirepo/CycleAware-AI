# plotter.py

import numpy as np
import matplotlib.pyplot as plt


class ResultsPlotter:

    @staticmethod
    def plot_model_comparison(all_results):

        models = list(all_results["Attention"].keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Attention R2
        axes[0, 0].bar(
            models,
            [all_results["Attention"][m]["r2_with"] for m in models]
        )
        axes[0, 0].set_title("Attention - R2 (With Cycle)")
        axes[0, 0].set_ylabel("R2")

        # Attention MAE
        axes[0, 1].bar(
            models,
            [all_results["Attention"][m]["mae_with"] for m in models]
        )
        axes[0, 1].set_title("Attention - MAE (With Cycle)")
        axes[0, 1].set_ylabel("MAE")

        # Memory R2
        axes[1, 0].bar(
            models,
            [all_results["Memory"][m]["r2_with"] for m in models]
        )
        axes[1, 0].set_title("Memory - R2 (With Cycle)")
        axes[1, 0].set_ylabel("R2")

        # Memory MAE
        axes[1, 1].bar(
            models,
            [all_results["Memory"][m]["mae_with"] for m in models]
        )
        axes[1, 1].set_title("Memory - MAE (With Cycle)")
        axes[1, 1].set_ylabel("MAE")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # Plot comparison graph WITH vs WITHOUT cycle (4 models)
    # ---------------------------------------------------------

    @staticmethod
    def plot_cycle_comparison(cv_results):

        models = ["linear", "rf", "xgb", "cat"]
        model_labels = ["Linear", "RF", "XGB", "CAT"]

        # Extract metrics
        att_r2_with = [cv_results[m]["Attention"]["r2_with_mean"] for m in models]
        att_r2_without = [cv_results[m]["Attention"]["r2_without_mean"] for m in models]

        mem_r2_with = [cv_results[m]["Memory"]["r2_with_mean"] for m in models]
        mem_r2_without = [cv_results[m]["Memory"]["r2_without_mean"] for m in models]

        att_mae_with = [cv_results[m]["Attention"]["mae_with_mean"] for m in models]
        att_mae_without = [cv_results[m]["Attention"]["mae_without_mean"] for m in models]

        mem_mae_with = [cv_results[m]["Memory"]["mae_with_mean"] for m in models]
        mem_mae_without = [cv_results[m]["Memory"]["mae_without_mean"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Cycle Phase Impact on Prediction Performance", fontsize=18)

        # Helper function to annotate bars
        def annotate_bars(ax, bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        # Helper function to highlight best model
        def highlight_best(ax, values, is_r2=True):
            if is_r2:
                best_index = np.argmax(values)
                offset = 0.02
            else:
                best_index = np.argmin(values)
                offset = 0.2

            ax.text(
                best_index,
                values[best_index] + offset,
                "⭐ Best",
                ha="center",
                fontsize=11,
                fontweight="bold"
            )

        # -------- R2 Attention --------
        ax = axs[0, 0]
        bars1 = ax.bar(x - width/2, att_r2_without, width, label="Without Cycle")
        bars2 = ax.bar(x + width/2, att_r2_with, width, label="With Cycle")
        ax.set_title("R² - Attention")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        annotate_bars(ax, bars1)
        annotate_bars(ax, bars2)
        highlight_best(ax, att_r2_with, is_r2=True)

        # -------- R2 Memory --------
        ax = axs[0, 1]
        bars1 = ax.bar(x - width/2, mem_r2_without, width, label="Without Cycle")
        bars2 = ax.bar(x + width/2, mem_r2_with, width, label="With Cycle")
        ax.set_title("R² - Memory")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        annotate_bars(ax, bars1)
        annotate_bars(ax, bars2)
        highlight_best(ax, mem_r2_with, is_r2=True)

        # -------- MAE Attention --------
        ax = axs[1, 0]
        bars1 = ax.bar(x - width/2, att_mae_without, width, label="Without Cycle")
        bars2 = ax.bar(x + width/2, att_mae_with, width, label="With Cycle")
        ax.set_title("MAE - Attention")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        annotate_bars(ax, bars1)
        annotate_bars(ax, bars2)
        highlight_best(ax, att_mae_with, is_r2=False)

        # -------- MAE Memory --------
        ax = axs[1, 1]
        bars1 = ax.bar(x - width/2, mem_mae_without, width, label="Without Cycle")
        bars2 = ax.bar(x + width/2, mem_mae_with, width, label="With Cycle")
        ax.set_title("MAE - Memory")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        annotate_bars(ax, bars1)
        annotate_bars(ax, bars2)
        highlight_best(ax, mem_mae_with, is_r2=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()