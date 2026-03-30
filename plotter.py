import numpy as np
import matplotlib.pyplot as plt


class ResultsPlotter:

    # ---------------------------------------------------------
    # FIGURE 1: MAIN COMPARISON (R2 + MAE, WITH vs WITHOUT)
    # ---------------------------------------------------------
    @staticmethod
    def plot_cycle_comparison(cv_results):

        models = ["linear", "rf", "xgb", "cat"]
        model_labels = ["Linear", "Random Forest", "XGBoost", "CatBoost"]

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
        fig.suptitle("Impact of Menstrual Cycle Phase on Model Performance", fontsize=16)

        def annotate(ax, bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8)

        # R2 Attention
        ax = axs[0, 0]
        b1 = ax.bar(x - width/2, att_r2_without, width, label="Without Cycle")
        b2 = ax.bar(x + width/2, att_r2_with, width, label="With Cycle")
        ax.set_title("R² - Attention")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("R² Score")
        ax.legend()
        annotate(ax, b1)
        annotate(ax, b2)

        # R2 Memory
        ax = axs[0, 1]
        b1 = ax.bar(x - width/2, mem_r2_without, width)
        b2 = ax.bar(x + width/2, mem_r2_with, width)
        ax.set_title("R² - Memory")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("R² Score")
        annotate(ax, b1)
        annotate(ax, b2)

        # MAE Attention
        ax = axs[1, 0]
        b1 = ax.bar(x - width/2, att_mae_without, width)
        b2 = ax.bar(x + width/2, att_mae_with, width)
        ax.set_title("MAE - Attention")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("MAE")
        annotate(ax, b1)
        annotate(ax, b2)

        # MAE Memory
        ax = axs[1, 1]
        b1 = ax.bar(x - width/2, mem_mae_without, width)
        b2 = ax.bar(x + width/2, mem_mae_with, width)
        ax.set_title("MAE - Memory")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("MAE")
        annotate(ax, b1)
        annotate(ax, b2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("figure1_cycle_comparison.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FIGURE 3: R2 COMPARISON CLEAN
    # ---------------------------------------------------------
    @staticmethod
    def plot_r2_comparison(cv_results):

        models = ["linear", "rf", "xgb", "cat"]
        labels = ["Linear", "Random Forest", "XGBoost", "CatBoost"]

        r2_with = [cv_results[m]["Attention"]["r2_with_mean"] for m in models]
        r2_without = [cv_results[m]["Attention"]["r2_without_mean"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(10, 6))

        b1 = plt.bar(x - width/2, r2_without, width, label="Without Cycle")
        b2 = plt.bar(x + width/2, r2_with, width, label="With Cycle")

        plt.xticks(x, labels)
        plt.ylabel("R² Score")
        plt.title("Model Comparison (R² Scores)")
        plt.legend()

        for bar in b1 + b2:
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height(),
                     f"{bar.get_height():.2f}",
                     ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("figure3_r2_comparison.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FIGURE 4: MAE REDUCTION
    # ---------------------------------------------------------
    @staticmethod
    def plot_mae_reduction(cv_results):

        models = ["linear", "rf", "xgb", "cat"]
        labels = ["Linear", "Random Forest", "XGBoost", "CatBoost"]

        mae_with = [cv_results[m]["Attention"]["mae_with_mean"] for m in models]
        mae_without = [cv_results[m]["Attention"]["mae_without_mean"] for m in models]

        reduction = [wo - w for wo, w in zip(mae_without, mae_with)]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, reduction)

        plt.ylabel("MAE Reduction")
        plt.title("Reduction in Prediction Error")

        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height(),
                     f"{bar.get_height():.2f}",
                     ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("figure4_mae_reduction.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FIGURE 5: FEATURE IMPORTANCE
    # ---------------------------------------------------------
    @staticmethod
    def plot_feature_importance(model_runner, X_train, y_train):

        model = model_runner.build_model("rf", model_runner.preprocess_with)
        model.fit(X_train, y_train)

        feature_names = model.named_steps["prep"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.title("Feature Importance (Top 10 Features)")

        plt.tight_layout()
        plt.savefig("figure5_feature_importance.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FIGURE 6: CYCLE PHASE VS PERFORMANCE
    # ---------------------------------------------------------
    @staticmethod
    def plot_cycle_phase_performance(df):

        phase_means = df.groupby("cycle_phase")[["attention_score", "memory_score"]].mean()

        phases = phase_means.index

        plt.figure(figsize=(10, 6))

        plt.plot(phases, phase_means["attention_score"], marker='o', label="Attention")
        plt.plot(phases, phase_means["memory_score"], marker='o', label="Memory")

        plt.xlabel("Cycle Phase")
        plt.ylabel("Score")
        plt.title("Cognitive Performance Across Cycle Phases")
        plt.legend()

        plt.tight_layout()
        plt.savefig("figure6_cycle_phase.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FIGURE 7: PIPELINE DIAGRAM
    # ---------------------------------------------------------
    @staticmethod
    def plot_pipeline():

        steps = [
            "Data",
            "Preprocessing",
            "Feature Engineering",
            "Model",
            "Evaluation"
        ]

        x = np.arange(len(steps))
        y = [1]*len(steps)

        plt.figure(figsize=(12, 3))
        plt.scatter(x, y)

        for i, step in enumerate(steps):
            plt.text(x[i], y[i]+0.02, step, ha='center')

        for i in range(len(steps)-1):
            plt.arrow(x[i], y[i], x[i+1]-x[i]-0.1, 0,
                      head_width=0.02, length_includes_head=True)

        plt.xticks([])
        plt.yticks([])
        plt.title("Machine Learning Pipeline")

        plt.tight_layout()
        plt.savefig("figure7_pipeline.png", dpi=300)
        plt.show()