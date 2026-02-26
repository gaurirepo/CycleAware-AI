import numpy as np
from scipy.stats import ttest_rel


class CycleAwareStats:

    def __init__(self, y_true, pred_with, pred_without):

        # Convert everything to NumPy arrays (IMPORTANT FIX)
        y_true = np.array(y_true)
        pred_with = np.array(pred_with)
        pred_without = np.array(pred_without)

        self.err_w = np.abs(y_true - pred_with)
        self.err_wo = np.abs(y_true - pred_without)

    def cohens_d(self):
        diff = self.err_wo - self.err_w
        return np.mean(diff) / np.std(diff)

    def confidence_interval(self, n_boot=2000):

        diffs = []
        n = len(self.err_w)

        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            diffs.append(
                np.mean(self.err_wo[idx]) - np.mean(self.err_w[idx])
            )

        return np.percentile(diffs, [2.5, 97.5])

    def full_report(self, label):

        delta = np.mean(self.err_wo) - np.mean(self.err_w)
        t_res = ttest_rel(self.err_w, self.err_wo)
        d = self.cohens_d()
        ci = self.confidence_interval()

        print(f"\n===== {label} Statistical Validation =====")
        print(f"ΔMAE = {delta:.3f}")
        print(f"p-value = {t_res.pvalue:.4f}")
        print(f"Cohen's d = {d:.3f}")
        print(f"95% CI = ({ci[0]:.3f}, {ci[1]:.3f})")