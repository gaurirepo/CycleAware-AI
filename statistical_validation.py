# statistical_validation.py

import numpy as np
from scipy.stats import ttest_rel

class CycleAwareStats:

    def __init__(self, y_att, pred_att_w, pred_att_wo,
                 y_mem, pred_mem_w, pred_mem_wo,
                 df_test):

        self.err_att_w  = np.abs(y_att - pred_att_w)
        self.err_att_wo = np.abs(y_att - pred_att_wo)
        self.err_mem_w  = np.abs(y_mem - pred_mem_w)
        self.err_mem_wo = np.abs(y_mem - pred_mem_wo)

        self.df_test = df_test.reset_index(drop=True)

    @staticmethod
    def cohens_d(x, y):
        return (np.mean(y - x)) / np.std(y - x)

    @staticmethod
    def confidence_interval(err_w, err_wo, n_boot=2000):
        diffs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(err_w), len(err_w), replace=True)
            diffs.append(np.mean(err_wo[idx]) - np.mean(err_w[idx]))
        return np.percentile(diffs, [2.5, 97.5])

    def phase_wise_error(self):
        results = []
        for phase in self.df_test["cycle_phase"].unique():
            mask = self.df_test["cycle_phase"] == phase
            results.append({
                "Phase": phase,
                "Attention ΔMAE": round(np.mean(self.err_att_wo[mask]) -
                                        np.mean(self.err_att_w[mask]), 2),
                "Memory ΔMAE": round(np.mean(self.err_mem_wo[mask]) -
                                     np.mean(self.err_mem_w[mask]), 2)
            })
        return results

    def full_report(self):
        t_att = ttest_rel(self.err_att_w, self.err_att_wo)
        t_mem = ttest_rel(self.err_mem_w, self.err_mem_wo)

        d_att = self.cohens_d(self.err_att_w, self.err_att_wo)
        d_mem = self.cohens_d(self.err_mem_w, self.err_mem_wo)

        ci_att = self.confidence_interval(self.err_att_w, self.err_att_wo)
        ci_mem = self.confidence_interval(self.err_mem_w, self.err_mem_wo)

        print("\n===== STATISTICAL VALIDATION =====")
        print(f"Attention ΔMAE = {np.mean(self.err_att_wo) - np.mean(self.err_att_w):.2f}")
        print(f"p = {t_att.pvalue:.4f}, Cohen’s d = {d_att:.2f}")
        print(f"95% CI = ({ci_att[0]:.2f}, {ci_att[1]:.2f})")

        print(f"\nMemory ΔMAE = {np.mean(self.err_mem_wo) - np.mean(self.err_mem_w):.2f}")
        print(f"p = {t_mem.pvalue:.4f}, Cohen’s d = {d_mem:.2f}")
        print(f"95% CI = ({ci_mem[0]:.2f}, {ci_mem[1]:.2f})")

        print("\n--- Phase-wise Error Reduction ---")
        for r in self.phase_wise_error():
            print(f"{r['Phase']}: Attention ↓ {r['Attention ΔMAE']} | Memory ↓ {r['Memory ΔMAE']}")