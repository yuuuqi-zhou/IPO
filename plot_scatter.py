import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 路径
# =========================
CSV_PATH = "pred_output.csv"
OUT_DIR = "plots"

# =========================
# 真实列名（来自你的 Excel / pred_output.csv）
# =========================
COL_RETURN = "相对发行价涨跌幅"
COL_VOL = "成交量"
COL_MS = "marginstress10"

# 预测列名（来自预测脚本）
PRED_RETURN = "pred_return"
PRED_VOL = "pred_vol"
PRED_MS_PROBA = "pred_ms_proba"

# 分类阈值（你截图是 0.5）
THRESHOLD = 0.5

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

# =========================
# 回归散点：带 45° 线
# =========================
def save_scatter(y_true, y_pred, title, xlabel, ylabel, outpath, diagonal=True):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    if diagonal:
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# =========================
# MarginStress：阈值 + 混淆矩阵 + 关键比率（文本 + 图）
# =========================
def confusion_report(y_true, y_proba, threshold=0.5):
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_proba)
    y_true = y_true[mask].astype(int)
    y_proba = y_proba[mask]
    y_pred = (y_proba >= threshold).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    # 关键风控比率（和你截图一致的两条）
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else np.nan  # 漏放率：FN / 实际风险
    block_rate = TP / (TP + FP) if (TP + FP) > 0 else np.nan # 拦截命中率：TP / 预测高风险（也就是precision）

    # 常见指标（可选，但建议输出）
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan
    precision = block_rate
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not np.nan and recall is not np.nan and (precision + recall) > 0) else np.nan

    report = {
        "threshold": threshold,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "miss_rate": miss_rate,
        "block_rate": block_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return report


def save_confusion_text(report, outpath):
    txt = []
    txt.append(f"Threshold = {report['threshold']}\n")
    txt.append("Confusion Matrix:\n")
    txt.append(f"TP (correctly blocked) : {report['TP']}\n")
    txt.append(f"FP (overly conservative): {report['FP']}\n")
    txt.append(f"FN (missed risk)        : {report['FN']}\n")
    txt.append(f"TN (safe approved)      : {report['TN']}\n\n")

    txt.append("Key rates:\n")
    txt.append(f"Miss rate (FN / actual breaks): {report['miss_rate']*100:.2f}%\n")
    txt.append(f"Block rate (TP / predicted high risk): {report['block_rate']*100:.2f}%\n\n")

    txt.append("Common metrics:\n")
    txt.append(f"Accuracy : {report['accuracy']*100:.2f}%\n")
    txt.append(f"Precision: {report['precision']*100:.2f}%\n")
    txt.append(f"Recall   : {report['recall']*100:.2f}%\n")
    txt.append(f"F1       : {report['f1']*100:.2f}%\n")

    with open(outpath, "w", encoding="utf-8") as f:
        f.writelines(txt)


def save_confusion_plot(report, outpath):
    cm = np.array([[report["TN"], report["FP"]],
                   [report["FN"], report["TP"]]], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # 基础 heatmap：仍然表示“数量”
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title(f"Confusion Matrix (threshold={report['threshold']})", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", rotation=270, labelpad=14)

    # 数字 + 自动颜色
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    # ==========================
    # 🔥 关键：高亮风险格子
    # FP: (0,1) / FN: (1,0)
    # ==========================
    risk_cells = [(0, 1), (1, 0)]
    for (i, j) in risk_cells:
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False,
                edgecolor="red",
                linewidth=3
            )
        )

    # 可选：加注释（非常推荐）
    ax.text(1, 0.35, "FP\n误报风险", color="red", ha="center", fontsize=9)
    ax.text(0, 1.35, "FN\n漏报风险", color="red", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # ---------
    # 回归：Return
    # ---------
    if COL_RETURN in df.columns and PRED_RETURN in df.columns:
        y_true = to_num(df[COL_RETURN])
        y_pred = to_num(df[PRED_RETURN])
        save_scatter(
            y_true, y_pred,
            "Actual vs Predicted: Return",
            "Actual Return (%)",
            "Predicted Return (%)",
            os.path.join(OUT_DIR, "Return_scatter.png"),
            diagonal=True
        )


    # ---------
    # 回归：Volume（log1p）
    # ---------
    if COL_VOL in df.columns and PRED_VOL in df.columns:
        y_true = np.log1p(to_num(df[COL_VOL]))
        y_pred = np.log1p(to_num(df[PRED_VOL]))
        save_scatter(
            y_true, y_pred,
            "Actual vs Predicted: Volume (log1p)",
            "log1p(Actual Volume)",
            "log1p(Predicted Volume)",
            os.path.join(OUT_DIR, "Volume_scatter_log.png"),
            diagonal=True
        )

    # ---------
    # 分类：MarginStress -> confusion matrix（你想要的那种）
    # ---------
    if COL_MS in df.columns and PRED_MS_PROBA in df.columns:
        y_true = to_num(df[COL_MS])
        y_proba = to_num(df[PRED_MS_PROBA])

        rep = confusion_report(y_true, y_proba, threshold=THRESHOLD)

        txt_path = os.path.join(OUT_DIR, "MarginStress_confusion_report.txt")
        save_confusion_text(rep, txt_path)

        fig_path = os.path.join(OUT_DIR, "MarginStress_confusion_matrix.png")
        save_confusion_plot(rep, fig_path)

        # 同时也在终端打印一份，和你截图体验一致
        print(f"Threshold = {rep['threshold']}")
        print("Confusion Matrix:")
        print(f"TP (correctly blocked) : {rep['TP']}")
        print(f"FP (overly conservative): {rep['FP']}")
        print(f"FN (missed risk)        : {rep['FN']}")
        print(f"TN (safe approved)      : {rep['TN']}\n")
        print("Key rates:")
        print(f"Miss rate (FN / actual breaks): {rep['miss_rate']*100:.2f}%")
        print(f"Block rate (TP / predicted high risk): {rep['block_rate']*100:.2f}%")
        print("\nCommon metrics:")
        print(f"Accuracy : {rep['accuracy']*100:.2f}%")
        print(f"Precision: {rep['precision']*100:.2f}%")
        print(f"Recall   : {rep['recall']*100:.2f}%")
        print(f"F1       : {rep['f1']*100:.2f}%")

    print(f"\nSaved plots/reports to: {OUT_DIR}/")
    print(" - Return_scatter.png")
    print(" - Volume_scatter_log.png")
    print(" - MarginStress_confusion_matrix.png")
    print(" - MarginStress_confusion_report.txt")


if __name__ == "__main__":
    main()
