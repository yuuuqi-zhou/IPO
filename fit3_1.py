#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit3_slim_v4.py

修正你指出的两点：
1) 不再输出「最差跌幅≤阈值」的概率图/表（首日这一项与 stress-10% 完全重复）。
2) 明确体现时间维度：首日 / 0-5日 / 0-10日 / 0-15日
   - 若数据缺少对应“最低价列”，仍会在输出中把该窗口列出来，并在「列缺失提示.txt」/「汇总.json」里明确提示缺失。
   - 若列存在，则输出该窗口的「最差跌幅分位数表 + 分位数曲线图」。

保留：
- stress-10%(首日) 概率（按 r 分桶）+ 图
- 涨跌幅事件 概率（按 r 分桶）+ 图
- 最差跌幅（Open0 基准）分位数（首日/0-5/0-10/0-15）+ 图

最差跌幅（mentor 口径）：
    最差跌幅_T = 最低价_T / 首日开盘价 - 1

时间维度解释：
- 首日：极端盘中风险（最容易触发首日强平）
- 0-5日：短期情绪 + 配售解禁前后
- 0-10/0-15日：真实价格发现 + 做市/流动性考验
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from leverage3 import LeveragePolicy, LeveragePolicyConfig

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 基础工具
# =========================

def _safe_mkdir(p: str):
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)


def _clean_colname(c: str) -> str:
    if c is None:
        return c
    c = str(c)
    c = c.replace("\n", "").replace("\r", "")
    c = c.replace("↓", "")
    return c.strip()


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_colname(c) for c in df.columns]
    return df


def to_return_frac(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(np.abs(x))
    return x / 100.0 if (np.isfinite(med) and med > 1.5) else x


# =========================
# 读取模型（只 Return + MarginStress）
# =========================

def load_artifacts(art_dir: str):
    meta_path = os.path.join(art_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json 不存在：{meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    candidates = {
        "Return": ["Return.joblib", "rf_return.joblib", "return.joblib", "model_return.joblib"],
        "MarginStress": ["MarginStress.joblib", "rf_ms.joblib", "ms.joblib", "model_ms.joblib"],
    }

    models = {}
    for k, names in candidates.items():
        found = None
        for nm in names:
            p = os.path.join(art_dir, nm)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"找不到 {k} 模型（{art_dir}），尝试过：{names}")
        models[k] = joblib.load(found)

    return meta, models


def ensure_industry_score(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    industry_col = meta.get("industry_col", "所属Wind行业名称")
    score_map = meta.get("industry_score_map", {})

    if "Industry_Score" not in df.columns:
        if (industry_col in df.columns) and isinstance(score_map, dict) and len(score_map) > 0:
            df["Industry_Score"] = (
                df[industry_col].astype(str).fillna("UNKNOWN").map(score_map).fillna(0.0).astype(float)
            )
        else:
            df["Industry_Score"] = 0.0
    return df


def predict_return_and_pms(df: pd.DataFrame, meta: dict, models: dict):
    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError("meta.json 里缺少 feature_cols")

    if feature_cols and isinstance(feature_cols[0], (list, tuple)):
        feature_cols = list(feature_cols[0])
    feature_cols = [str(c) for c in feature_cols]

    df2 = ensure_industry_score(df, meta)
    missing = [c for c in feature_cols if c not in df2.columns]
    if missing:
        raise KeyError(f"数据缺少特征列：{missing}")

    X = df2[feature_cols].copy()

    pred_ret = np.asarray(models["Return"].predict(X), dtype=float)
    pred_ret_frac = to_return_frac(pred_ret)

    m = models["MarginStress"]
    if hasattr(m, "predict_proba"):
        p_ms = m.predict_proba(X)[:, 1]
    else:
        z = m.decision_function(X)
        p_ms = 1.0 / (1.0 + np.exp(-z))
    p_ms = np.asarray(p_ms, dtype=float)

    return pred_ret_frac, p_ms


# =========================
# 指标定义
# =========================

def stress10_from_open_low(open_px: np.ndarray, low_px: np.ndarray) -> np.ndarray:
    op = np.asarray(open_px, dtype=float)
    lo = np.asarray(low_px, dtype=float)
    out = np.full_like(op, False, dtype=bool)
    m = np.isfinite(op) & np.isfinite(lo) & (op > 0)
    out[m] = lo[m] <= 0.9 * op[m]
    return out


def worst_drop_from_open(open0: np.ndarray, min_low: np.ndarray) -> np.ndarray:
    op = np.asarray(open0, dtype=float)
    lo = np.asarray(min_low, dtype=float)
    out = np.full_like(op, np.nan, dtype=float)
    m = np.isfinite(op) & np.isfinite(lo) & (op > 0)
    out[m] = (lo[m] / op[m]) - 1.0
    return out


# =========================
# 分桶
# =========================

def _parse_bins(s: str, default_bins: np.ndarray) -> np.ndarray:
    if s is None:
        return default_bins
    s = str(s).strip()
    if not s:
        return default_bins
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) < 2:
        return default_bins
    return np.array(vals, dtype=float)


def _bucket_index(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, bins, right=False) - 1
    return np.clip(idx, 0, len(bins) - 2)


def bucket_event_rate(r: np.ndarray, event: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    r = np.asarray(r, dtype=float)
    e = np.asarray(event, dtype=bool)

    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    idx = _bucket_index(r, bins)

    rows = []
    for k, lab in enumerate(labels):
        mk = (idx == k)
        cnt = int(mk.sum())
        hit = int(e[mk].sum()) if cnt > 0 else 0
        rate = (hit / cnt) if cnt > 0 else np.nan
        rows.append({"杠杆区间(r)": lab, "样本数": cnt, "触发数": hit, "概率": rate})

    return pd.DataFrame(rows)


def _nanquantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))


def bucket_drop_quantiles(r: np.ndarray, drop: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    r = np.asarray(r, dtype=float)
    d = np.asarray(drop, dtype=float)

    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    idx = _bucket_index(r, bins)

    rows = []
    for k, lab in enumerate(labels):
        mk = (idx == k) & np.isfinite(d)
        cnt = int(mk.sum())
        if cnt == 0:
            rows.append({
                "杠杆区间(r)": lab, "样本数": 0,
                "均值": np.nan, "P50": np.nan, "P75": np.nan, "P90": np.nan, "最差": np.nan
            })
            continue

        dk = d[mk]
        rows.append({
            "杠杆区间(r)": lab,
            "样本数": int(len(dk)),
            "均值": float(np.mean(dk)),
            "P50": _nanquantile(dk, 0.50),
            "P75": _nanquantile(dk, 0.75),
            "P90": _nanquantile(dk, 0.90),
            "最差": float(np.min(dk)),  # 更负 = 更差
        })

    return pd.DataFrame(rows)


# =========================
# 绘图（只保留概率柱状图 & 分位数折线图）
# =========================

def plot_event_bucket_rates(df_bucket: pd.DataFrame, outpath: str, title: str):
    plt.figure(figsize=(7.2, 4.8))
    x = np.arange(len(df_bucket))
    y = df_bucket["概率"].to_numpy(dtype=float)
    plt.bar(x, y)
    plt.xticks(x, df_bucket["杠杆区间(r)"].tolist(), rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("概率")
    plt.xlabel("杠杆区间(r)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_drop_quantile_lines(df_q: pd.DataFrame, outpath: str, title: str):
    plt.figure(figsize=(7.8, 5.0))
    x = np.arange(len(df_q))
    plt.plot(x, df_q["P50"].to_numpy(dtype=float), marker="o", label="P50")
    plt.plot(x, df_q["P75"].to_numpy(dtype=float), marker="o", label="P75")
    plt.plot(x, df_q["P90"].to_numpy(dtype=float), marker="o", label="P90")
    plt.plot(x, df_q["最差"].to_numpy(dtype=float), marker="o", label="最差")

    plt.xticks(x, df_q["杠杆区间(r)"].tolist(), rotation=35, ha="right")
    plt.xlabel("杠杆区间(r)")
    plt.ylabel("最差跌幅(小数)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# =========================
# 主流程
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--art_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sheet", default=None)

    # columns
    ap.add_argument("--col_open", default="开盘价")
    ap.add_argument("--col_low0", default="最低价")
    ap.add_argument("--col_return", default="相对发行价涨跌幅")
    ap.add_argument("--col_volume", default="成交量")

    # 你可以用参数对齐真实列名（否则会被判定缺失）
    ap.add_argument("--col_low_5d", default="上市后5日最低")
    ap.add_argument("--col_low_10d", default="上市后10日最低")
    ap.add_argument("--col_low_15d", default="上市后15日最低")

    # r config
    ap.add_argument("--r_base", type=float, default=0.30)
    ap.add_argument("--r_min", type=float, default=0.00)
    ap.add_argument("--r_max", type=float, default=0.60)

    ap.add_argument("--r_bins", default="0,0.3,0.4,0.5,0.6")
    ap.add_argument("--ret_threshold", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    _safe_mkdir(args.out_dir)

    def log(*a):
        if args.verbose:
            print("[fit3_slim_v4]", *a, flush=True)

    # 读数据
    log("读取数据...")
    if args.data.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.data, sheet_name=0 if args.sheet is None else args.sheet)
        if isinstance(df, dict):
            df = df[next(iter(df.keys()))]
    else:
        df = pd.read_csv(args.data)

    df = clean_dataframe_columns(df)

    # 模型
    log("载入模型...")
    meta, models = load_artifacts(args.art_dir)

    log("预测 Return / stress 概率...")
    pred_ret_frac, p_ms = predict_return_and_pms(df, meta, models)

    # 成交量：直接用数据列
    if args.col_volume in df.columns:
        pred_vol = pd.to_numeric(df[args.col_volume], errors="coerce").to_numpy(dtype=float)
    else:
        pred_vol = np.full(len(df), np.nan, dtype=float)

    # 动态杠杆
    log("计算动态杠杆 r...")
    cfg = LeveragePolicyConfig(
        r_base=float(args.r_base),
        r_min=float(args.r_min),
        r_max=float(args.r_max),
    )
    policy = LeveragePolicy(cfg).fit(pred_vol=pred_vol, pred_return_frac=pred_ret_frac, set_if_missing_only=False)
    r_dyn = policy.predict_r(pred_ret_frac, pred_vol, p_ms)
    joblib.dump({"policy": policy.to_dict()}, os.path.join(args.out_dir, "policy.joblib"))

    # 必需列
    for c in [args.col_open, args.col_low0, args.col_return]:
        if c not in df.columns:
            raise KeyError(f"缺少必要列：{c}")

    open0 = pd.to_numeric(df[args.col_open], errors="coerce").to_numpy(dtype=float)
    low0 = pd.to_numeric(df[args.col_low0], errors="coerce").to_numpy(dtype=float)

    # 事件：stress-10%（首日）
    stress10 = stress10_from_open_low(open0, low0)

    # 事件：涨跌幅（真实值）
    actual_ret = to_return_frac(pd.to_numeric(df[args.col_return], errors="coerce").to_numpy(dtype=float))
    ret_event = actual_ret < float(args.ret_threshold)

    # r bins
    r_bins = _parse_bins(args.r_bins, default_bins=np.array([0, 0.3, 0.4, 0.5, 0.6], dtype=float))

    # 输出：事件概率（只保留 stress10 + return event）
    bucket_stress10 = bucket_event_rate(r_dyn, stress10, r_bins)
    bucket_ret = bucket_event_rate(r_dyn, ret_event, r_bins)

    bucket_stress10.to_csv(os.path.join(args.out_dir, "分桶_stress10_概率.csv"), index=False, encoding="utf-8-sig")
    bucket_ret.to_csv(os.path.join(args.out_dir, "分桶_涨跌幅事件_概率.csv"), index=False, encoding="utf-8-sig")

    plot_event_bucket_rates(bucket_stress10, os.path.join(args.out_dir, "图_分桶_stress10_概率.png"),
                            "不同杠杆区间内 stress-10% 概率(首日)")
    plot_event_bucket_rates(bucket_ret, os.path.join(args.out_dir, "图_分桶_涨跌幅事件_概率.png"),
                            f"不同杠杆区间内 涨跌幅<{float(args.ret_threshold):.2f} 概率")

    # 时间维度窗口（强制列出 4 个窗口）
    time_notes = {
        "首日": "极端盘中风险（最容易触发首日强平）",
        "0-5日": "短期情绪 + 配售解禁前后",
        "0-10日": "真实价格发现 + 做市/流动性考验",
        "0-15日": "真实价格发现 + 做市/流动性考验",
    }
    window_specs = [
        ("首日", args.col_low0, "首日"),
        ("0-5日", args.col_low_5d, "5日"),
        ("0-10日", args.col_low_10d, "10日"),
        ("0-15日", args.col_low_15d, "15日"),
    ]

    missing_cols = []
    window_outputs = []

    for label, col_name, key in window_specs:
        if col_name in df.columns:
            low_arr = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
            missing = False
        else:
            low_arr = np.full(len(df), np.nan, dtype=float)  # 占位，确保窗口“体现出来”
            missing = True
            missing_cols.append(col_name)

        drop = worst_drop_from_open(open0, low_arr)
        df_q = bucket_drop_quantiles(r_dyn, drop, r_bins)

        csv_name = f"分桶_最差跌幅_分位数_{key}.csv"
        png_name = f"图_分桶_最差跌幅_分位数_{key}.png"

        df_q.to_csv(os.path.join(args.out_dir, csv_name), index=False, encoding="utf-8-sig")
        plot_drop_quantile_lines(df_q, os.path.join(args.out_dir, png_name), f"最差跌幅分位数({label})")

        window_outputs.append({
            "窗口": label,
            "列名": col_name,
            "是否缺失": missing,
            "解释": time_notes[label],
            "csv": csv_name,
            "png": png_name,
        })

    # 缺列提示
    if len(missing_cols) > 0:
        lines = ["以下列在数据中未找到（因此对应窗口的分位数表/图为空值占位）："]
        lines += [f"- {c}" for c in missing_cols]
        lines += ["", "你可以用命令行参数 --col_low_5d / --col_low_10d / --col_low_15d 指定真实列名。"]
    else:
        lines = ["0/5/10/15 日最低价列均已找到，已完整输出四个窗口的分位数表/图。"]
    with open(os.path.join(args.out_dir, "列缺失提示.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 明细输出（只保留连续最差跌幅，不再输出“最差跌幅<=阈值”的概率/事件列）
    out_df = pd.DataFrame({
        "预测涨跌幅(小数)": pred_ret_frac,
        "预测stress概率": p_ms,
        "动态r": r_dyn,
        "动态杠杆(1/(1-r))": 1.0 / np.maximum(1e-9, (1.0 - r_dyn)),

        "stress-10%(首日)": stress10.astype(int),
        "真实涨跌幅(小数)": actual_ret,
        "涨跌幅事件(真实<阈值)": ret_event.astype(int),
    })

    # 连续最差跌幅列
    for label, col_name, key in window_specs:
        if col_name in df.columns:
            low_arr = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
        else:
            low_arr = np.full(len(df), np.nan, dtype=float)
        out_df[f"最差跌幅({label})"] = worst_drop_from_open(open0, low_arr)

    for c in ["证券代码", "证券简称", "上市日期", "所属Wind行业名称"]:
        if c in df.columns:
            out_df[c] = df[c].astype(str)

    out_df.to_csv(os.path.join(args.out_dir, "逐条验证_结果明细.csv"), index=False, encoding="utf-8-sig")

    # 时间维度说明（可直接放报告）
    note_lines = [
        "时间维度与风险暴露持续性说明",
        f"首日：{time_notes['首日']}",
        f"0-5日：{time_notes['0-5日']}",
        f"0-10日：{time_notes['0-10日']}",
        f"0-15日：{time_notes['0-15日']}",
        "",
        "输出口径：最差跌幅 = 最低价 / 首日开盘价 - 1（mentor 口径）",
        "展示方式：按杠杆区间(r)统计分位数(P50/P75/P90/最差)，用于回答“最差会跌到什么程度”。",
    ]
    with open(os.path.join(args.out_dir, "时间维度说明.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines))

    summary = {
        "样本数": int(len(out_df)),
        "r_base": float(args.r_base),
        "r_min": float(args.r_min),
        "r_max": float(args.r_max),
        "涨跌幅阈值": float(args.ret_threshold),
        "时间维度解释": time_notes,
        "窗口输出": window_outputs,
        "缺失列": missing_cols,
        "输出文件": {
            "逐条结果": "逐条验证_结果明细.csv",
            "事件概率": [
                "分桶_stress10_概率.csv",
                "分桶_涨跌幅事件_概率.csv",
                "图_分桶_stress10_概率.png",
                "图_分桶_涨跌幅事件_概率.png",
            ],
            "最差跌幅分位数(四窗口)": window_outputs,
            "列缺失提示": "列缺失提示.txt",
            "时间说明": "时间维度说明.txt",
        },
        "说明": "已删除所有“最差跌幅阈值概率”图/表；首日阈值事件与 stress-10% 等价，仅保留 stress-10%。",
    }

    with open(os.path.join(args.out_dir, "汇总.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("完成：输出已保存到", args.out_dir)


if __name__ == "__main__":
    main()
