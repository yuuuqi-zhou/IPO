import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager as fm

from utils import (
    load_artifacts,
    predict_all,
    load_leverage_policy,
    compute_dynamic_funding_ratio,
)

# =========================
# Page
# =========================
st.set_page_config(page_title="Prediction", layout="wide")
st.title("首日表现预测及融资杠杆方案")

# =========================
# Button style (keep yours)
# =========================
st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: #e74c3c !important;
    color: white !important;
    height: 3em !important;
    font-weight: 700 !important;
    border: 0px !important;
}
div.stButton > button:first-child:hover {
    background-color: #cf3f33 !important;
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# (2) Font: Cloud 没有中文字体 -> 必须显式加载字体文件才不会方框
# =========================
def setup_cn_font():
    candidates = [
        "assets/fonts/NotoSansSC-Regular.otf",
        "assets/fonts/SourceHanSansSC-Regular.otf",
        "assets/fonts/NotoSansCJK-Regular.ttc",
        "assets/fonts/SimHei.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True, fp
    # fallback：只能尽量用系统字体（Cloud 大概率没有）
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return False, None

font_ok, font_path = setup_cn_font()
if not font_ok:
    st.caption("服务器环境可能没有中文字体，图里中文可能显示为方框。建议把中文字体文件放到 repo 的 assets/fonts/ 目录。")

# =========================
# History data (真实分布基准)
# 你已上传：港股_new.csv （在仓库根目录）
# =========================
HIST_CSV = "港股_new.csv"
COL_RETURN = "相对发行价涨跌幅"
COL_VOL = "成交量"
COL_MS = "marginstress10"

@st.cache_data(show_spinner=False)
def load_hist_df(csv_path: str):
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None

hist_df = load_hist_df(HIST_CSV)

# =========================
# Helpers
# =========================
def to_float_or_nan(s: str):
    try:
        s = "" if s is None else str(s).strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def percentile_of_score(arr, x):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or x is None or (not np.isfinite(x)):
        return None
    return float((arr <= x).mean() * 100.0)

def trimmed_xlim(arr, low_q=0.01, high_q=0.99):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    if lo == hi:
        return None
    return (lo, hi)

# =========================
# (1) 更明显的“定位图”
# =========================
def draw_hist_on_axis(ax, hist_vals, marker, title, xlabel, bins=28, xlim=None, fmt_value=None):
    hist_vals = np.asarray(hist_vals, dtype=float)
    hist_vals = hist_vals[np.isfinite(hist_vals)]

    ax.hist(hist_vals, bins=bins, alpha=0.35, edgecolor="white")

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("个数")

    if marker is None or (not np.isfinite(marker)) or hist_vals.size == 0:
        return

    pct = percentile_of_score(hist_vals, marker)

    # 一个 bin 宽度做高亮带
    try:
        _, bin_edges = np.histogram(hist_vals, bins=bins)
        bw = float(bin_edges[1] - bin_edges[0])
    except Exception:
        bw = 0.0

    if bw > 0:
        ax.axvspan(marker - 0.5 * bw, marker + 0.5 * bw, alpha=0.18)

    ax.axvline(marker, linewidth=3)
    ymax = ax.get_ylim()[1]

    ax.annotate(
        f"{val_text}\nPercentile: {pct_text}",
        xy=(marker, ymax*0.85),
        xytext=(marker, ymax*0.98),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.20),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, alpha=0.8),
    )

    if fmt_value is None:
        val_text = f"{marker:.3f}"
    else:
        val_text = fmt_value(marker)

    pct_text = f"{pct:.1f}%" if pct is not None else "N/A"
    ax.text(
        0.98,
        0.95,
        f"{val_text}\nPercentile: {pct_text}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.20),
    )

# =========================
# (3) 跌破更直观：用真实0/1做“条件跌破率地图”，并把当前点标进去
# =========================
def compute_breach_grid(df, col_ret, col_vol, col_ms, n_ret=10, n_vol=10):
    d = df[[col_ret, col_vol, col_ms]].copy()
    d = d.dropna()
    d[col_ret] = pd.to_numeric(d[col_ret], errors="coerce")
    d[col_vol] = pd.to_numeric(d[col_vol], errors="coerce")
    d[col_ms] = pd.to_numeric(d[col_ms], errors="coerce")
    d = d.dropna()

    d["_vol_log"] = np.log1p(d[col_vol].values.astype(float))

    ret_edges = np.quantile(d[col_ret].values, np.linspace(0, 1, n_ret + 1))
    vol_edges = np.quantile(d["_vol_log"].values, np.linspace(0, 1, n_vol + 1))

    ret_edges = np.unique(ret_edges)
    vol_edges = np.unique(vol_edges)
    if ret_edges.size < 3 or vol_edges.size < 3:
        return None

    d["_ret_bin"] = np.digitize(d[col_ret].values, ret_edges[1:-1], right=True)
    d["_vol_bin"] = np.digitize(d["_vol_log"].values, vol_edges[1:-1], right=True)

    grid = np.full((vol_edges.size - 1, ret_edges.size - 1), np.nan, dtype=float)
    cnt = np.zeros_like(grid, dtype=int)

    for (vb, rb), g in d.groupby(["_vol_bin", "_ret_bin"]):
        rate = float(g[col_ms].mean())
        grid[vb, rb] = rate
        cnt[vb, rb] = int(len(g))

    return grid, cnt, ret_edges, vol_edges

def draw_breach_heatmap(ax, grid, cnt, ret_edges, vol_edges, pr, pv_raw):
    ax.set_title("跌破率地图（真实历史 0/1 统计）", fontsize=12, fontweight="bold")
    ax.set_xlabel("首日涨跌幅（分位分箱）")
    ax.set_ylabel("log(1 + 成交量)（分位分箱）")

    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels([f"Q{i+1}" for i in range(grid.shape[1])], fontsize=9)
    ax.set_yticklabels([f"Q{i+1}" for i in range(grid.shape[0])], fontsize=9)

    # 标注样本数（太挤可删）
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if cnt[y, x] > 0:
                ax.text(x, y, str(cnt[y, x]), ha="center", va="center", fontsize=8, alpha=0.7)

    # 当前预测点落点（pr 在真实空间，pv_raw 在 log1p 空间）
    if np.isfinite(pr) and np.isfinite(pv_raw):
        rb = int(np.digitize([pr], ret_edges[1:-1], right=True)[0])
        vb = int(np.digitize([pv_raw], vol_edges[1:-1], right=True)[0])
        rb = max(0, min(rb, grid.shape[1] - 1))
        vb = max(0, min(vb, grid.shape[0] - 1))

        ax.scatter([rb], [vb], s=180, marker="o", facecolors="none", linewidths=3, zorder=5)

        rate_here = grid[vb, rb]
        if np.isfinite(rate_here):
            ax.text(
                0.98,
                0.95,
                f"当前落点跌破率≈{rate_here*100:.1f}%",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.20),
            )
    return im

# =========================
# Load artifacts / policy
# =========================
meta, models = load_artifacts()
policy = load_leverage_policy()

feature_cols = meta.get("feature_cols", [])
industry_map = meta.get("industry_score_map", {})
LABEL_MAP = meta.get("label_map", {})

# =========================
# Inputs
# =========================
x = {}

st.subheader("行业")
industry_list = sorted(industry_map.keys()) if isinstance(industry_map, dict) else []
industry_choice = st.selectbox("所属 Wind 行业名称", industry_list, index=0, key="industry_choice")

if industry_choice == "":
    x["所属Wind行业名称"] = np.nan
    x["Industry_Score"] = np.nan
else:
    x["所属Wind行业名称"] = industry_choice
    x["Industry_Score"] = float(industry_map.get(industry_choice, 0.0))

st.divider()
st.subheader("参数")

left, right = st.columns(2)
skip_cols = {"所属Wind行业名称", "Industry_Score", "AH_dummy"}

ah_checked = st.checkbox("是否为 AH 股", value=False, key="ah_dummy")
x["AH_dummy"] = 1 if ah_checked else 0

cols_to_render = [c for c in feature_cols if c not in skip_cols]
for i, col in enumerate(cols_to_render):
    box = left if i % 2 == 0 else right
    label = LABEL_MAP.get(col, col)
    s = box.text_input(label, value="", key=f"in_{col}")
    x[col] = to_float_or_nan(s)

st.divider()

# =========================
# Predict button
# =========================
run = st.button("进行预测")

if run:
    try:
        pred = predict_all(x, meta, models)

        pr = safe_float(pred.get("pred_return", np.nan))   # %
        pv_raw = safe_float(pred.get("pred_vol", np.nan))  # log1p 空间(通常)
        pms = safe_float(pred.get("p_ms", np.nan))         # 0-1

        lev = compute_dynamic_funding_ratio(
            pred_return=pr,
            pred_vol=pv_raw,
            p_ms=pms,
            policy=policy,
        )

        st.session_state["pred_result"] = {"pr": pr, "pv_raw": pv_raw, "pms": pms}
        st.session_state["lev_result"] = lev

    except Exception as e:
        st.session_state["pred_result"] = None
        st.session_state["lev_result"] = None
        st.error(f"预测失败：{e}")

# =========================
# Display
# =========================
pred_result = st.session_state.get("pred_result", None)
lev_result = st.session_state.get("lev_result", None)

if pred_result is not None:
    # 兼容旧版本 session_state：pv / pv_raw 都支持
    pr = pred_result.get("pr", np.nan)
    pv_raw = pred_result.get("pv_raw", pred_result.get("pv", np.nan))
    pms = pred_result.get("pms", np.nan)

    c1, c2, c3 = st.columns(3)
    c1.metric("涨跌幅（预测）", f"{pr:.2f}%")

    pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else pv_raw
    c2.metric("成交量（预测）", f"{pv_show:,.0f}")
    c3.metric("跌破概率（预测）", f"{pms * 100:.2f}%")

    st.subheader("预测结果在整体分布中的位置")

    if hist_df is None:
        st.info("未能读取历史基准文件：港股_new.csv（确认已 push 且在仓库根目录）")
    else:
        if (COL_RETURN not in hist_df.columns) or (COL_VOL not in hist_df.columns) or (COL_MS not in hist_df.columns):
            st.info(f"历史数据缺少必要列：需要 {COL_RETURN} / {COL_VOL} / {COL_MS}。")
        else:
            hist_ret = pd.to_numeric(hist_df[COL_RETURN], errors="coerce").dropna().values.astype(float)
            xlim_ret = trimmed_xlim(hist_ret, 0.01, 0.99)

            hist_vol = pd.to_numeric(hist_df[COL_VOL], errors="coerce").dropna().values.astype(float)
            hist_vol_log = np.log1p(hist_vol)
            xlim_vol = trimmed_xlim(hist_vol_log, 0.01, 0.99)

            colA, colB, colC = st.columns(3)

            with colA:
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                draw_hist_on_axis(
                    ax,
                    hist_ret,
                    pr,
                    "涨跌幅（真实历史分布）",
                    COL_RETURN,
                    bins=28,
                    xlim=xlim_ret,
                    fmt_value=lambda v: f"{v:.2f}%",
                )
                st.pyplot(fig, clear_figure=True)

            with colB:
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                draw_hist_on_axis(
                    ax,
                    hist_vol_log,
                    pv_raw,
                    "成交量（真实历史分布）",
                    "log(1 + 成交量)",
                    bins=28,
                    xlim=xlim_vol,
                    fmt_value=lambda v: f"{np.expm1(v):,.0f}",
                )
                st.pyplot(fig, clear_figure=True)

            # 第三张：跌破率地图（不需要历史概率）
            with colC:
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                res = compute_breach_grid(hist_df, COL_RETURN, COL_VOL, COL_MS, n_ret=10, n_vol=10)
                if res is None:
                    ax.text(0.5, 0.5, "数据分位边界不足\n（可能样本太少或值重复）", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    st.pyplot(fig, clear_figure=True)
                else:
                    grid, cnt, ret_edges, vol_edges = res
                    im = draw_breach_heatmap(ax, grid, cnt, ret_edges, vol_edges, pr, pv_raw)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Breach Rate", rotation=90)
                    st.pyplot(fig, clear_figure=True)

    st.divider()

    st.subheader("融资杠杆")
    if not lev_result:
        st.error("杠杆计算失败：无结果")
    elif not lev_result.get("ok", False):
        st.error(lev_result.get("error", "杠杆计算失败"))
    else:
        a, b = st.columns(2)
        a.metric("建议杠杆比例", f"{lev_result['ratio'] * 100:.0f}%")
        b.metric("风险分层", lev_result.get("risk_band", "N/A"))
        with st.expander("Leverage 解释", expanded=False):
            st.write(lev_result.get("detail", {}))
