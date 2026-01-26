import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    load_artifacts,
    predict_all,
    load_leverage_policy,
    compute_dynamic_funding_ratio,
)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="Prediction", layout="wide")
st.title("首日表现预测及融资杠杆方案")

# ==========
# 按钮样式（沿用你原来的风格）
# ==========
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

# ==========
# ✅ 历史分布基准：真实数据（你已上传的 csv）
# Streamlit Cloud 的 cwd 通常就是仓库根目录，所以直接写文件名即可
# ==========
HIST_CSV = "港股_new.csv"

# 真实历史列名（按你数据）
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
    """返回 x 在 arr 中的分位（0~100）。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or (x is None) or (not np.isfinite(x)):
        return None
    return float((arr <= x).mean() * 100.0)

def trimmed_xlim(arr, low_q=0.01, high_q=0.99):
    """用分位裁剪 x 轴范围，避免极端值把图挤扁。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    if lo == hi:
        return None
    return (lo, hi)

def draw_hist_on_axis(ax, hist_vals, marker, title, xlabel, bins=30, xlim=None):
    hist_vals = np.asarray(hist_vals, dtype=float)
    hist_vals = hist_vals[np.isfinite(hist_vals)]
    if hist_vals.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        return

    ax.hist(hist_vals, bins=bins)
    if marker is not None and np.isfinite(marker):
        ax.axvline(marker, linewidth=2)

        pct = percentile_of_score(hist_vals, marker)
        if pct is not None:
            ax.text(
                0.98, 0.95, f"Percentile: {pct:.1f}%",
                ha="right", va="top", transform=ax.transAxes
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    if xlim is not None:
        ax.set_xlim(xlim)

# ==========
# load artifacts / policy
# ==========
meta, models = load_artifacts()
policy = load_leverage_policy()

feature_cols = meta.get("feature_cols", [])
industry_map = meta.get("industry_score_map", {})
LABEL_MAP = meta.get("label_map", {})

# ==========
# 1) Inputs
# ==========
x = {}

st.subheader("行业")

industry_list = sorted(industry_map.keys()) if isinstance(industry_map, dict) else []
industry_options = industry_list

industry_choice = st.selectbox(
    "所属 Wind 行业名称",
    industry_options,
    index=0,
    key="industry_choice",
)

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

# ==========
# 2) Button
# ==========
run = st.button("进行预测")

# ==========
# 3) Predict + save
# ==========
if run:
    try:
        pred = predict_all(x, meta, models)

        pr = safe_float(pred.get("pred_return", np.nan))  # %
        pv_raw = safe_float(pred.get("pred_vol", np.nan))  # 可能是 log1p 空间
        pms = safe_float(pred.get("p_ms", np.nan))  # 0-1

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

# ==========
# 4) Display
# ==========
pred_result = st.session_state.get("pred_result", None)
lev_result = st.session_state.get("lev_result", None)

if pred_result is not None:
    pr = pred_result.get("pr", np.nan)
    pv_raw = pred_result.get("pv_raw", pred_result.get("pv", np.nan))
    pms = pred_result.get("pms", np.nan)

    # 顶部三个 metric
    c1, c2, c3 = st.columns(3)
    c1.metric("涨跌幅（预测）", f"{pr:.2f}%")

    # 成交量显示：若 pv_raw 是 log1p 空间 -> expm1 还原显示
    pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else pv_raw
    c2.metric("成交量（预测）", f"{pv_show:,.0f}")
    c3.metric("跌破概率（预测）", f"{pms * 100:.2f}%")

    # ==========
    # ✅ 三个图并排：整体分布定位
    # ==========
    st.subheader("预测结果在整体分布中的位置")

    if hist_df is None:
        st.info("未能读取历史基准文件：港股_new.csv（确认已 push 且在仓库根目录）")
    else:
        # 取真实历史分布
        if (COL_RETURN not in hist_df.columns) or (COL_VOL not in hist_df.columns) or (COL_MS not in hist_df.columns):
            st.info(
                f"历史数据缺少必要列：需要 {COL_RETURN} / {COL_VOL} / {COL_MS}。"
            )
        else:
            # 真实历史：涨跌幅
            hist_ret = hist_df[COL_RETURN].dropna().values.astype(float)
            xlim_ret = trimmed_xlim(hist_ret, 0.01, 0.99)

            # 真实历史：成交量（用 log1p 与 pv_raw 对齐）
            hist_vol = hist_df[COL_VOL].dropna().values.astype(float)
            hist_vol_log = np.log1p(hist_vol)
            xlim_vol = trimmed_xlim(hist_vol_log, 0.01, 0.99)

            # 真实历史：风险指标（通常是 0/1 或概率）
            hist_ms = hist_df[COL_MS].dropna().values.astype(float)
            xlim_ms = trimmed_xlim(hist_ms, 0.01, 0.99)

            colA, colB, colC = st.columns(3)

            # 统一更小画布，让三图并排更紧凑
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
                )
                st.pyplot(fig, clear_figure=True)

            with colB:
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                draw_hist_on_axis(
                    ax,
                    hist_vol_log,
                    pv_raw,
                    "成交量（log1p 历史分布）",
                    "log(1 + 成交量)",
                    bins=28,
                    xlim=xlim_vol,
                )
                st.pyplot(fig, clear_figure=True)

            with colC:
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                draw_hist_on_axis(
                    ax,
                    hist_ms,
                    pms,
                    "跌破概率/指标（真实历史分布）",
                    COL_MS,
                    bins=28,
                    xlim=xlim_ms,
                )
                st.pyplot(fig, clear_figure=True)

    st.divider()

    # ==========
    # 融资杠杆
    # ==========
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
