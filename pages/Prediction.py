import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_artifacts,
    predict_all,
    load_leverage_policy,
    compute_dynamic_funding_ratio,
)

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
# load artifacts / policy
# ==========
meta, models = load_artifacts()
policy = load_leverage_policy()

feature_cols = meta.get("feature_cols", [])
industry_map = meta.get("industry_score_map", {})
LABEL_MAP = meta.get("label_map", {})

# ==========
# ✅ 历史分布：直接读 repo 根目录的 港股_new.xlsx
# Prediction.py 在 pages/，所以 ROOT = pages 的上一级
# ==========
HIST_PATH = "../pred_output.csv"
hist_df = load_hist_df(HIST_CSV)

# 你历史数据里的列名（按你截图）
COL_RETURN = "相对发行价涨跌幅"
COL_VOL = "成交量"
COL_MS = "marginstress10"

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_hist_df(path: Path):
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in [".csv"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        # 云端没装 openpyxl 会失败，所以这里不强依赖
        try:
            return pd.read_excel(path)
        except Exception:
            return None
    return None

def plot_dist_with_marker(hist_vals, marker, title, xlabel, bins=35):
    hist_vals = np.asarray(hist_vals, dtype=float)
    hist_vals = hist_vals[np.isfinite(hist_vals)]
    if hist_vals.size == 0:
        return
    if marker is None or (not np.isfinite(marker)):
        return

    fig, ax = plt.subplots()
    ax.hist(hist_vals, bins=bins)
    ax.axvline(marker, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)

hist_df = load_hist_df(HIST_XLSX)

# ==========
# utils: parse float (empty/invalid -> NaN)
# ==========
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

# ==========
# 1) Inputs (always render; do NOT put inside if run)
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

# 这些列已经在上面处理/或单独处理
skip_cols = {"所属Wind行业名称", "Industry_Score", "AH_dummy"}

ah_checked = st.checkbox("是否为 AH 股", value=False, key="ah_dummy")
x["AH_dummy"] = 1 if ah_checked else 0

cols_to_render = [c for c in feature_cols if c not in skip_cols]

for i, col in enumerate(cols_to_render):
    box = left if i % 2 == 0 else right
    label = LABEL_MAP.get(col, col)

    # 全部用 text_input，才能真正“可空”
    s = box.text_input(label, value="", key=f"in_{col}")
    x[col] = to_float_or_nan(s)

st.divider()

# ==========
# 2) Button
# ==========
run = st.button("进行预测")

# ==========
# 3) On click: predict + save to session_state
# ==========
if run:
    try:
        pred = predict_all(x, meta, models)

        pr = safe_float(pred.get("pred_return", np.nan))
        pv = safe_float(pred.get("pred_vol", np.nan))
        pms = safe_float(pred.get("p_ms", np.nan))

        # 杠杆建议
        lev = compute_dynamic_funding_ratio(
            pred_return=pr,
            pred_vol=pv,
            p_ms=pms,
            policy=policy,
        )

        # 保存结果，避免 rerun 后“没了”
        st.session_state["pred_result"] = {"pr": pr, "pv": pv, "pms": pms}
        st.session_state["lev_result"] = lev

    except Exception as e:
        st.session_state["pred_result"] = None
        st.session_state["lev_result"] = None
        st.error(f"预测失败：{e}")

# ==========
# 4) Display (use session_state)
# ==========
pred_result = st.session_state.get("pred_result", None)
lev_result = st.session_state.get("lev_result", None)

if pred_result is not None:
    pr = pred_result["pr"]
    pv_raw = pred_result["pv"]  # 原始模型输出（可能是 log1p 空间）
    pms = pred_result["pms"]

    c1, c2, c3 = st.columns(3)
    c1.metric("涨跌幅（预测）", f"{pr:.2f}%")

    pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else pv_raw
    c2.metric("成交量（预测）", f"{pv_show:,.0f}")
    c3.metric("跌破概率（预测）", f"{pms * 100:.2f}%")

    # ==========
    # ✅ 直接在输出预测结果后面加：整体分布定位图（不加上传模块）
    # ==========
    st.subheader("预测结果在整体分布中的位置")

    if hist_df is None:
        st.info(f"未找到历史基准文件：{HIST_XLSX}（确认它在仓库根目录且已 push）")
    else:
        # 1) 涨跌幅分布（用真实历史列）
        if COL_RETURN in hist_df.columns and np.isfinite(pr):
            plot_dist_with_marker(
                hist_df[COL_RETURN].dropna().values,
                pr,
                "首日涨跌幅：历史分布定位",
                "首日涨跌幅(%)",
                bins=35,
            )
        else:
            st.caption(f"涨跌幅分布图未绘制：历史数据缺列「{COL_RETURN}」或预测值无效。")

        # 2) 成交量分布：用 log1p 对齐 pv_raw（避免竖线位置错）
        if COL_VOL in hist_df.columns and np.isfinite(pv_raw):
            hv = hist_df[COL_VOL].dropna().values.astype(float)
            hv_log = np.log1p(hv)
            plot_dist_with_marker(
                hv_log,
                pv_raw,
                "成交量：历史分布定位（log1p，对齐模型输出）",
                "log(1 + 成交量)",
                bins=35,
            )
        else:
            st.caption(f"成交量分布图未绘制：历史数据缺列「{COL_VOL}」或预测值无效。")

        # 3) 风险指标/概率分布：marginstress10
        if COL_MS in hist_df.columns and np.isfinite(pms):
            plot_dist_with_marker(
                hist_df[COL_MS].dropna().values.astype(float),
                pms,
                "下行风险：历史分布定位",
                COL_MS,
                bins=35,
            )
        else:
            st.caption(f"风险分布图未绘制：历史数据缺列「{COL_MS}」或预测值无效。")

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
