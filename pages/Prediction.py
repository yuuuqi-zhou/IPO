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
# ✅ 历史分布基准（直接读 repo 根目录的 pred_output.csv）
# 当前文件在 pages/ 下，所以用 ..
# ==========
HIST_CSV = "pred_output.csv"

@st.cache_data(show_spinner=False)
def load_hist_df(csv_path: str):
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None

hist_df = load_hist_df(HIST_CSV)

def pick_hist_cols(df: pd.DataFrame):
    """
    优先用真实列做“整体分布”（更符合 mentor 的“整体分布”）
    如果真实列不存在，就用 pred_* 列（一定存在，也合理）
    """
    if df is None:
        return None

    # 真实列（如果 pred_output.csv 是在原数据上追加 pred_*，通常这些会存在）
    real_ret = "相对发行价涨跌幅"
    real_vol = "成交量"
    real_ms  = "marginstress10"

    if (real_ret in df.columns) and (real_vol in df.columns) and (real_ms in df.columns):
        return {
            "mode": "real",
            "ret": real_ret,
            "vol": real_vol,
            "ms":  real_ms,
        }

    # 预测列兜底
    pred_ret = "pred_return"
    pred_vol = "pred_vol"
    pred_ms  = "pred_ms_proba"

    if (pred_ret in df.columns) and (pred_vol in df.columns) and (pred_ms in df.columns):
        return {
            "mode": "pred",
            "ret": pred_ret,
            "vol": pred_vol,
            "ms":  pred_ms,
        }

    return None

HIST_COLS = pick_hist_cols(hist_df)

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

# ==========
# load artifacts / policy
# ==========
meta, models = load_artifacts()
policy = load_leverage_policy()

feature_cols = meta.get("feature_cols", [])
industry_map = meta.get("industry_score_map", {})
LABEL_MAP = meta.get("label_map", {})

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

        pr = safe_float(pred.get("pred_return", np.nan))  # 单位：%
        pv = safe_float(pred.get("pred_vol", np.nan))     # 可能是 log1p 空间
        pms = safe_float(pred.get("p_ms", np.nan))        # 0-1

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

    # 成交量显示：若 pv_raw 是 log1p 空间 -> expm1 还原显示
    pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else pv_raw
    c2.metric("成交量（预测）", f"{pv_show:,.0f}")
    c3.metric("跌破概率（预测）", f"{pms * 100:.2f}%")

    # ==========
    # ✅ mentor 要的：整体分布定位图（直接放在预测结果下面）
    # ==========
    st.subheader("预测结果在整体分布中的位置")

    if hist_df is None:
        st.info("未能读取历史基准文件：../pred_output.csv（确认它在仓库根目录并已 push）")
    elif HIST_COLS is None:
        st.info("历史基准数据缺少必要列：需要真实列(相对发行价涨跌幅/成交量/marginstress10) 或 预测列(pred_return/pred_vol/pred_ms_proba)。")
    else:
        mode = HIST_COLS["mode"]
        col_ret = HIST_COLS["ret"]
        col_vol = HIST_COLS["vol"]
        col_ms  = HIST_COLS["ms"]

        # 1) Return 分布
        if col_ret in hist_df.columns and np.isfinite(pr):
            plot_dist_with_marker(
                hist_df[col_ret].dropna().values,
                pr,
                f"首日涨跌幅：整体分布定位（基准：{('真实值' if mode=='real' else '历史预测值')}）",
                col_ret,
                bins=35,
            )

        # 2) Volume 分布
        # - 如果基准是“真实成交量”，用 log1p 再对齐 pv_raw（因为你的 pv_raw 很可能在 log1p 空间）
        # - 如果基准是 pred_vol，就直接用 pred_vol 分布（同一空间）
        if col_vol in hist_df.columns and np.isfinite(pv_raw):
            if mode == "real":
                hv = hist_df[col_vol].dropna().values.astype(float)
                hv_log = np.log1p(hv)
                plot_dist_with_marker(
                    hv_log,
                    pv_raw,
                    "成交量：整体分布定位（log1p，对齐模型输出）",
                    "log(1 + 成交量)",
                    bins=35,
                )
            else:
                plot_dist_with_marker(
                    hist_df[col_vol].dropna().values.astype(float),
                    pv_raw,
                    "成交量：整体分布定位（基准：历史预测值 pred_vol）",
                    col_vol,
                    bins=35,
                )

        # 3) Risk 分布（概率/指标）
        if col_ms in hist_df.columns and np.isfinite(pms):
            plot_dist_with_marker(
                hist_df[col_ms].dropna().values.astype(float),
                pms,
                f"下行风险：整体分布定位（基准：{('真实值' if mode=='real' else '历史预测值')}）",
                col_ms,
                bins=35,
            )

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
