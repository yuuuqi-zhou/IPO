import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

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
# Font setup (Streamlit Cloud)
# =========================
def setup_cn_font():
    # pages/Prediction.py -> repo root = ../
    repo_root = os.path.dirname(os.path.dirname(__file__))
    font_path = os.path.join(repo_root, "assets", "fonts", "NotoSansSC-Regular.otf")

    try:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True, font_path
        else:
            plt.rcParams["axes.unicode_minus"] = False
            return False, font_path
    except Exception:
        plt.rcParams["axes.unicode_minus"] = False
        return False, font_path

ok_font, _fp = setup_cn_font()

# =========================
# Load artifacts / policy
# =========================
meta, models = load_artifacts()
policy = load_leverage_policy()

feature_cols = meta.get("feature_cols", [])
industry_map = meta.get("industry_score_map", {})
LABEL_MAP = meta.get("label_map", {})

# =========================
# History data (use 港股_new.csv)
# =========================
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
HIST_CSV = os.path.join(REPO_ROOT, "港股_new.csv")

COL_RETURN = "相对发行价涨跌幅"
COL_VOL = "成交量"
COL_MS = "marginstress10"  # 真实0/1

@st.cache_data(show_spinner=False)
def load_hist_df(path: str):
    df = pd.read_csv(path)
    return df

# =========================
# Utils
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

def pct_rank(x: float, arr: np.ndarray):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(x):
        return np.nan
    return float((arr <= x).mean())

def _clean_num_series(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# Plot helpers
# =========================
def draw_hist_with_marker(ax, data, marker, title, xlabel, fmt_value, n_bins=18):
    """
    极简报告风格：
    - 灰/浅色直方图（默认颜色）
    - marker 位置：透明高亮带 + 粗竖线
    - 箭头注释：value + percentile
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    ax.hist(data, bins=n_bins, alpha=0.35, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("个数")

    if not np.isfinite(marker) or data.size == 0:
        return np.nan

    pct = pct_rank(marker, data) * 100.0

    # 高亮带：用 bin 宽近似
    try:
        q25, q75 = np.percentile(data, [25, 75])
        span = max((q75 - q25) * 0.02, (np.max(data) - np.min(data)) * 0.01)
        span = max(span, 1e-6)
    except Exception:
        span = 1.0

    ax.axvspan(marker - span, marker + span, alpha=0.15)
    ax.axvline(marker, linewidth=3)

    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0

    ax.annotate(
        f"{fmt_value}\n分位: {pct:.1f}%",
        xy=(marker, ymax * 0.80),
        xytext=(marker, ymax * 0.98),
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.28", alpha=0.20),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, alpha=0.85),
    )

    return pct

def compute_breach_grid(hist_df: pd.DataFrame, col_ret: str, col_vol: str, col_ms: str, n_ret=5, n_vol=5):
    """
    用真实历史数据：
    - ret 分位分箱
    - vol 用 log1p 后分位分箱
    - 每格计算真实 breach rate = mean(ms)
    """
    df = hist_df.copy()

    ret = pd.to_numeric(df[col_ret], errors="coerce")
    vol = pd.to_numeric(df[col_vol], errors="coerce")
    ms  = pd.to_numeric(df[col_ms], errors="coerce")

    df["_ret"] = ret
    df["_logv"] = np.log1p(vol)
    df["_ms"] = ms

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_ret", "_logv", "_ms"])

    if df.empty:
        return None

    # 分位边界
    ret_edges = np.quantile(df["_ret"].values, np.linspace(0, 1, n_ret + 1))
    vol_edges = np.quantile(df["_logv"].values, np.linspace(0, 1, n_vol + 1))

    # 防止重复边界导致 cut 报错：轻微扰动
    def _fix_edges(edges):
        edges = np.array(edges, dtype=float)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1e-9
        return edges

    ret_edges = _fix_edges(ret_edges)
    vol_edges = _fix_edges(vol_edges)

    df["_ret_bin"] = pd.cut(df["_ret"], bins=ret_edges, include_lowest=True, labels=False)
    df["_vol_bin"] = pd.cut(df["_logv"], bins=vol_edges, include_lowest=True, labels=False)

    grid = np.full((n_vol, n_ret), np.nan, dtype=float)
    cnt  = np.zeros((n_vol, n_ret), dtype=int)

    g = df.groupby(["_vol_bin", "_ret_bin"])["_ms"]
    for (vb, rb), s in g:
        if pd.isna(vb) or pd.isna(rb):
            continue
        vb = int(vb); rb = int(rb)
        if 0 <= vb < n_vol and 0 <= rb < n_ret:
            cnt[vb, rb] = int(s.shape[0])
            grid[vb, rb] = float(s.mean())

    base = float(df["_ms"].mean())
    return {
        "grid": grid,
        "cnt": cnt,
        "ret_edges": ret_edges,
        "vol_edges": vol_edges,
        "base": base,
        "df_used": df,
    }

def locate_bin(value: float, edges: np.ndarray):
    if not np.isfinite(value):
        return None
    # 返回所属 bin index
    idx = np.searchsorted(edges, value, side="right") - 1
    if idx < 0:
        idx = 0
    if idx >= len(edges) - 1:
        idx = len(edges) - 2
    return int(idx)

def draw_breach_heatmap_minimal(ax, res, pred_ret, pred_vol, title="类似首日表现下的历史跌破风险（真实样本）"):
    """
    极简 5x5：
    - 不显示格子数字
    - 白色粗框标记当前落点
    - 右上角写：怎么看 + 当前格子跌破率 vs 全样本平均
    """
    grid = res["grid"]
    base = res["base"]
    ret_edges = res["ret_edges"]
    vol_edges = res["vol_edges"]

    # pred_vol：你展示用 exmp1，但分箱用 log1p(真实成交量)
    # 这里 pred_vol 传进来应该是 pv_show（真实尺度），再 log1p
    pred_logv = np.log1p(pred_vol) if np.isfinite(pred_vol) else np.nan

    rb = locate_bin(pred_ret, ret_edges)
    vb = locate_bin(pred_logv, vol_edges)

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="Blues",  
        vmin=0,
        vmax=0.6
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("首日涨跌幅（分位区间）")
    ax.set_ylabel("log(1+成交量)（分位区间）")

    n_vol, n_ret = grid.shape
    ax.set_xticks(range(n_ret))
    ax.set_yticks(range(n_vol))
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_ret)])
    ax.set_yticklabels([f"Q{i+1}" for i in range(n_vol)])

    # 标当前格子
    rate_here = np.nan
    if rb is not None and vb is not None:
        rate_here = grid[vb, rb]
        rect = Rectangle((rb - 0.5, vb - 0.5), 1, 1, fill=False, linewidth=3, edgecolor="white")
        ax.add_patch(rect)

    # colorbar（简洁）
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("历史跌破概率", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # 右上角解释
    if np.isfinite(rate_here):
        delta = rate_here - base
        txt = (
            "该图展示了在不同首日表现区间内的历史跌破风险分布\n"
            f"当前预测对应区间的历史跌破率约为 {rate_here*100:.1f}%，"
            f"相较全样本平均水平（{base*100:.1f}%）{delta*100:+.1f}pct"
        )
    else:
        txt = (
            "该图展示了在不同首日表现区间内的历史跌破风险分布\n"
            f"全样本平均跌破率约为 {base*100:.1f}%"
        )

    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.30", alpha=0.18),
    )

# =========================
# 1) Inputs (always render)
# =========================
x = {}

st.subheader("行业")

industry_list = sorted(industry_map.keys()) if isinstance(industry_map, dict) else []
industry_options = industry_list

industry_choice = st.selectbox(
    "所属 Wind 行业名称",
    industry_options,
    index=0 if len(industry_options) > 0 else 0,
    key="industry_choice",
)

if industry_choice == "" or industry_choice is None:
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
# 2) Button
# =========================
run = st.button("进行预测")

# =========================
# 3) On click: predict + save
# =========================
if run:
    try:
        pred = predict_all(x, meta, models)

        pr = safe_float(pred.get("pred_return", np.nan))  # 预测涨跌幅（%）
        pv_raw = safe_float(pred.get("pred_vol", np.nan))  # 可能是log1p空间
        pms = safe_float(pred.get("p_ms", np.nan))         # 预测跌破概率

        # 展示用：把成交量从 log1p 空间还原
        pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else pv_raw

        lev = compute_dynamic_funding_ratio(
            pred_return=pr,
            pred_vol=pv_raw,   # 杠杆函数如果按你原来写的是用“模型空间”，这里保持不变
            p_ms=pms,
            policy=policy,
        )

        st.session_state["pred_result"] = {
            "pr": pr,
            "pv_raw": pv_raw,
            "pv_show": pv_show,
            "pms": pms,
        }
        st.session_state["lev_result"] = lev

    except Exception as e:
        st.session_state["pred_result"] = None
        st.session_state["lev_result"] = None
        st.error(f"预测失败：{e}")

# =========================
# 4) Display
# =========================
pred_result = st.session_state.get("pred_result", None)
lev_result = st.session_state.get("lev_result", None)

if pred_result is not None:
    pr = pred_result.get("pr", np.nan)
    pv_raw = pred_result.get("pv_raw", pred_result.get("pv", np.nan))
    pv_show = pred_result.get("pv_show", np.nan)
    if not np.isfinite(pv_show):
        pv_show = float(np.expm1(pv_raw)) if np.isfinite(pv_raw) else np.nan
    pms = pred_result.get("pms", np.nan)

    c1, c2, c3 = st.columns(3)
    c1.metric("涨跌幅（预测）", f"{pr:.2f}%")
    c2.metric("成交量（预测）", f"{pv_show:,.0f}")
    c3.metric("跌破概率（预测）", f"{pms*100:.2f}%")

    st.divider()

    # =========================
    # 分布定位图（3并排）
    # =========================
    st.subheader("预测结果在整体分布中的位置")

    # 历史数据读取
    if not os.path.exists(HIST_CSV):
        st.warning(f"未找到历史基准文件：{HIST_CSV}（确认已 push 到仓库根目录）")
    else:
        try:
            hist_df = load_hist_df(HIST_CSV)

            # 清洗真实列
            s_ret = _clean_num_series(hist_df, COL_RETURN)
            s_vol = _clean_num_series(hist_df, COL_VOL)
            s_ms  = _clean_num_series(hist_df, COL_MS)

            # 画布：3列
            colA, colB, colC = st.columns(3)

            # 图1：涨跌幅真实分布 + marker
            with colA:
                fig1, ax1 = plt.subplots(figsize=(5.2, 3.6))
                draw_hist_with_marker(
                    ax1,
                    data=s_ret.values,
                    marker=pr,
                    title="涨跌幅（真实历史分布）",
                    xlabel="相对发行价涨跌幅",
                    fmt_value=f"{pr:.2f}%",
                    n_bins=18,
                )
                st.pyplot(fig1, use_container_width=True)

            # 图2：成交量 log1p 分布 + marker
            with colB:
                fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
                s_logv = np.log1p(s_vol.values)
                marker_logv = np.log1p(pv_show) if np.isfinite(pv_show) and pv_show >= 0 else np.nan
                draw_hist_with_marker(
                    ax2,
                    data=s_logv,
                    marker=marker_logv,
                    title="成交量（真实历史分布）",
                    xlabel="log(1 + 成交量)",
                    fmt_value=f"{pv_show:,.0f}",
                    n_bins=18,
                )
                st.pyplot(fig2, use_container_width=True)

            # 图3：极简 5x5 历史条件跌破率
            with colC:
                fig3, ax3 = plt.subplots(figsize=(5.2, 3.6))
                res = compute_breach_grid(hist_df, COL_RETURN, COL_VOL, COL_MS, n_ret=5, n_vol=5)
                if res is None:
                    ax3.axis("off")
                    ax3.text(0.5, 0.5, "历史数据不足，无法计算跌破率分布", ha="center", va="center")
                else:
                    draw_breach_heatmap_minimal(
                        ax3,
                        res=res,
                        pred_ret=pr,
                        pred_vol=pv_show,
                        title="类似首日表现下的历史跌破风险（真实样本）",
                    )
                st.pyplot(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"历史分布图生成失败：{e}")

    st.divider()

    # =========================
    # 杠杆展示
    # =========================
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
