import os
import json
import numpy as np
import pandas as pd
import joblib

# =========================
# 路径配置（按你的真实结构）
# =========================
DATA_PATH = "港股_2.xlsx"
MODEL_DIR = "/Users/yuqizhou/IPO_6/rf_shap_outputs/artifacts"
OUTPUT_CSV = "pred_output.csv"

# =========================
# Excel 里的真实列名（必须与你 Excel 完全一致）
# =========================
COL_INDUSTRY = "所属Wind行业名称"
COL_MM = "MM恐惧与贪婪指数"  # <-- 新因子列名（你截图最后一列）

# （可选）若你的表里本来就带这些真实值列，用于对照，不影响预测
COL_RETURN = "相对发行价涨跌幅"
COL_VOL = "成交量"
COL_MS = "marginstress10"


# =========================
# 读取 meta.json（强烈建议用它来保证训练/预测一致）
# =========================
def load_meta(model_dir: str) -> dict:
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 用训练期保存的 industry_score_map 计算 Industry_Score（不泄漏）
# =========================
def compute_industry_score_from_map(df: pd.DataFrame, industry_col: str, score_map: dict) -> pd.Series:
    ind = df[industry_col].astype(str).fillna("UNKNOWN").str.strip()
    # score_map 的 key 也是字符串行业名
    s = ind.map(score_map)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# =========================
# 构造模型输入 X（严格按 meta["feature_cols"]）
# =========================
def build_X(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError("meta.json missing 'feature_cols'.")

    X = df.copy()

    # 1) Industry_Score：如果模型需要，则用 artifacts 里的 map 来生成
    if "Industry_Score" in feature_cols:
        if COL_INDUSTRY not in X.columns:
            raise ValueError(f"Missing industry column in data: {COL_INDUSTRY}")
        score_map = meta.get("industry_score_map", {})
        if not isinstance(score_map, dict) or len(score_map) == 0:
            raise ValueError("meta.json missing 'industry_score_map' or it is empty.")
        X["Industry_Score"] = compute_industry_score_from_map(X, COL_INDUSTRY, score_map)

    # 2) MM 情绪：如果模型需要，必须存在并转为数值
    if COL_MM in feature_cols:
        if COL_MM not in X.columns:
            raise ValueError(f"Missing MM column in data: {COL_MM}")
        X[COL_MM] = pd.to_numeric(X[COL_MM], errors="coerce")

    # 3) 其余特征列：必须存在（不建议默默补 0）
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # 4) 清理无穷/缺失（交给 pipeline 的 SimpleImputer 处理，这里只做基本替换）
    X = X[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# =========================
# 主流程
# =========================
def main():
    print("Loading data...")
    df = pd.read_excel(DATA_PATH)

    print("Loading meta.json...")
    meta = load_meta(MODEL_DIR)
    feature_cols = meta.get("feature_cols", [])
    print("Model feature_cols:", feature_cols)

    print("Loading models...")
    m_return = joblib.load(os.path.join(MODEL_DIR, "Return.joblib"))
    m_vol    = joblib.load(os.path.join(MODEL_DIR, "Volume.joblib"))
    m_ms     = joblib.load(os.path.join(MODEL_DIR, "MarginStress.joblib"))

    print("Building X (with Industry_Score + MM if required)...")
    X = build_X(df, meta)

    print("Predicting Return...")
    df["pred_return"] = m_return.predict(X)

    print("Predicting Volume (log -> raw)...")
    df["pred_vol"] = np.expm1(m_vol.predict(X))

    print("Predicting MarginStress (probability)...")
    df["pred_ms_proba"] = m_ms.predict_proba(X)[:, 1]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
