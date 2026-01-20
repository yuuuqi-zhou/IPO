import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# =========================
# 0) 确保项目根目录在 sys.path（解决 policy.joblib 反序列化找不到类的问题）
#    utils.py 在 IPO_6/ 下，所以根目录就是它所在目录
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# 1) 你项目的真实路径（按你给的）
# =========================
ARTIFACT_DIR = "/Users/yuqizhou/IPO_6/rf_shap_outputs/artifacts"
POLICY_PATH  = "/Users/yuqizhou/IPO_6/leverage_outputs/policy.joblib"


# =========================
# 2) 小工具
# =========================
def _is_model(obj) -> bool:
    # sklearn pipeline / estimator 一般都有 predict
    return hasattr(obj, "predict")

def _is_classifier(obj) -> bool:
    return hasattr(obj, "predict_proba")

def _looks_like_meta_dict(d: dict) -> bool:
    # 只要出现这些 key，基本可以确定是 meta 或 meta bundle
    keys = set(d.keys())
    return ("feature_cols" in keys) or ("industry_score_map" in keys) or ("meta" in keys) or ("models" in keys)

def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _scan_joblibs(dir_path: str):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"找不到 artifacts 目录：{dir_path}")
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".joblib")]
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f"{dir_path} 下没有任何 .joblib 文件")
    return files

def band_to_cn(band: str) -> str:
    if not band:
        return ""
    m = {
        "p<p1": "低风险",
        "p1<=p<p2": "中低风险",
        "p2<=p<p3(smooth)": "中风险",
        "p3<=p<p4(smooth)": "中高风险",
        "p>=p4": "高风险",
        "p=nan": "N/A",
    }
    return m.get(band, band)  # 兜底：没匹配就原样返回

# =========================
# 3) load_artifacts：不依赖文件名，自动识别 meta / models
# =========================
def load_artifacts(artifact_dir: str = ARTIFACT_DIR):
    """
    兼容两种常见产物：
    A) 每个模型一个 joblib（3 个：Return/Volume/MarginStress）
    B) 某个 joblib 里是 dict：{"meta":..., "models":...} 或直接包含 feature_cols 等
    """
    joblibs = _scan_joblibs(artifact_dir)

    meta = None
    models = {}

    # 先把所有 joblib load 进来（按文件名 + 内容自动识别）
    loaded = []
    for p in joblibs:
        obj = joblib.load(p)
        loaded.append((p, obj))

    # 1) 先找 meta bundle（dict 且像 meta）
    for p, obj in loaded:
        if isinstance(obj, dict) and _looks_like_meta_dict(obj):
            # 情况：{"meta":..., "models":...}
            if "meta" in obj and isinstance(obj["meta"], dict):
                meta = obj["meta"]
            elif meta is None:
                # 情况：dict 自身就是 meta
                meta = obj

            # 情况：{"models": {...}}
            if "models" in obj and isinstance(obj["models"], dict):
                for k, m in obj["models"].items():
                    if _is_model(m):
                        models[k] = m

    # 2) 如果还没拿到 meta，尝试同目录找 meta.json
    if meta is None:
        meta_json = os.path.join(artifact_dir, "meta.json")
        if os.path.exists(meta_json):
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)

    # 3) 如果 models 还不全：把“纯模型 joblib”按文件名关键词归类
    #    （你说 artifacts 里只有 3 个 joblib，基本会走到这里）
    for p, obj in loaded:
        if _is_model(obj):
            name = os.path.basename(p).lower()
            # 用关键词猜测属于哪个目标
            if "return" in name:
                models.setdefault("return", obj)
            elif "vol" in name or "volume" in name:
                models.setdefault("vol", obj)
            elif "ms" in name or "margin" in name or "stress" in name:
                models.setdefault("ms", obj)

    # 4) 如果仍然没分类出来（文件名没关键词），就按“出现顺序”兜底塞进去
    #    （Return/Volume/MarginStress 的顺序你可以自己调整）
    if len(models) == 0:
        pure_models = [(p, obj) for p, obj in loaded if _is_model(obj)]
        keys = ["return", "vol", "ms"]
        for i, (p, obj) in enumerate(pure_models[:3]):
            models[keys[i]] = obj

    if meta is None:
        # 最低限度：feature_cols 用空，让 predict_all 直接用输入 x 的 key
        meta = {"feature_cols": [], "industry_score_map": {}}

    if len(models) == 0:
        raise FileNotFoundError(f"在 {artifact_dir} 的 joblib 中没有识别出任何可 predict 的模型")

    return meta, models


# =========================
# 4) predict_all：输出统一键（Prediction 页面不会 KeyError）
# =========================
def predict_all(x: dict, meta: dict, models: dict):
    feature_cols = meta.get("feature_cols") or list(x.keys())

    # 保留 NaN（让 pipeline 内部 imputer 处理）
    row = {c: x.get(c, np.nan) for c in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)

    pred_return = np.nan
    pred_vol = np.nan
    p_ms = np.nan

    if "return" in models:
        try:
            pred_return = float(models["return"].predict(X)[0])
        except Exception:
            pred_return = np.nan

    if "vol" in models:
        try:
            pred_vol = float(models["vol"].predict(X)[0])
        except Exception:
            pred_vol = np.nan

    if "ms" in models:
        try:
            m = models["ms"]
            if _is_classifier(m):
                proba = m.predict_proba(X)[0]
                p_ms = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                p_ms = float(m.predict(X)[0])
        except Exception:
            p_ms = np.nan

    return {"pred_return": pred_return, "pred_vol": pred_vol, "p_ms": p_ms}


# =========================
# 5) load_leverage_policy：最鲁棒（兼容对象/字典/反序列化路径问题）
# =========================
def load_leverage_policy(policy_path: str = POLICY_PATH):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"找不到 policy.joblib：{policy_path}")

    try:
        obj = joblib.load(policy_path)
    except Exception as e:
        # 这里最常见就是 pickle 找不到 leverage_policy.LevaragePolicy
        raise RuntimeError(
            f"policy.joblib 反序列化失败：{e}\n"
            f"建议确认：1) leverage_policy.py 在项目根目录；2) streamlit 从 IPO_6 目录启动"
        )

    # 1) 直接是对象
    if hasattr(obj, "score") and hasattr(obj, "predict_r"):
        return obj

    # 2) {"policy": <对象 或 dict>}
    if isinstance(obj, dict) and "policy" in obj:
        pol = obj["policy"]
        if hasattr(pol, "score") and hasattr(pol, "predict_r"):
            return pol
        if isinstance(pol, dict):
            from leverage3 import LeveragePolicy
            return LeveragePolicy.from_dict(pol)

    # 3) 直接 dict（to_dict）
    if isinstance(obj, dict):
        from leverage3 import LeveragePolicy
        return LeveragePolicy.from_dict(obj)

    raise TypeError("policy.joblib 内容不是 LeveragePolicy / dict，无法解析")


# =========================
# 6) compute_dynamic_funding_ratio：一律用 score（避免 pred_return 参数名坑）
# =========================
def compute_dynamic_funding_ratio(pred_return: float, pred_vol: float, p_ms: float, policy):
    pr = _safe_float(pred_return, np.nan)
    pv = _safe_float(pred_vol, np.nan)
    p  = _safe_float(p_ms, np.nan)

    try:
        out = policy.score(pred_return_frac=pr, pred_vol=pv, p_ms=p)
    except Exception as e:
        return {"ok": False, "error": f"policy.score 失败：{e}"}

    # 兼容 key 名：r / broker_r
    r = out.get("r", out.get("broker_r", None))
    if r is None:
        return {"ok": False, "error": "policy.score 输出缺少 r（请检查 leverage_policy.py 的 score 返回 dict）"}

    r = float(r)
    r = max(0.0, min(1.0, r))

    # -------- 风险分层：优先用 policy 输出；否则用 p_ms 兜底 --------
    risk_band = (
        out.get("band")
        or out.get("risk_band")
        or out.get("risk_tier")
        or out.get("tier")
        or out.get("level")
        or ""
    )

    if not risk_band:
    # 五档兜底：与 leverage3 / fit3 的 p1~p4 完全一致
        c = policy.cfg
        if np.isnan(p):
            risk_band = "p=nan"
        elif p < c.p1:
            risk_band = "p<p1"
        elif p < c.p2:
            risk_band = "p1<=p<p2"
        elif p < c.p3:
            risk_band = "p2<=p<p3(smooth)"
        elif p < c.p4:
            risk_band = "p3<=p<p4(smooth)"
        else:
            risk_band = "p>=p4"

    risk_band_raw = risk_band
    risk_band_cn = band_to_cn(risk_band_raw) if risk_band_raw else risk_band_raw

    return {
        "ok": True,
        "ratio": r,
        "risk_band": risk_band_cn,      # ✅ UI 用：低/中低/中/中高/高风险
        "risk_rule": risk_band_raw,     # ✅ 可选：保留原始 band：p2<=p<p3(smooth)
        "detail": out,
    }

