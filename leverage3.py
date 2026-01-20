import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple


@dataclass
class LeveragePolicyConfig:
    """Linear (additive) leverage policy.

    Definition:
        r = broker_funding_ratio = broker_principal / (client + broker)

    Output range:
        r in [r_min, r_max]

    Notes:
        - This file keeps your original structure (piecewise risk + quantile bands)
        - Adds two upgrades:
            1) Risk gate: in high-risk regime, forbid positive Return/Liquidity boosts
            2) Smooth risk: linear interpolation inside risk bands to reduce "horizontal stripes"
    """

    # broker funding ratio r in [r_min, r_max]
    r_base: float = 0.30
    r_min: float = 0.00
    r_max: float = 0.60

    # =========================
    # 1) Risk (p(MS)) as MAIN AXIS
    # =========================
    p1: float = 0.25
    p2: float = 0.45
    p3: float = 0.65
    p4: float = 0.80

    # risk deltas (tune these ONLY if you want)
    delta_p_lt_p1: float = +0.10
    delta_p_p1_p2: float = +0.00
    delta_p_p2_p3: float = -0.05
    delta_p_p3_p4: float = -0.15
    delta_p_ge_p4: float = -0.30

    # =========================
    # 2) Return band delta (based on quantiles)
    # =========================
    delta_ret_q01: float = -0.15
    delta_ret_q05: float = -0.05
    delta_ret_q20: float = -0.01
    delta_ret_q40: float = +0.00
    delta_ret_q60: float = +0.01
    delta_ret_q80: float = +0.02
    delta_ret_q95: float = +0.04
    delta_ret_q99: float = +0.06

    # =========================
    # 3) Liquidity / Volume band delta (based on quantiles)
    # =========================
    delta_vol_q01: float = -0.10
    delta_vol_q05: float = -0.05
    delta_vol_q20: float = -0.03
    delta_vol_q40: float = -0.01
    delta_vol_q60: float = +0.00
    delta_vol_q80: float = +0.03
    delta_vol_q95: float = +0.08
    delta_vol_q99: float = +0.14

    # =========================
    # 4) Tail guards (hard caps)
    # =========================
    r_hard_cap_p: float = 0.85
    r_hard_cap: float = 0.20

    joint_cap_p: float = 0.70
    joint_cap_ret: float = -0.05
    r_joint_cap: float = 0.30


class LeveragePolicy:
    """Your additive leverage policy with small but high-impact upgrades.

    fit(): build reference quantiles for pred_vol and pred_return_frac
    score(): return r and explainability bands
    predict_r(): vectorized wrapper

    Upgrades vs your original:
      (A) Risk gate: in high-risk regime, positive boosts from Return/Liquidity are reduced/disabled
      (B) Smooth risk: inside risk bands, risk delta is linearly interpolated (reduces banding)
    """

    def __init__(self, cfg: Optional[LeveragePolicyConfig] = None):
        self.cfg = cfg or LeveragePolicyConfig()
        self.vol_ref_quantiles: Optional[np.ndarray] = None
        self.ret_ref_quantiles: Optional[np.ndarray] = None

    def fit(
        self,
        pred_vol: np.ndarray,
        pred_return_frac: np.ndarray,
        set_if_missing_only: bool = True,
    ) -> "LeveragePolicy":
        if (not set_if_missing_only) or (self.vol_ref_quantiles is None):
            v = np.asarray(pred_vol, dtype=float)
            v = v[np.isfinite(v)]
            if v.size > 0:
                qs_v = np.array([0.01, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99])
                self.vol_ref_quantiles = np.quantile(v, qs_v)
            else:
                self.vol_ref_quantiles = None

        if (not set_if_missing_only) or (self.ret_ref_quantiles is None):
            r = np.asarray(pred_return_frac, dtype=float)
            r = r[np.isfinite(r)]
            if r.size > 0:
                qs_r = np.array([0.01, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99])
                self.ret_ref_quantiles = np.quantile(r, qs_r)
            else:
                self.ret_ref_quantiles = None
        return self

    # ---------- helper: map value to coarse quantile label ----------
    def _piecewise_quantile(self, x: float, ref: Optional[np.ndarray]) -> float:
        if ref is None or (not np.isfinite(x)):
            return 0.50
        qv = ref
        if x <= qv[0]:
            return 0.01
        if x <= qv[1]:
            return 0.05
        if x <= qv[2]:
            return 0.20
        if x <= qv[3]:
            return 0.40
        if x <= qv[4]:
            return 0.60
        if x <= qv[5]:
            return 0.80
        if x <= qv[6]:
            return 0.95
        return 0.99

    # ---------- score deltas ----------
    def _delta_from_ret_q(self, q: float) -> float:
        c = self.cfg
        if q <= 0.01:
            return c.delta_ret_q01
        if q <= 0.05:
            return c.delta_ret_q05
        if q <= 0.20:
            return c.delta_ret_q20
        if q <= 0.40:
            return c.delta_ret_q40
        if q <= 0.60:
            return c.delta_ret_q60
        if q <= 0.80:
            return c.delta_ret_q80
        if q <= 0.95:
            return c.delta_ret_q95
        return c.delta_ret_q99

    def _delta_from_vol_q(self, q: float) -> float:
        c = self.cfg
        if q <= 0.01:
            return c.delta_vol_q01
        if q <= 0.05:
            return c.delta_vol_q05
        if q <= 0.20:
            return c.delta_vol_q20
        if q <= 0.40:
            return c.delta_vol_q40
        if q <= 0.60:
            return c.delta_vol_q60
        if q <= 0.80:
            return c.delta_vol_q80
        if q <= 0.95:
            return c.delta_vol_q95
        return c.delta_vol_q99

    def _delta_from_p_smooth(self, p: float) -> Tuple[float, str]:
        """Risk delta with *in-band* linear interpolation.

        Keeps your original anchors but makes delta vary continuously inside bands:
          - [p2, p3): linearly from 0 -> delta_p_p2_p3
          - [p3, p4): linearly from delta_p_p2_p3 -> delta_p_p3_p4

        This reduces the "horizontal stripes" in r scatter plots.
        """
        c = self.cfg
        if not np.isfinite(p):
            return 0.0, "p=nan"
        if p < c.p1:
            return c.delta_p_lt_p1, "p<p1"
        if p < c.p2:
            return c.delta_p_p1_p2, "p1<=p<p2"
        if p < c.p3:
            # linear from 0 -> delta_p_p2_p3
            t = (p - c.p2) / max(1e-9, (c.p3 - c.p2))
            t = float(np.clip(t, 0.0, 1.0))
            return t * c.delta_p_p2_p3, "p2<=p<p3(smooth)"
        if p < c.p4:
            # linear from delta_p_p2_p3 -> delta_p_p3_p4
            t = (p - c.p3) / max(1e-9, (c.p4 - c.p3))
            t = float(np.clip(t, 0.0, 1.0))
            d = c.delta_p_p2_p3 + t * (c.delta_p_p3_p4 - c.delta_p_p2_p3)
            return d, "p3<=p<p4(smooth)"
        return c.delta_p_ge_p4, "p>=p4"

    def _risk_gain_cap(self, p_ms: float) -> float:
        """Gate positive boosts (Return/Liquidity) by risk.

        - p_ms >= p4: forbid positive boosts (cap=0)
        - p3 <= p_ms < p4: halve positive boosts (cap=0.5)
        - else: full boosts

        Negative adjustments are NEVER reduced.
        """
        c = self.cfg
        if not np.isfinite(p_ms):
            return 1.0
        if p_ms >= c.p4:
            return 0.0
        if p_ms >= c.p3:
            return 0.5
        return 1.0

    @staticmethod
    def _apply_gain_cap(delta: float, cap: float) -> float:
        """Keep penalties; cap only the positive part."""
        neg = min(delta, 0.0)
        pos = max(delta, 0.0) * cap
        return float(neg + pos)

    # ---------- main scoring ----------
    def score(self, pred_return_frac: float, pred_vol: float, p_ms: float) -> Dict[str, Any]:
        c = self.cfg

        ret_q = self._piecewise_quantile(pred_return_frac, self.ret_ref_quantiles)
        vol_q = self._piecewise_quantile(pred_vol, self.vol_ref_quantiles)

        d_p, risk_band = self._delta_from_p_smooth(p_ms)
        d_ret_raw = self._delta_from_ret_q(ret_q)
        d_vol_raw = self._delta_from_vol_q(vol_q)

        # (Upgrade A) Risk gate on positive boosts
        cap = self._risk_gain_cap(p_ms)
        d_ret = self._apply_gain_cap(d_ret_raw, cap)
        d_vol = self._apply_gain_cap(d_vol_raw, cap)

        r = c.r_base + d_p + d_ret + d_vol

        # hard caps
        if np.isfinite(p_ms) and p_ms >= c.r_hard_cap_p:
            r = min(r, c.r_hard_cap)

        if (
            np.isfinite(p_ms)
            and np.isfinite(pred_return_frac)
            and (p_ms >= c.joint_cap_p)
            and (pred_return_frac <= c.joint_cap_ret)
        ):
            r = min(r, c.r_joint_cap)

        # clamp
        r = float(np.clip(r, c.r_min, c.r_max))

        # leverage multiple L = 1/(1-r)
        L = float(1.0 / max(1e-9, (1.0 - r)))

        return {
            "r": r,
            "L": L,
            "risk_band": risk_band,
            "ret_q": float(ret_q),
            "vol_q": float(vol_q),
            "ret_band": f"ret_q={ret_q}",
            "liq_band": f"vol_q={vol_q}",
            "d_p": float(d_p),
            "d_ret": float(d_ret),
            "d_vol": float(d_vol),
            "d_ret_raw": float(d_ret_raw),
            "d_vol_raw": float(d_vol_raw),
            "gain_cap": float(cap),
        }

    def predict_r(self, pred_return_frac: np.ndarray, pred_vol: np.ndarray, p_ms: np.ndarray) -> np.ndarray:
        pr = np.asarray(pred_return_frac, dtype=float)
        pv = np.asarray(pred_vol, dtype=float)
        pm = np.asarray(p_ms, dtype=float)
        n = len(pr)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            out[i] = self.score(pr[i], pv[i], pm[i])["r"]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "vol_ref_quantiles": None if self.vol_ref_quantiles is None else self.vol_ref_quantiles.tolist(),
            "ret_ref_quantiles": None if self.ret_ref_quantiles is None else self.ret_ref_quantiles.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LeveragePolicy":
        cfg = LeveragePolicyConfig(**d.get("config", {}))
        pol = LeveragePolicy(cfg)

        vq = d.get("vol_ref_quantiles", None)
        if vq is not None:
            pol.vol_ref_quantiles = np.asarray(vq, dtype=float)

        rq = d.get("ret_ref_quantiles", None)
        if rq is not None:
            pol.ret_ref_quantiles = np.asarray(rq, dtype=float)

        return pol


# =========================
# Validation helpers — DOES NOT CHANGE r
# =========================

def safety_margin(r: np.ndarray, d_max: np.ndarray) -> np.ndarray:
    """Price-risk margin:

    SafetyMargin = (1 - r) - d_max

    r: broker funding ratio
    d_max: worst drawdown fraction (positive), e.g. 0.35 means -35%
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d_max, dtype=float)
    return (1.0 - r) - d


def liquidity_coverage_multiple(turnover: np.ndarray, r: np.ndarray, total_notional: np.ndarray) -> np.ndarray:
    """Broker-principal coverage multiple (LCM).

    BrokerPrincipal = r * TotalNotional
    LCM = Turnover / BrokerPrincipal

    turnover: ideally monetary turnover (e.g. HKD), not shares
    """
    t = np.asarray(turnover, dtype=float)
    r = np.asarray(r, dtype=float)
    n = np.asarray(total_notional, dtype=float)

    denom = r * n
    out = np.full_like(t, np.nan, dtype=float)
    m = np.isfinite(t) & np.isfinite(denom) & (denom > 0)
    out[m] = t[m] / denom[m]
    return out


def bucket_breach_rate(
    r: np.ndarray,
    breach: np.ndarray,
    bins: Tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0),
) -> Dict[str, Any]:
    r = np.asarray(r, dtype=float)
    b = np.asarray(breach, dtype=bool)

    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]:.2f},{bins[i+1]:.2f})")

    idx = np.digitize(r, bins, right=False) - 1
    idx = np.clip(idx, 0, len(labels) - 1)

    out = []
    for k, lab in enumerate(labels):
        mk = idx == k
        cnt = int(mk.sum())
        br = int(b[mk].sum()) if cnt > 0 else 0
        rate = (br / cnt) if cnt > 0 else np.nan
        out.append({"bucket": lab, "count": cnt, "breach": br, "breach_rate": rate})
    return {"bins": bins, "table": out}
