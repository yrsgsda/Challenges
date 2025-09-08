from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Literal
import numpy as np
import pandas as pd

Mode = Literal['sequential', 'pooled']

@dataclass
class TextSignalKalmanDoc:
    signal_cols: List[str]
    date_col: str = "date"
    id_col: str = "id"
    novelty_col: Optional[str] = None
    skill_col: Optional[str] = None
    signal_weights: Optional[List[float]] = None
    phi: float = 1.0
    Q: float = 0.05
    R: float = 1.0
    business_day_grid: bool = True
    adapt_measurement_noise: bool = True
    precision_betas: Dict[str, float] = field(default_factory=lambda: {
        "intercept": 0.0,
        "novelty": 1.0,
        "skill": 0.5,
    })
    precision_bounds: Tuple[float, float] = (0.5, 3.0)
    cap_daily_precision: Optional[float] = 6.0
    mode: Mode = "sequential"
    
    def _validate(self, df: pd.DataFrame):
        base_cols = [self.date_col, self.id_col] + self.signal_cols
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if self.signal_weights is not None and len(self.signal_weights) != len(self.signal_cols):
            raise ValueError("signal_weights must match length of signal_cols")
        if not (0 < self.phi <= 1.0):
            raise ValueError("phi must be in (0, 1]")
        if self.Q < 0 or self.R <= 0:
            raise ValueError("Q must be >= 0 and R must be > 0")
        if self.mode not in ("sequential", "pooled"):
            raise ValueError("mode must be 'sequential' or 'pooled'")
    
    def _combine_doc_signal(self, row: pd.Series) -> float:
        if self.signal_weights is None:
            w = np.ones(len(self.signal_cols)) / len(self.signal_cols)
        else:
            w = np.asarray(self.signal_weights, dtype=float)
            w = w / w.sum()
        val = 0.0
        for j, col in enumerate(self.signal_cols):
            val += w[j] * float(row[col])
        return float(val)
    
    def _doc_precision_multiplier(self, row: pd.Series) -> float:
        if not self.adapt_measurement_noise:
            return 1.0
        b = self.precision_betas
        novelty = float(row[self.novelty_col]) if (self.novelty_col and pd.notna(row.get(self.novelty_col))) else 0.0
        skill = float(row[self.skill_col]) if (self.skill_col and pd.notna(row.get(self.skill_col))) else 1.0
        lin = b.get("intercept", 0.0) + b.get("novelty", 0.0) * novelty + b.get("skill", 0.0) * (skill - 1.0)
        w = float(np.exp(lin))
        lo, hi = self.precision_bounds
        return float(np.clip(w, lo, hi))
    
    def _date_grid(self, s: pd.Series) -> pd.DatetimeIndex:
        freq = "B" if self.business_day_grid else "D"
        return pd.date_range(s.min(), s.max(), freq=freq)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.id_col, self.date_col]).reset_index(drop=True)
        
        outputs = []
        for gid, g in df.groupby(self.id_col):
            grid = self._date_grid(g[self.date_col])
            day_groups = {d: sub for d, sub in g.groupby(self.date_col)}
            theta = 0.0
            P = self.R * 10.0  # diffuse-ish prior
            
            rows_out = []
            for d in grid:
                theta_pred = self.phi * theta
                P_pred = (self.phi ** 2) * P + self.Q
                
                docs_today = day_groups.get(d, None)
                if docs_today is None or len(docs_today) == 0:
                    theta, P = theta_pred, P_pred
                    rows_out.append({
                        self.date_col: d,
                        self.id_col: gid,
                        "signal_kalman": theta,
                        "docs": 0,
                        "kalman_gain": 0.0,
                        "daily_precision": 0.0,
                        "obs_y": float("nan")
                    })
                else:
                    if self.mode == "sequential":
                        theta_day, P_day = theta_pred, P_pred
                        daily_W = 0.0
                        last_K = 0.0
                        ys, ws = [], []
                        for _, row in docs_today.iterrows():
                            y_doc = self._combine_doc_signal(row)
                            w_doc = self._doc_precision_multiplier(row)
                            Rt = self.R / w_doc
                            K = P_day / (P_day + Rt)
                            theta_day = theta_day + K * (y_doc - theta_day)
                            P_day = (1.0 - K) * P_day
                            last_K = float(K)
                            daily_W += w_doc
                            ys.append(y_doc); ws.append(w_doc)
                        theta, P = theta_day, P_day
                        if self.cap_daily_precision is not None:
                            daily_W = float(min(daily_W, self.cap_daily_precision))
                        obs_y_diag = (np.dot(ws, ys) / max(np.sum(ws), 1e-12)) if np.sum(ws) > 0 else float("nan")
                        rows_out.append({
                            self.date_col: d,
                            self.id_col: gid,
                            "signal_kalman": theta,
                            "docs": int(len(docs_today)),
                            "kalman_gain": last_K,
                            "daily_precision": daily_W,
                            "obs_y": obs_y_diag
                        })
                    else:  # pooled
                        ys, ws = [], []
                        for _, row in docs_today.iterrows():
                            y_doc = self._combine_doc_signal(row)
                            w_doc = self._doc_precision_multiplier(row)
                            ys.append(y_doc); ws.append(w_doc)
                        W = float(np.sum(ws))
                        if self.cap_daily_precision is not None:
                            W = float(min(W, self.cap_daily_precision))
                        if W <= 0:
                            theta, P = theta_pred, P_pred
                            K = 0.0
                            y_bar = float("nan")
                        else:
                            y_bar = float(np.dot(ws, ys) / W)
                            Rt_day = self.R / W
                            K = P_pred / (P_pred + Rt_day)
                            theta = theta_pred + K * (y_bar - theta_pred)
                            P = (1.0 - K) * P_pred
                        rows_out.append({
                            self.date_col: d,
                            self.id_col: gid,
                            "signal_kalman": theta,
                            "docs": int(len(docs_today)),
                            "kalman_gain": float(K),
                            "daily_precision": float(W),
                            "obs_y": y_bar
                        })
            outputs.append(pd.DataFrame(rows_out))
        
        res = pd.concat(outputs, axis=0, ignore_index=True)
        res = res.sort_values([self.id_col, self.date_col]).reset_index(drop=True)
        return res
