
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
import numpy as np
import pandas as pd

Mode = Literal['sequential', 'pooled']

@dataclass
class TextSignalKalmanDocV2:
    signal_cols: List[str]
    date_col: str = "date"
    id_col: str = "id"
    novelty_col: Optional[str] = None
    skill_col: Optional[str] = None
    signal_weights: Optional[List[float]] = None
    state_half_life: float = 15.0
    state_var: float = 0.25
    R: float = 1.0
    business_day_grid: bool = True
    use_precision_features: bool = True
    beta_novelty: float = 1.0
    beta_skill: float = 0.5
    w_min: float = 0.5
    w_max: float = 3.0
    mode: Mode = "sequential"
    
    def _validate(self, df: pd.DataFrame):
        base_cols = [self.date_col, self.id_col] + self.signal_cols
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if self.signal_weights is not None and len(self.signal_weights) != len(self.signal_cols):
            raise ValueError("signal_weights must match length of signal_cols")
        if self.state_half_life <= 0:
            raise ValueError("state_half_life must be positive")
        if self.state_var < 0 or self.R <= 0:
            raise ValueError("state_var must be >=0 and R must be >0")
    
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
        if not self.use_precision_features:
            return 1.0
        novelty = float(row[self.novelty_col]) if (self.novelty_col and pd.notna(row.get(self.novelty_col))) else 0.0
        skill = float(row[self.skill_col]) if (self.skill_col and pd.notna(row.get(self.skill_col))) else 1.0
        lin = self.beta_novelty * novelty + self.beta_skill * (skill - 1.0)
        w = float(np.exp(lin))
        return float(np.clip(w, self.w_min, self.w_max))
    
    def _date_grid(self, series: pd.Series) -> pd.DatetimeIndex:
        freq = "B" if self.business_day_grid else "D"
        return pd.date_range(series.min(), series.max(), freq=freq)
    
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
            P = max(self.state_var, 1e-8) * 10.0
            prev_date = None
            
            rows_out = []
            for d in grid:
                if prev_date is None:
                    dt = 1.0
                else:
                    dt = float((d - prev_date).days) if not self.business_day_grid else 1.0
                    if dt <= 0:
                        dt = 1.0
                phi_dt = 2.0 ** (-dt / self.state_half_life)
                Q_dt = self.state_var * (1.0 - phi_dt**2)
                
                theta_pred = phi_dt * theta
                P_pred = (phi_dt**2) * P + Q_dt
                
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
                        obs_y_diag = (np.dot(ws, ys) / max(np.sum(ws), 1e-12)) if np.sum(ws) > 0 else float("nan")
                        rows_out.append({
                            self.date_col: d,
                            self.id_col: gid,
                            "signal_kalman": theta,
                            "docs": int(len(docs_today)),
                            "kalman_gain": last_K,
                            "daily_precision": float(daily_W),
                            "obs_y": obs_y_diag
                        })
                    else:
                        ys, ws = [], []
                        for _, row in docs_today.iterrows():
                            y_doc = self._combine_doc_signal(row)
                            w_doc = self._doc_precision_multiplier(row)
                            ys.append(y_doc); ws.append(w_doc)
                        W = float(np.sum(ws))
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
                
                prev_date = d
            
            outputs.append(pd.DataFrame(rows_out))
        
        res = pd.concat(outputs, axis=0, ignore_index=True)
        res = res.sort_values([self.id_col, self.date_col]).reset_index(drop=True)
        return res
