import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

TARGETS    = ["Tensile", "Youngs", "Hardness", "Buckling"]
UNITS      = {"Tensile": "GPa", "Youngs": "GPa", "Hardness": "HV", "Buckling": "N/m"}
FEAT_NAMES = ["Boron Nitride %", "Aluminium Oxide %"]
EXP_W      = 0.90
BN_MIN, BN_MAX = 2.5, 7.5
AO_MIN, AO_MAX = 2.5, 7.5
MODEL_PATH = "models/trained_models.pkl"


def load_data(theory_path, fea_path, exp_path):
    df_theory_raw = pd.read_excel(theory_path)
    ycol = [c for c in df_theory_raw.columns if "modulus" in str(c).lower()][0]
    df_theory = df_theory_raw[["Carbon Fiber", "Boron Nitride", "Aluminium Oxide",
                                "Tensile", ycol, "Hardness", "Buckling"]].copy()
    df_theory.columns = ["CF", "BN", "AO", "Tensile", "Youngs", "Hardness", "Buckling"]
    df_theory = df_theory.dropna()

    df_fea_raw = pd.read_excel(fea_path)
    tcols = [c for c in df_fea_raw.columns if "ensile" in str(c)]
    ftc   = next(c for c in tcols if df_fea_raw[c].dropna().mean() < 2)
    df_fea = df_fea_raw[["Carbon Fiber", "Boron Nitride", "Aluminium Oxide", ftc]].copy()
    df_fea.columns = ["CF", "BN", "AO", "Tensile"]
    df_fea = df_fea.dropna()

    df_exp_raw = pd.read_excel(exp_path, header=None)
    hr = next(i for i, row in df_exp_raw.iterrows() if "Carbon" in str(row.values))
    df_exp = df_exp_raw.iloc[hr+1:].reset_index(drop=True).dropna(how="all", axis=1)
    df_exp.columns = ["CF", "BN", "AO", "Tensile", "Youngs", "Hardness", "Buckling"]
    df_exp = df_exp.apply(pd.to_numeric, errors="coerce").dropna()

    return df_theory, df_fea, df_exp


def build_gpr(X, y):
    k = ConstantKernel(1.0) * Matern(length_scale=1.5, nu=2.5) + WhiteKernel(0.01)
    m = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=3,
                                 normalize_y=True, random_state=42)
    m.fit(X, y)
    return m


def _r2_gpr_loo(X, y):
    """Leave-one-out CV R² for GPR."""
    if len(X) < 3:
        return float("nan")
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        m = build_gpr(X[train_idx], y[train_idx])
        y_true.append(float(y[test_idx[0]]))
        y_pred.append(float(m.predict(X[test_idx])[0]))
    score = r2_score(y_true, y_pred)
    return round(float(score), 4)


def train_models(theory_path, fea_path, exp_path):
    os.makedirs("models", exist_ok=True)
    df_theory, df_fea, df_exp = load_data(theory_path, fea_path, exp_path)

    best_models = {}
    all_r2      = {}

    for prop in TARGETS:
        exp = df_exp[["BN", "AO", prop]].dropna()
        Xe  = exp[["BN", "AO"]].values
        ye  = exp[prop].values

        mdl_gpr = build_gpr(Xe, ye)
        r2_gpr  = _r2_gpr_loo(Xe, ye)

        all_r2[prop]      = {"GPR": round(r2_gpr, 4)}
        best_models[prop] = ("GPR", mdl_gpr)

    payload = {
        "models":   best_models,
        "all_r2":   all_r2,
        "exp_data": df_exp[["BN", "AO"] + TARGETS].to_dict(orient="records"),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    return payload


def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(bn, ao, payload):
    xi = np.array([[bn, ao]])
    results = {}
    for prop in TARGETS:
        name, mdl = payload["models"][prop]
        mu, std = mdl.predict(xi, return_std=True)
        results[prop] = {
            "value":       round(float(mu[0]), 5),
            "uncertainty": round(float(std[0]), 5),
            "unit":        UNITS[prop],
            "model":       "GPR",
            "r2":          payload["all_r2"][prop]["GPR"],
        }
    return results


def inverse_predict(targets, payload, weights=None, grid_size=150):
    bn_vals = np.linspace(BN_MIN, BN_MAX, grid_size)
    ao_vals = np.linspace(AO_MIN, AO_MAX, grid_size)
    BN_grid, AO_grid = np.meshgrid(bn_vals, ao_vals)
    X_grid = np.column_stack([BN_grid.ravel(), AO_grid.ravel()])

    active_props = [p for p, v in targets.items() if v is not None]
    if not active_props:
        raise ValueError("At least one target property must be specified.")

    if weights is None:
        weights = {p: 1.0 for p in active_props}

    pred_matrix = {}
    for prop in active_props:
        name, mdl = payload["models"][prop]
        mu, _ = mdl.predict(X_grid, return_std=True)
        pred_matrix[prop] = mu

    total_error = np.zeros(len(X_grid))
    for prop in active_props:
        pmin = pred_matrix[prop].min()
        pmax = pred_matrix[prop].max()
        prop_range = max(pmax - pmin, 1e-9)
        norm_err = (pred_matrix[prop] - targets[prop]) / prop_range
        total_error += weights.get(prop, 1.0) * norm_err ** 2

    best_idx = int(np.argmin(total_error))
    best_bn  = float(X_grid[best_idx, 0])
    best_ao  = float(X_grid[best_idx, 1])

    xi = np.array([[best_bn, best_ao]])
    achieved = {}
    for prop in TARGETS:
        name, mdl = payload["models"][prop]
        mu, std = mdl.predict(xi, return_std=True)
        achieved[prop] = {
            "value":       round(float(mu[0]), 5),
            "uncertainty": round(float(std[0]), 5),
            "unit":        UNITS[prop],
            "model":       "GPR",
            "r2":          payload["all_r2"][prop]["GPR"],
        }

    per_prop_error = {
        prop: round(abs(achieved[prop]["value"] - targets[prop]), 6)
        for prop in active_props
    }

    return {
        "bn":             round(best_bn, 4),
        "ao":             round(best_ao, 4),
        "achieved":       achieved,
        "per_prop_error": per_prop_error,
        "total_error":    round(float(total_error[best_idx]), 8),
        "error_surface":  total_error.reshape(grid_size, grid_size).tolist(),
        "grid_bn":        bn_vals.tolist(),
        "grid_ao":        ao_vals.tolist(),
    }


def get_pd_curves(bn, ao, payload):
    sweep = np.linspace(BN_MIN, BN_MAX, 40)
    curves = {}
    for prop in TARGETS:
        name, mdl = payload["models"][prop]
        bn_sw = np.column_stack([sweep, np.full(40, ao)])
        ao_sw = np.column_stack([np.full(40, bn), sweep])
        mu_bn, sd_bn = mdl.predict(bn_sw, return_std=True)
        mu_ao, sd_ao = mdl.predict(ao_sw, return_std=True)
        curves[prop] = {
            "sweep":  sweep.tolist(),
            "bn_mu":  mu_bn.tolist(), "bn_sd": sd_bn.tolist(),
            "ao_mu":  mu_ao.tolist(), "ao_sd": sd_ao.tolist(),
        }
    return curves
