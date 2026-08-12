"""
models.py
---------
Treina 4 modelos para srag_taxa_casos com:
  - Busca de hiperparâmetros por RandomizedSearchCV + TimeSeriesSplit de painel
  - Avaliação no hold-out temporal configurado em periods.yaml

MODELOS:
  1. Poisson GLM regularizado  (linear, interpretável)
  2. Random Forest              (não-linear, robusto a colinearidade)
  3. Histogram Gradient Boosting com perda Poisson (sklearn, sem dep. extra)
  4. XGBoost                    (se instalado: pip install xgboost)

SEPARAÇÃO TREINO / TESTE:
  Puramente cronológica (sem embaralhamento).
  Teste = últimos `test_holdout_months` meses com dado observado.
  Busca de hiperparâmetros usa TimeSeriesSplit DENTRO do conjunto de treino.

BUSCA DE HIPERPARÂMETROS:
  RandomizedSearchCV — não exaustiva. Número de tentativas e folds
  configurados em config/periods.yaml (hparam_n_iter, hparam_cv_folds).
"""

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import loguniform, uniform
from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_poisson_deviance)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# XGBoost é opcional
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT         = Path(__file__).resolve().parents[1]
PROCESSED    = ROOT / "data" / "processed"
OUT_TABLES   = ROOT / "outputs" / "tables"
OUT_FIGS     = ROOT / "outputs" / "figures"

ID_COLS = ["cod_mun", "date", "municipio"]


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        vcfg = yaml.safe_load(f)
    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)
    return vcfg, pcfg


def get_feature_cols(panel, vcfg):
    target  = vcfg.get("model_target", "srag_taxa_casos")
    exclude = set(vcfg.get("model_features_exclude", [])) | set(ID_COLS) | {target}
    feats   = [c for c in panel.columns if c not in exclude and panel[c].dtype != object]
    return target, feats


# ══════════════════════════════════════════════════════════════════════════════
# Imputação
# ══════════════════════════════════════════════════════════════════════════════

def impute_features(df, feature_cols):
    df = df.sort_values(["cod_mun", "date"]).copy()
    for col in feature_cols:
        if col == "has_uti":
            df[col] = df[col].fillna(0)
            continue
        df[col] = df.groupby("cod_mun")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TimeSeriesSplit para painel (divide por data, não por linha)
# ══════════════════════════════════════════════════════════════════════════════

class PanelTimeSeriesSplit:
    """
    Validação cruzada temporal para painéis (município × mês).
    Divide nas DATAS únicas — garante que todos os municípios de um mesmo
    mês ficam no mesmo fold, evitando vazamento de informação futura.

    Compatível com a interface de CV do sklearn (fit/split/get_n_splits).
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """groups deve ser o array de datas (strings ou timestamps) das linhas."""
        if groups is None:
            raise ValueError("PanelTimeSeriesSplit requer groups=dates_array")
        unique_dates = np.sort(np.unique(groups))
        tss = TimeSeriesSplit(n_splits=self.n_splits)
        for tr_idx, val_idx in tss.split(unique_dates):
            tr_dates  = set(unique_dates[tr_idx])
            val_dates = set(unique_dates[val_idx])
            yield (np.where([d in tr_dates  for d in groups])[0],
                   np.where([d in val_dates for d in groups])[0])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ══════════════════════════════════════════════════════════════════════════════
# Estimador customizado: Poisson GLM com escala interna (target sub-1)
# Encapsula StandardScaler + escala do target → funciona dentro do CV
# ══════════════════════════════════════════════════════════════════════════════

class ScaledPoissonGLM(BaseEstimator, RegressorMixin):
    """
    PoissonRegressor do sklearn com escala automática do target.
    PoissonRegressor foi projetado para contagens; com target < 1 tem
    instabilidade numérica. Esta classe escala y pelo inverso da mediana
    dos positivos em cada fold de treino, garantindo que o fit seja estável
    independentemente da escala absoluta do alvo.
    """
    def __init__(self, alpha=0.1, max_iter=3000):
        self.alpha    = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        yp        = np.clip(y, 0, None)
        med       = np.median(yp[yp > 0]) if (yp > 0).any() else 1.0
        self.scale_  = (1.0 / med) if med < 0.1 else 1.0
        self.scaler_ = StandardScaler()
        Xs           = self.scaler_.fit_transform(X)
        self.model_  = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.model_.fit(Xs, yp * self.scale_)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X)) / self.scale_

    @property
    def coef_(self):
        return self.model_.coef_


# ══════════════════════════════════════════════════════════════════════════════
# Espaços de hiperparâmetros
# ══════════════════════════════════════════════════════════════════════════════

def get_param_distributions():
    dists = {
        "Poisson_GLM": (
            ScaledPoissonGLM(),
            {"alpha": loguniform(1e-4, 20),
             "max_iter": [1000, 2000, 5000]}
        ),
        "Random_Forest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators":    [200, 400, 600, 800],
             "max_depth":       [6, 8, 12, 16, None],
             "min_samples_leaf":[1, 2, 3, 5, 10],
             "max_features":    ["sqrt", "log2", 0.3, 0.5],
             "min_samples_split":[2, 5, 10]}
        ),
        "GradientBoosting": (
            HistGradientBoostingRegressor(loss="poisson", random_state=42),
            {"max_depth":          [3, 4, 5, 6, 8],
             "learning_rate":      loguniform(0.01, 0.3),
             "max_iter":           [200, 400, 600, 800],
             "min_samples_leaf":   [5, 10, 20, 40],
             "l2_regularization":  loguniform(1e-4, 1.0),
             "max_leaf_nodes":     [15, 31, 63, None]}
        ),
    }
    if HAS_XGBOOST:
        dists["XGBoost"] = (
            XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=1.5,
                tree_method="hist",
                random_state=42,
                verbosity=0,
            ),
            {"n_estimators":     [200, 400, 600, 800],
             "max_depth":        [3, 4, 5, 6],
             "learning_rate":    loguniform(0.01, 0.3),
             "subsample":        uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.6, 0.4),
             "reg_alpha":        loguniform(1e-4, 10),
             "reg_lambda":       loguniform(0.1, 10),
             "min_child_weight": [1, 3, 5, 10]}
        )
    else:
        print("  XGBoost não instalado — pulando. (pip install xgboost)")
    return dists


# ══════════════════════════════════════════════════════════════════════════════
# Busca de hiperparâmetros
# ══════════════════════════════════════════════════════════════════════════════

def search_hyperparams(Xtr, ytr, dates_tr, pcfg, n_jobs):
    n_iter   = pcfg.get("hparam_n_iter",   50)
    n_folds  = pcfg.get("hparam_cv_folds",  5)
    scoring  = pcfg.get("hparam_scoring", "neg_mean_absolute_error")
    dists    = get_param_distributions()
    cv       = PanelTimeSeriesSplit(n_splits=n_folds)
    best_estimators = {}
    search_results  = []

    for name, (estimator, param_dist) in dists.items():
        print(f"  [{name}] RandomizedSearchCV  n_iter={n_iter}  cv={n_folds} folds ...",
              flush=True)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=42,
            refit=True,
            error_score="raise",
        )
        search.fit(Xtr, ytr, groups=dates_tr)
        best_estimators[name] = search.best_estimator_
        search_results.append({
            "modelo":          name,
            "melhor_score_cv": round(-search.best_score_, 8),   # MAE (positivo)
            **{f"param_{k}": v for k, v in search.best_params_.items()},
        })
        print(f"    → melhor MAE-CV: {-search.best_score_:.6f}  |  "
              f"params: {search.best_params_}")

    return best_estimators, pd.DataFrame(search_results)


# ══════════════════════════════════════════════════════════════════════════════
# Métricas
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, label):
    y_pred_clip = np.clip(y_pred, 1e-10, None)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    try:
        pd_val = mean_poisson_deviance(np.clip(y_true, 0, None), y_pred_clip)
    except Exception:
        pd_val = np.nan
    mask_pos = y_true > 0
    mape = ((np.abs(y_pred[mask_pos] - y_true[mask_pos]) / y_true[mask_pos]).mean() * 100
            if mask_pos.sum() > 0 else np.nan)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.where(denom > 0, np.abs(y_pred - y_true) / denom * 100, 0.0).mean()
    pr = pearsonr(y_true, y_pred)[0]   if len(y_true) > 2 else np.nan
    sr = spearmanr(y_true, y_pred)[0]  if len(y_true) > 2 else np.nan
    def r(v, d=6): return round(float(v), d) if pd.notna(v) else np.nan
    return {"modelo": label, "MAE": r(mae), "RMSE": r(rmse), "R2": r(r2, 4),
            "MAPE_pct": r(mape, 2), "sMAPE_pct": r(smape, 2),
            "Pearson_r": r(pr, 4), "Spearman_rho": r(sr, 4),
            "Poisson_deviance": r(pd_val)}


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_models():
    vcfg, pcfg = load_config()
    n_jobs = pcfg.get("n_processadores", 4)

    panel  = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    target, feature_cols = get_feature_cols(panel, vcfg)
    mp = panel[panel[target].notna()].copy()

    if pcfg.get("apenas_municipios_com_casos", False):
        muns = mp.groupby("cod_mun")[target].max().pipe(lambda s: s[s > 0]).index
        antes = mp["cod_mun"].nunique()
        mp = mp[mp["cod_mun"].isin(muns)]
        print(f"  Filtro ativo: {len(muns)}/{antes} municípios com casos")

    mp = impute_features(mp, feature_cols)

    n_hold = pcfg.get("test_holdout_months", 6)
    test_dates = set(sorted(mp["date"].unique())[-n_hold:])
    train = mp[~mp["date"].isin(test_dates)].copy()
    test  = mp[ mp["date"].isin(test_dates)].copy()

    Xtr, ytr = train[feature_cols].values, train[target].values
    Xte, yte = test[feature_cols].values,  test[target].values
    dates_tr  = train["date"].astype(str).values

    print(f"  Alvo: {target}  |  Features: {len(feature_cols)}  |  "
          f"Treino: {len(train)} obs  |  Teste: {len(test)} obs")
    print(f"  Processadores: {n_jobs}  |  n_iter: {pcfg.get('hparam_n_iter',50)}  |  "
          f"cv_folds: {pcfg.get('hparam_cv_folds',5)}")

    # ── Busca de hiperparâmetros ───────────────────────────────────────────
    print(f"\n  Buscando hiperparâmetros (TimeSeriesSplit de painel)...")
    best_estimators, search_df = search_hyperparams(Xtr, ytr, dates_tr, pcfg, n_jobs)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(OUT_TABLES / "hparam_search_results.csv", index=False)

    # ── Avaliação no teste ─────────────────────────────────────────────────
    results = []
    preds   = {"municipio": test["municipio"].values, "cod_mun": test["cod_mun"].values,
                "date": test["date"].values, "y_true": yte}

    for name, est in best_estimators.items():
        pred = est.predict(Xte)
        results.append(evaluate(yte, pred, name))
        preds[f"pred_{name}"] = pred

        # Importâncias / coeficientes
        if hasattr(est, "coef_"):
            (pd.DataFrame({"variavel": feature_cols, "coeficiente": est.coef_})
               .sort_values("coeficiente", key=abs, ascending=False)
               .to_csv(OUT_TABLES / f"{name}_coeficientes.csv", index=False))
        if hasattr(est, "feature_importances_"):
            (pd.DataFrame({"variavel": feature_cols, "importancia": est.feature_importances_})
               .sort_values("importancia", ascending=False)
               .to_csv(OUT_TABLES / f"{name}_importancia.csv", index=False))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_TABLES / "metricas_modelos_global.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT_TABLES / "predicoes_teste.csv", index=False)

    # ── Persistir modelos treinados em disco para reutilização nos plots ──────
    import joblib
    OUT_MODELS = ROOT / "outputs" / "models"
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for name, est in best_estimators.items():
        joblib.dump(est, OUT_MODELS / f"{name}.pkl")
    # Salvar também feature_cols para garantir consistência
    pd.Series(feature_cols).to_csv(OUT_MODELS / "feature_cols.csv", index=False, header=False)
    print(f"  Modelos salvos em {OUT_MODELS}/")

    return results_df, pd.DataFrame(preds), feature_cols


if __name__ == "__main__":
    results_df, preds_df, _ = run_models()
    pd.set_option("display.width", 200)
    print("\n=== Métricas globais (teste) ===")
    print(results_df.to_string(index=False))
