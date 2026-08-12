"""
run_experiments.py — Orquestrador de experimentos
==================================================
Cruza todas as combinações definidas em config/experiments.yaml e executa
cada cenário com todos os modelos implementados + busca de hiperparâmetros.

GRADE: 2 (temporal) × 2 (municípios) × 2 (features) = 8 cenários
       × N modelos com busca = N×8 experimentos

USO:
  python run_experiments.py                    # todos os experimentos
  python run_experiments.py --fast             # modo rápido (n_iter=5)
  python run_experiments.py --scenario 3       # só o cenário 3

SAÍDAS (em outputs/experiments/):
  summary_all_experiments.csv                  # tabela mestre para artigo
  best_models_per_scenario.csv                 # melhores por cenário
  feature_importance_heatmap.png               # importâncias por cenário
  scenarios_comparison.png                     # comparação visual
  scenario_<ID>/                               # pasta por experimento
    metrics.csv
    feature_importance.csv
    mrmr_report.csv
    best_params.csv
    model_*.pkl
"""

import sys
import json
import yaml
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import loguniform, uniform, pearsonr, spearmanr

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import (RandomizedSearchCV, KFold,
                                     TimeSeriesSplit, cross_val_score)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT = Path(__file__).resolve().parents[0]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_sets import get_feature_cols, EXCLUDED_ALWAYS
from seasonal_grouping import build_seasonal_panel, plot_humidity_profile
# mRMR removido — substituído por filtragem de correlação
from models import ScaledPoissonGLM, PanelTimeSeriesSplit, impute_features

PROCESSED  = ROOT / "data" / "processed"
OUT_BASE   = ROOT / "outputs" / "experiments"


# ══════════════════════════════════════════════════════════════════════════════
# Configuração
# ══════════════════════════════════════════════════════════════════════════════

def load_configs():
    with open(ROOT / "config" / "experiments.yaml", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        per_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        var_cfg = yaml.safe_load(f)
    return exp_cfg, per_cfg, var_cfg


# ══════════════════════════════════════════════════════════════════════════════
# Definição dos modelos e espaços de busca
# ══════════════════════════════════════════════════════════════════════════════

def get_regressors(target_max: float):
    """Escolhe perda adequada: Poisson para contagens, squared para proporções."""
    is_proportion = target_max <= 1.01
    regs = {
        "Poisson_GLM": (
            ScaledPoissonGLM(),
            {"alpha": loguniform(1e-4, 20), "max_iter": [1000, 2000, 5000]}
        ),
        "Random_Forest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators":     [200, 400, 600],
             "max_depth":        [4, 6, 8, 12, None],
             "min_samples_leaf": [1, 2, 3, 5],
             "max_features":     ["sqrt", "log2", 0.3, 0.5]}
        ),
        "GradientBoosting": (
            HistGradientBoostingRegressor(
                loss="squared_error" if is_proportion else "poisson",
                random_state=42),
            {"max_depth":         [3, 4, 5, 6],
             "learning_rate":     loguniform(0.01, 0.3),
             "max_iter":          [200, 400, 600],
             "min_samples_leaf":  [5, 10, 20],
             "l2_regularization": loguniform(1e-4, 1.0)}
        ),
    }
    if HAS_XGBOOST:
        regs["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror" if is_proportion else "reg:tweedie",
                tweedie_variance_power=1.5,
                tree_method="hist", random_state=42, verbosity=0),
            {"n_estimators":     [200, 400, 600],
             "max_depth":        [3, 4, 5, 6],
             "learning_rate":    loguniform(0.01, 0.3),
             "subsample":        uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.6, 0.4),
             "reg_alpha":        loguniform(1e-4, 10)}
        )
    return regs


# ══════════════════════════════════════════════════════════════════════════════
# Métricas
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, label) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    pr   = pearsonr(y_true, y_pred)[0]  if len(y_true) > 2 else np.nan
    sr   = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    mask = y_true > 0
    mape = ((np.abs(y_pred[mask]-y_true[mask])/y_true[mask]).mean()*100
            if mask.sum() > 0 else np.nan)
    def r(v, d=5): return round(float(v), d) if pd.notna(v) else np.nan
    return {"modelo": label, "MAE": r(mae), "RMSE": r(rmse), "R2": r(r2, 4),
            "MAPE_pct": r(mape, 2), "Pearson_r": r(pr, 4), "Spearman_rho": r(sr, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Construção do painel para cada cenário
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_panel(scenario: dict, exp_cfg: dict, per_cfg: dict):
    """Lê o painel processado e aplica os filtros do cenário."""
    panel = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    target = per_cfg.get("model_target", "srag_taxa_casos")

    # Features disponíveis para o cenário
    feature_cols = get_feature_cols(panel.columns.tolist(), scenario["feature_set"])
    # Garantir que o target não entra como feature
    feature_cols = [c for c in feature_cols if c != target and c not in EXCLUDED_ALWAYS]

    # Filtrar à janela com alvo observado
    panel_obs = panel[panel[target].notna()].copy()

    # Excluir anos do período pandêmico (configúravel em periods.yaml)
    exclude_years = per_cfg.get("exclude_years", [])
    if exclude_years:
        year_col = pd.to_datetime(panel_obs["date"]).dt.year
        n_before = len(panel_obs)
        panel_obs = panel_obs[~year_col.isin(exclude_years)].copy()
        print(f"  Anos excluídos {exclude_years}: {n_before}#{len(panel_obs)} linhas")

    # Imputar features ANTES do filtro de municípios (usa todos para imputação)
    panel_obs = impute_features(panel_obs, feature_cols)

    # Filtro de municípios
    if scenario["municipality_filter"] == "cases_only":
        muns = panel_obs.groupby("cod_mun")[target].max()
        muns = muns[muns > 0].index
        panel_obs = panel_obs[panel_obs["cod_mun"].isin(muns)].copy()

    # Agrupamento sazonal (se aplicável)
    if scenario["temporal_grouping"] == "seasonal_humidity":
        panel_obs, month_to_phase, phase_labels = build_seasonal_panel(
            panel_obs, target, feature_cols, exp_cfg.get("seasonal", {})
        )
        scenario["_month_to_phase"] = month_to_phase
        scenario["_phase_labels"]   = phase_labels
    else:
        month_to_phase = None

    return panel_obs, target, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# Execução de um experimento
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(scenario_id: str, scenario: dict,
                          exp_cfg: dict, per_cfg: dict,
                          fast_mode: bool = False) -> pd.DataFrame:
    """
    Executa um único cenário (todos os modelos + busca + mRMR).
    Retorna DataFrame com métricas de todos os modelos deste cenário.
    """
    out_dir = OUT_BASE / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Cenário {scenario_id}: {scenario}")
    print(f"{'─'*60}")

    # Salvar definição do cenário
    with open(out_dir / "scenario.json", "w") as f:
        json.dump({k: v for k, v in scenario.items()
                   if not k.startswith("_")}, f, indent=2)

    panel, target, feature_cols = build_scenario_panel(scenario, exp_cfg, per_cfg)

    n_jobs = per_cfg.get("n_processadores", 4)
    n_iter = exp_cfg.get("fast_n_iter", 5) if fast_mode else per_cfg.get("hparam_n_iter", 50)
    mrmr_cfg = exp_cfg.get("mrmr", {})
    n_select  = mrmr_cfg.get("n_features_to_select", 8)
    use_mrmr  = mrmr_cfg.get("enabled", True)

    # Split treino/teste
    is_temporal = scenario["temporal_grouping"] == "monthly"
    if is_temporal:
        n_hold     = per_cfg.get("test_holdout_months", 6)
        test_dates = set(sorted(panel["date"].unique())[-n_hold:])
        train = panel[~panel["date"].isin(test_dates)].copy()
        test  = panel[ panel["date"].isin(test_dates)].copy()
        dates_tr = train["date"].astype(str).values
        cv = PanelTimeSeriesSplit(n_splits=per_cfg.get("hparam_cv_folds", 5))
        cv_groups = dates_tr
        cv_kwargs  = {"groups": dates_tr}
    else:
        # Sazonal: split TEMPORAL por ano — último ano como teste.
        # Aleatório seria errado: o lag de 2022 usa dados de 2021;
        # se 2021 cair no teste e 2022 no treino, há leakage do lag.
        # Separar por year garante que o modelo nunca vê o futuro.
        year_col = "year" if "year" in panel.columns else None
        if year_col is None:
            panel["year"] = pd.to_datetime(panel["date"]).dt.year
            year_col = "year"
        last_year  = panel[year_col].max()
        train = panel[panel[year_col] < last_year].copy()
        test  = panel[panel[year_col] == last_year].copy()
        # CV dentro do treino: KFold estratificado por ano
        # (KFold aleatório é aceitável DENTRO do treino porque o split
        # treino/validação já respeita a fronteira temporal treino < teste)
        cv = KFold(n_splits=min(5, panel[year_col].nunique() - 1),
                   shuffle=True, random_state=42)
        cv_groups  = None
        cv_kwargs  = {}

    X_tr = train[feature_cols].values
    y_tr = train[target].values
    X_te = test[feature_cols].values
    y_te = test[target].values

    print(f"  Treino: {len(train)} obs  |  Teste: {len(test)} obs  "
          f"|  Features disponíveis: {len(feature_cols)}")

    # ── Seleção de features: obrigatórias + redução por correlação ─────────
    #
    # GRUPO 1 — OBRIGATÓRIAS (sempre entram, sem filtro):
    #   • Temporais/autorregressivas: lags, baseline, trend, sin/cos, covid
    #   • Climáticas/meteorológicas: todas as meteo_* e fogo
    #   Razão: são o objeto científico do estudo. Eliminá-las tornaria a
    #   simulação de mudanças climáticas impossível.
    #
    # GRUPO 2 — OPCIONAIS (sujeitas à redução por correlação):
    #   • Cobertura do solo (MapBiomas): farming, forest, cerrado, urban, water
    #   • Variáveis sociais: PIB, CadÚnico, CAGED, APS, demografia
    #   Razão: há colinearidade alta entre algumas (ex: farming ↔ forest
    #   são quase complementares). Remover as redundantes melhora a
    #   estabilidade numérica sem perder informação.
    #
    # MÉTODO: para cada par de features opcionais com |ρ| > threshold,
    #   remove a que tem menor |correlação com o target|.

    MANDATORY_ALWAYS = [
        # Temporais (SEM lags — removidos a pedido do usuário)
        "baseline_srag_municipio", "year_trend",
        "month_sin", "month_cos", "fase_covid",
        # Climáticas / meteorológicas
        "meteo_temp_media", "meteo_temp_min", "meteo_temp_max",
        "meteo_temp_amplitude", "meteo_precipitacao", "meteo_umidade",
        "fogo_pct_territorio",
    ]

    corr_threshold = exp_cfg.get("feature_selection", {}).get(
        "correlation_threshold", 0.85)

    mandatory  = [f for f in MANDATORY_ALWAYS if f in feature_cols]
    optional   = [f for f in feature_cols if f not in MANDATORY_ALWAYS]

    # Redução por correlação nas features opcionais
    if optional:
        X_opt = pd.DataFrame(X_tr, columns=feature_cols)[optional]
        y_ser = pd.Series(y_tr)
        corr_target = X_opt.corrwith(y_ser).abs()
        corr_matrix = X_opt.corr().abs()

        kept = list(optional)
        for i, fi in enumerate(optional):
            if fi not in kept:
                continue
            for fj in optional[i+1:]:
                if fj not in kept:
                    continue
                if corr_matrix.loc[fi, fj] > corr_threshold:
                    # Remove a menos correlacionada com o target
                    drop = fi if corr_target.get(fi, 0) < corr_target.get(fj, 0) else fj
                    if drop in kept:
                        kept.remove(drop)

        removed = [f for f in optional if f not in kept]
        if removed:
            print(f"  Redução por correlação (|ρ|>{corr_threshold}): "
                  f"removidas {removed}")
        optional_sel = kept
    else:
        optional_sel = []

    selected_features = mandatory + optional_sel
    sel_idx   = [feature_cols.index(f) for f in selected_features]
    X_tr_mrmr = X_tr[:, sel_idx]
    X_te_mrmr = X_te[:, sel_idx]

    print(f"  Features selecionadas ({len(selected_features)}):")
    print(f"    Obrigatórias ({len(mandatory)}): {mandatory}")
    print(f"    Opcionais após filtro ({len(optional_sel)}): {optional_sel}")

    # Salvar relatório de seleção
    report_rows = []
    X_tr_df = pd.DataFrame(X_tr, columns=feature_cols)
    y_ser   = pd.Series(y_tr)
    for f in feature_cols:
        report_rows.append({
            "feature":      f,
            "grupo":        "obrigatoria" if f in MANDATORY_ALWAYS else "opcional",
            "selecionada":  f in selected_features,
            "corr_target":  round(abs(X_tr_df[f].corr(y_ser)), 4),
        })
    pd.DataFrame(report_rows).to_csv(out_dir / "feature_selection_report.csv",
                                      index=False)

    regressors = get_regressors(float(y_tr.max()))
    results, imp_rows, param_rows = [], [], []

    for model_name, (est, param_dist) in regressors.items():
        print(f"\n  [{model_name}] n_iter={n_iter}...", flush=True)
        search = RandomizedSearchCV(
            est, param_dist, n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv, n_jobs=n_jobs, random_state=42, refit=True,
        )
        try:
            search.fit(X_tr_mrmr, y_tr, **cv_kwargs)
        except TypeError:
            # PanelTimeSeriesSplit requer groups; KFold não
            search.fit(X_tr_mrmr, y_tr)

        best = search.best_estimator_
        mae_cv = -search.best_score_

        y_pred = best.predict(X_te_mrmr)
        metrics = evaluate(y_te, y_pred, model_name)
        metrics["MAE_cv"] = round(mae_cv, 6)
        results.append(metrics)
        print(f"    MAE-CV={mae_cv:.5f}  |  MAE-test={metrics['MAE']:.5f}  "
              f"|  Spearman={metrics['Spearman_rho']:.4f}")

        # Importâncias
        if hasattr(best, "feature_importances_"):
            for feat, imp in zip(selected_features, best.feature_importances_):
                imp_rows.append({"model": model_name, "feature": feat,
                                 "importance": round(imp, 6)})
        elif hasattr(best, "coef_"):
            coef = best.coef_
            if hasattr(coef, "__len__"):
                for feat, c in zip(selected_features, coef):
                    imp_rows.append({"model": model_name, "feature": feat,
                                     "importance": round(abs(c), 6)})

        # Params
        param_rows.append({"model": model_name, "MAE_cv": round(mae_cv, 6),
                           **{f"param_{k}": v
                              for k, v in search.best_params_.items()}})

        # Salvar modelo
        if exp_cfg.get("save_models", True):
            joblib.dump(best, out_dir / f"model_{model_name}.pkl")

        # Salvar índices das features selecionadas (para simulação)
        joblib.dump(selected_features, out_dir / "selected_features.pkl")

    # Salvar resultados do cenário
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "metrics.csv", index=False)
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out_dir / "best_params.csv", index=False)

    # Salvar painel usado (para simulação posterior)
    pd.DataFrame({"feature": feature_cols}).to_csv(
        out_dir / "all_features.csv", index=False)
    train[feature_cols].median().to_csv(
        out_dir / "feature_medians.csv", header=["median"])

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# Plots de comparação (gerados ao final, sobre todos os experimentos)
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(summary_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Comparação de Spearman_rho por cenário e modelo
    pivot = summary_df.pivot_table(
        index="scenario_id", columns="modelo", values="Spearman_rho")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(pivot)*0.4 + 2)))

    im = axes[0].imshow(pivot.fillna(0).values, cmap="RdYlGn", vmin=-0.1, vmax=1.0,
                        aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index, fontsize=8)
    axes[0].set_title("Spearman ρ por cenário e modelo\n(verde=melhor)", fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                axes[0].text(j, i, f"{v:.3f}", ha="center", va="center",
                             fontsize=7, color="black")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # 2. Ranking dos melhores experimentos por Spearman_rho
    best_per_scenario = (summary_df.groupby("scenario_id")["Spearman_rho"]
                         .max().sort_values(ascending=True))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(best_per_scenario)))
    axes[1].barh(range(len(best_per_scenario)),
                 best_per_scenario.values, color=colors)
    axes[1].set_yticks(range(len(best_per_scenario)))
    axes[1].set_yticklabels(best_per_scenario.index, fontsize=8)
    for i, v in enumerate(best_per_scenario.values):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].set_xlabel("Melhor Spearman ρ (qualquer modelo)")
    axes[1].set_title("Ranking de cenários\n(melhor modelo por cenário)", fontsize=10)
    axes[1].axvline(0, color="black", lw=0.8, ls="--")

    fig.suptitle("Comparação de todos os experimentos\n"
                 f"({len(summary_df)} experimentos: "
                 f"{summary_df['scenario_id'].nunique()} cenários × modelos)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "scenarios_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ scenarios_comparison.png")


def plot_feature_importance_heatmap(summary_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_imp = []
    for scen_id in summary_df["scenario_id"].unique():
        imp_file = out_dir / scen_id / "feature_importance.csv"
        if imp_file.exists():
            df = pd.read_csv(imp_file)
            df["scenario_id"] = scen_id
            all_imp.append(df)

    if not all_imp:
        return

    imp_all = pd.concat(all_imp, ignore_index=True)
    pivot = imp_all.groupby(["scenario_id", "feature"])["importance"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.7), max(6, len(pivot)*0.5)))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Importância média das features por cenário\n"
                 "(média sobre modelos com feature_importances_)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ feature_importance_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Geração do relatório do perfil sazonal de umidade
# ══════════════════════════════════════════════════════════════════════════════

def generate_humidity_profile_plot(exp_cfg: dict, out_dir: Path):
    panel = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    seasonal_cfg = exp_cfg.get("seasonal", {})
    humidity_col = seasonal_cfg.get("humidity_variable", "meteo_umidade")
    if humidity_col not in panel.columns:
        print(f"  (perfil de umidade: '{humidity_col}' não disponível no painel)")
        return

    from seasonal_grouping import compute_humidity_phases
    n_phases     = seasonal_cfg.get("n_phases", 4)
    phase_labels = seasonal_cfg.get("phase_labels")
    month_to_phase, phase_labels = compute_humidity_phases(
        panel, humidity_col, n_phases, phase_labels)
    from seasonal_grouping import plot_humidity_profile
    plot_humidity_profile(panel, humidity_col, month_to_phase, phase_labels,
                          out_dir / "humidity_seasonal_profile.png")


# ══════════════════════════════════════════════════════════════════════════════
# Orquestrador principal
# ══════════════════════════════════════════════════════════════════════════════

def generate_scenarios(exp_cfg: dict) -> list[dict]:
    """Gera todas as combinações de cenários."""
    combos = list(itertools.product(
        exp_cfg["temporal_groupings"],
        exp_cfg["municipality_filters"],
        exp_cfg["feature_sets"],
    ))
    scenarios = []
    for i, (tg, mf, fs) in enumerate(combos, 1):
        scenarios.append({
            "id":                 f"S{i:02d}",
            "temporal_grouping":  tg,
            "municipality_filter": mf,
            "feature_set":        fs,
            "label": f"{tg}__{mf}__{fs}",
        })
    return scenarios


def run_all(fast_mode: bool = False, only_scenario: str | None = None, force: bool = False):
    exp_cfg, per_cfg, var_cfg = load_configs()
    scenarios = generate_scenarios(exp_cfg)

    if only_scenario:
        scenarios = [s for s in scenarios if s["id"] == only_scenario.upper()]
        if not scenarios:
            print(f"Cenário '{only_scenario}' não encontrado. "
                  f"Disponíveis: {[s['id'] for s in generate_scenarios(exp_cfg)]}")
            return

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ORQUESTRADOR DE EXPERIMENTOS — SRAG/GO")
    print(f"  {len(scenarios)} cenários  |  fast_mode={fast_mode}")
    print(f"  Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    for s in scenarios:
        print(f"  {s['id']}: {s['label']}")

    # Perfil sazonal de umidade (para apresentação)
    if any(s["temporal_grouping"] == "seasonal_humidity" for s in scenarios):
        generate_humidity_profile_plot(exp_cfg, OUT_BASE)

    all_results = []
    completed = []

    for scenario in scenarios:
        sid = scenario["id"]

        # Verificar se já foi executado (checkpoint)
        metrics_file = OUT_BASE / sid / "metrics.csv"
        if metrics_file.exists() and not fast_mode and not force:
            print(f"\n  [SKIP] {sid} já executado — carregando resultado")
            df = pd.read_csv(metrics_file)
            df["scenario_id"] = sid
            df["scenario_label"] = scenario["label"]
            for k, v in scenario.items():
                if not k.startswith("_"):
                    df[k] = v
            all_results.append(df)
            completed.append(sid)
            continue

        try:
            df = run_single_experiment(sid, scenario, exp_cfg, per_cfg, fast_mode)
            df["scenario_id"] = sid
            df["scenario_label"] = scenario["label"]
            for k, v in scenario.items():
                if not k.startswith("_"):
                    df[k] = v
            all_results.append(df)
            completed.append(sid)
        except Exception as e:
            print(f"\n  ERRO no cenário {sid}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not all_results:
        print("Nenhum experimento concluído.")
        return

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(OUT_BASE / "summary_all_experiments.csv", index=False)

    # Melhor por cenário
    best = (summary.loc[summary.groupby("scenario_id")["Spearman_rho"].idxmax()]
            [["scenario_id","scenario_label","modelo","Spearman_rho","R2",
              "MAE","Pearson_r","MAPE_pct"]]
            .sort_values("Spearman_rho", ascending=False))
    best.to_csv(OUT_BASE / "best_models_per_scenario.csv", index=False)

    # Plots de comparação
    if exp_cfg.get("generate_plots", True):
        plot_comparison(summary, OUT_BASE)
        plot_feature_importance_heatmap(summary, OUT_BASE)

    print(f"\n{'='*60}")
    print("RESUMO FINAL — Melhores resultados por cenário:")
    print(best.to_string(index=False))
    print(f"\nResultados completos: {OUT_BASE}/summary_all_experiments.csv")
    print(f"Concluído: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido: n_iter=5 (teste, ~10 min)")
    parser.add_argument("--scenario", default=None,
                        help="Executar só um cenário (ex: S03)")
    parser.add_argument("--force", action="store_true",
                        help="Ignorar resultados anteriores e re-executar tudo")
    args = parser.parse_args()
    run_all(fast_mode=args.fast, only_scenario=args.scenario, force=args.force)
