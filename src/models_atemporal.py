"""
models_atemporal.py
--------------------
Modelo atemporal: agrega o painel (município × mês) para 1 linha/município
e faz regressão cross-sectional de RISCO ESTRUTURAL de SRAG.

AGREGAÇÕES DISPONÍVEIS (atemporal_aggregation em periods.yaml):
  pct_meses_com_casos  → % meses em que o município teve ≥1 caso  ← RECOMENDADO
                          (apenas ~24% de zeros vs 88% na median)
  mean                 → taxa média mensal (mais sensível a surtos)
  mean_nonzero         → taxa média SOMENTE nos meses com casos
                          (ignora zeros, captura intensidade típica)
  max                  → pico histórico (potencial máximo)
  q75                  → 3º quartil (período mais intenso)
  median               → mediana (NÃO recomendado: 88% zeros)

SAÍDAS:
  outputs/tables/atemporal_municipios_ranqueados.csv
  outputs/figures/atemporal_scatter.png
  outputs/figures/atemporal_importancias.png
  outputs/figures/atemporal_ranking_municipios.png
  outputs/figures/atemporal_mapa_risco.png
"""

import sys, json, yaml, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import loguniform, uniform, pearsonr, spearmanr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"
OUT_MODELS = ROOT / "outputs" / "models"

sys.path.insert(0, str(ROOT / "src"))
from models import load_config, get_feature_cols, impute_features, ScaledPoissonGLM

PALETTE = ["#E67E22", "#2980B9", "#27AE60", "#8E44AD", "#C0392B"]


# ══════════════════════════════════════════════════════════════════════════════
# Agregação temporal → cross-section
# ══════════════════════════════════════════════════════════════════════════════

def build_cross_section(pcfg: dict):
    agg_func = pcfg.get("atemporal_aggregation", "pct_meses_com_casos")

    panel = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    vcfg, _ = load_config()
    target, feature_cols = get_feature_cols(panel, vcfg)

    # Restringir à janela com alvo observado
    mp = panel[panel[target].notna()].copy()
    mp = impute_features(mp, feature_cols)

    # Filtro opcional (ver nota abaixo sobre por que raramente muda os resultados)
    if pcfg.get("apenas_municipios_com_casos", False):
        muns_com_caso = (mp.groupby("cod_mun")[target]
                         .max().pipe(lambda s: s[s > 0]).index)
        antes = mp["cod_mun"].nunique()
        mp    = mp[mp["cod_mun"].isin(muns_com_caso)]
        print(f"  Filtro apenas_municipios_com_casos: {len(muns_com_caso)}/{antes} municípios")
        print(f"  NOTA: no modo atemporal, municípios sem casos são exemplos válidos de")
        print(f"        'baixo risco estrutural'. O filtro pode REDUZIR a qualidade do modelo.")

    # Agregar features
    cs_feats = mp.groupby("cod_mun")[feature_cols].median().reset_index()

    # Agregar target segundo a função escolhida
    grp = mp.groupby("cod_mun")[target]
    if agg_func == "pct_meses_com_casos":
        target_agg = grp.apply(lambda x: (x > 0).mean())
        target_col = "pct_meses_com_casos"
    elif agg_func == "mean_nonzero":
        target_agg = grp.apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0.0)
        target_col = "taxa_media_meses_com_casos"
    elif agg_func == "q75":
        target_agg = grp.quantile(0.75)
        target_col = target + "_q75"
    elif agg_func in ("mean", "max", "median"):
        target_agg = getattr(grp, agg_func)()
        target_col = f"{target}_{agg_func}"
    else:
        raise ValueError(f"atemporal_aggregation inválido: '{agg_func}'. "
                         f"Use: pct_meses_com_casos, mean, mean_nonzero, max, q75, median")

    target_agg = target_agg.rename(target_col).reset_index()
    cs = cs_feats.merge(target_agg, on="cod_mun", how="left")

    # Nome do município
    mun_names = (panel[["cod_mun","municipio"]].drop_duplicates())
    cs = cs.merge(mun_names, on="cod_mun", how="left")

    y = cs[target_col].values
    n_zero = (y == 0).sum()
    n_mun  = len(cs)

    print(f"\n  Alvo: '{target_col}' (agregação: {agg_func})")
    print(f"  {n_mun} municípios  |  zeros: {n_zero} ({n_zero/n_mun*100:.0f}%)")
    print(f"  min={y.min():.4f}  mediana={np.median(y):.4f}  "
          f"max={y.max():.4f}  std={y.std():.4f}")

    if n_zero / n_mun > 0.7:
        print(f"\n  ⚠  {n_zero/n_mun*100:.0f}% zeros: considere atemporal_aggregation: "
              f"pct_meses_com_casos ou mean")

    return cs, target_col, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# Modelos
# ══════════════════════════════════════════════════════════════════════════════

def get_regressors(y):
    """Escolhe modelos adequados ao target. Se target é proporção [0,1],
    usa Ridge em vez de Poisson GLM."""
    is_proportion = (y.max() <= 1.0 + 1e-6) and (y.min() >= 0)

    regs = {}
    if is_proportion:
        regs["Ridge"] = (
            Ridge(),
            {"alpha": loguniform(0.01, 100)}
        )
    else:
        regs["Poisson_GLM"] = (
            ScaledPoissonGLM(),
            {"alpha": loguniform(1e-4, 20), "max_iter": [1000, 2000, 5000]}
        )

    regs["Random_Forest"] = (
        RandomForestRegressor(random_state=42),
        {"n_estimators":     [100, 200, 400],
         "max_depth":        [4, 6, 8, 12, None],
         "min_samples_leaf": [1, 2, 3, 5],
         "max_features":     ["sqrt", "log2", 0.5]}
    )
    regs["GradientBoosting"] = (
        HistGradientBoostingRegressor(
            loss="squared_error" if is_proportion else "poisson",
            random_state=42),
        {"max_depth":         [3, 4, 5, 6],
         "learning_rate":     loguniform(0.01, 0.3),
         "max_iter":          [100, 200, 400],
         "min_samples_leaf":  [3, 5, 10, 20],
         "l2_regularization": loguniform(1e-4, 1.0)}
    )
    if HAS_XGBOOST:
        regs["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror" if is_proportion else "reg:tweedie",
                tweedie_variance_power=1.5,
                tree_method="hist", random_state=42, verbosity=0),
            {"n_estimators":     [100, 200, 400],
             "max_depth":        [3, 4, 5, 6],
             "learning_rate":    loguniform(0.01, 0.3),
             "subsample":        uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.6, 0.4),
             "reg_alpha":        loguniform(1e-4, 10)}
        )
    return regs


def evaluate(y_true, y_pred, label):
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
# Plots
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


def plot_scatter(cs, predictions, target_col, results_df, pcfg):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5*n + 1, 5))
    if n == 1: axes = [axes]

    y_true = cs[target_col].values
    agg    = pcfg.get("atemporal_aggregation", "pct_meses_com_casos")

    for ax, (name, y_pred), color in zip(axes, predictions.items(), PALETTE):
        # Métricas
        row = results_df[results_df.modelo == name].iloc[0]
        r2  = row["R2"];  sr = row["Spearman_rho"]; pr = row["Pearson_r"]
        mae = row["MAE"]

        ax.scatter(y_true, y_pred, alpha=0.55, s=22, color=color, edgecolors="none")
        lim = max(y_true.max(), y_pred.max()) * 1.08
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="Perfeito")
        ax.set_xlim(-lim*0.02, lim); ax.set_ylim(-lim*0.02, lim)

        # Métricas no gráfico
        txt = f"R² = {r2:.3f}\nSpearman ρ = {sr:.3f}\nPearson r = {pr:.3f}\nMAE = {mae:.5f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.8, edgecolor=color))

        # Destacar top-10 municípios
        top10_idx = np.argsort(y_true)[-10:]
        ax.scatter(y_true[top10_idx], y_pred[top10_idx], s=50, color="red",
                   zorder=5, edgecolors="darkred", lw=0.8, label="Top 10")
        for idx in top10_idx:
            nome = str(cs.iloc[idx]["municipio"])[:12]
            ax.annotate(nome, (y_true[idx], y_pred[idx]),
                        fontsize=5.5, alpha=0.8,
                        xytext=(4, 2), textcoords="offset points")

        ax.set_xlabel(f"Real ({agg})", fontsize=9)
        ax.set_ylabel("Predito", fontsize=9)
        ax.set_title(name.replace("_"," "), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Predito vs Real — modelo atemporal\n"
        f"1 ponto = 1 município  |  alvo: {target_col}  |  N={len(cs)}",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "atemporal_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ atemporal_scatter.png")


def plot_importancias(models_fitted, feature_cols, results_df):
    imp_models = [(n, m) for n, m in models_fitted.items()
                  if hasattr(m, "feature_importances_")]
    coef_models = [(n, m) for n, m in models_fitted.items()
                   if hasattr(m, "coef_") and not hasattr(m, "feature_importances_")]

    all_models = imp_models + [(n, m) for n, m in coef_models]
    if not all_models:
        print("  (importâncias: nenhum modelo com feature_importances_ ou coef_)")
        return

    n = len(all_models)
    fig, axes = plt.subplots(1, n, figsize=(6*n + 1, max(6, len(feature_cols)*0.35)))
    if n == 1: axes = [axes]

    short = {
        "cobertura_aps":"APS (%)", "pct_pop_60mais":"Pop 60+",
        "pct_pop_menor12":"Pop<12a", "has_uti":"Tem UTI",
        "cadunico_pct_pop":"CadÚnico%", "caged_taxa_pop":"CAGED/1k",
        "pib_per_capita":"PIB/cap", "mapbiomas_pct_farming":"MB Agropec",
        "mapbiomas_pct_forest":"MB Floresta", "mapbiomas_pct_non_forest_natural":"MB Cerrado",
        "mapbiomas_pct_non_vegetated":"MB Urbano", "mapbiomas_pct_water":"MB Água",
        "meteo_temp_media":"Temp. média", "meteo_temp_amplitude":"Amplit. térm.",
        "meteo_precipitacao":"Precipitação", "meteo_umidade":"Umidade",
        "fogo_pct_territorio":"% Queimado",
    }

    for ax, (name, model), color in zip(axes, all_models, PALETTE[1:]):
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            xlabel = "Importância (Gini)"
        else:
            coef = model.coef_ if hasattr(model, "coef_") else np.zeros(len(feature_cols))
            vals = np.abs(coef)
            xlabel = "|Coeficiente|"

        series = pd.Series(vals, index=feature_cols).rename(lambda x: short.get(x, x))
        series = series.sort_values()

        bars = ax.barh(series.index, series.values, color=color, edgecolor="white")
        for bar, v in zip(bars, series.values):
            ax.text(v + series.values.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=7)

        # Título com métricas
        row = results_df[results_df.modelo == name]
        if not row.empty:
            r2 = row.iloc[0]["R2"]; sr = row.iloc[0]["Spearman_rho"]
            ax.set_title(f"{name.replace('_',' ')}\nR²={r2:.3f}  ρ={sr:.3f}",
                         fontsize=9, fontweight="bold")
        else:
            ax.set_title(name.replace("_"," "), fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Importância das variáveis — modelo atemporal\n"
                 "(cross-sectional: 1 observação por município)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "atemporal_importancias.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ atemporal_importancias.png")


def plot_ranking(cs, target_col, best_name, best_pred, results_df):
    df = cs[["municipio", target_col]].copy()
    df["pred"] = best_pred
    df = df.sort_values(target_col, ascending=False).reset_index(drop=True)

    n = min(25, len(df))
    row  = results_df[results_df.modelo == best_name].iloc[0]
    r2   = row["R2"]; sr = row["Spearman_rho"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))

    cmap_colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, n))

    # Ranking real
    top_real = df.head(n)
    axes[0].barh(range(n), top_real[target_col].values[::-1], color=cmap_colors[::-1])
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(top_real["municipio"].values[::-1], fontsize=7)
    for i, v in enumerate(top_real[target_col].values[::-1]):
        axes[0].text(v + df[target_col].max()*0.01, i, f"{v:.4f}", va="center", fontsize=6)
    axes[0].set_xlabel(target_col, fontsize=9)
    axes[0].set_title(f"Top {n} por risco REAL\n(valor observado — {target_col})", fontsize=9)

    # Ranking predito
    top_pred = df.sort_values("pred", ascending=False).head(n)
    axes[1].barh(range(n), top_pred["pred"].values[::-1],
                 color=plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, n))[::-1])
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(top_pred["municipio"].values[::-1], fontsize=7)
    for i, v in enumerate(top_pred["pred"].values[::-1]):
        axes[1].text(v + df["pred"].max()*0.01, i, f"{v:.4f}", va="center", fontsize=6)
    axes[1].set_xlabel("Predito pelo modelo", fontsize=9)
    axes[1].set_title(
        f"Top {n} por risco PREDITO\n({best_name.replace('_',' ')}  R²={r2:.3f}  ρ={sr:.3f})",
        fontsize=9)

    fig.suptitle(
        f"Ranqueamento de municípios — risco estrutural de SRAG\n"
        f"(modelo atemporal: 1 valor por município, sem dimensão temporal)",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "atemporal_ranking_municipios.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ atemporal_ranking_municipios.png")


def plot_mapa_risco(cs, target_col, best_pred, best_name):
    GEOJSON_CANDIDATES = [
        ROOT / "data" / "external" / "ibge_municipios_go.geojson",
        ROOT / "data" / "external" / "goias_municipios.geojson",
    ]
    geojson_path = None
    for p in GEOJSON_CANDIDATES:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                gj = json.load(f)
            if len(gj.get("features", [])) > 50:
                geojson_path = p
                break

    if geojson_path is None:
        print("  (mapa pulado — GeoJSON municipal não encontrado em data/external/)")
        return

    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors

    try:
        cmap = matplotlib.colormaps["YlOrRd"]
    except AttributeError:
        cmap = plt.get_cmap("YlOrRd")

    cs_plot = cs.copy()
    cs_plot["pred"] = best_pred
    lookup_real = cs_plot.set_index("cod_mun")[target_col].to_dict()
    lookup_pred = cs_plot.set_index("cod_mun")["pred"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, lookup, title in [
        (axes[0], lookup_real, f"Risco real\n({target_col})"),
        (axes[1], lookup_pred, f"Risco predito\n({best_name.replace('_',' ')})"),
    ]:
        all_vals = [v for v in lookup.values() if pd.notna(v) and v > 0]
        if not all_vals:
            ax.set_title(f"{title}\n(sem dados)")
            ax.set_axis_off()
            continue

        vmin, vmax = 0, np.percentile(all_vals, 98)  # clip outliers
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        patches, colors = [], []

        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            raw_id = str(props.get("id") or props.get("codarea") or
                         props.get("CD_MUN") or props.get("codigo_ibge") or "")
            cod6 = raw_id.strip()[:6]
            val  = lookup.get(cod6, 0.0)
            geom = feat.get("geometry", {})
            rings = ([geom["coordinates"][0]] if geom["type"] == "Polygon"
                     else [p[0] for p in geom["coordinates"]])
            for ring in rings:
                try:
                    patches.append(MplPolygon(np.array(ring, dtype=float)))
                    colors.append(float(val) if pd.notna(val) else 0.0)
                except Exception:
                    continue

        if not patches:
            ax.set_title(f"{title}\n(erro: polígonos não extraídos)")
            ax.set_axis_off()
            continue

        pc = PatchCollection(patches, array=np.array(colors),
                             cmap=cmap, norm=norm,
                             edgecolor="white", linewidth=0.2)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.65,
                     label=target_col if "real" in title.lower() else "predito")

    fig.suptitle(
        "Risco estrutural de SRAG por município — Goiás\n"
        "(valores agregados por município, sem dimensão temporal)\n"
        "Vermelho escuro = maior risco crônico",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "atemporal_mapa_risco.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ atemporal_mapa_risco.png")


def plot_metricas_resumo(results_df):
    metrics = ["R2", "Spearman_rho", "Pearson_r", "MAE", "MAPE_pct"]
    metrics = [m for m in metrics if m in results_df.columns]
    n_m = len(metrics)
    n_mod = len(results_df)

    fig, axes = plt.subplots(1, n_m, figsize=(3.5*n_m + 1, 4))
    if n_m == 1: axes = [axes]

    labels  = results_df["modelo"].str.replace("_"," ").tolist()
    colors  = PALETTE[:n_mod]

    desc = {"R2": "R² (variância explicada)",
            "Spearman_rho": "Spearman ρ (rank correlation)",
            "Pearson_r": "Pearson r (linear correlation)",
            "MAE": "MAE (erro absoluto médio)",
            "MAPE_pct": "MAPE % (erro percentual)"}

    for ax, m in zip(axes, metrics):
        vals = results_df[m].values
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.55)
        ax.set_title(desc.get(m, m), fontsize=8, fontweight="bold")
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        if m in ("R2", "Spearman_rho", "Pearson_r"):
            ax.axhline(0, color="black", lw=0.8, ls="--")
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height() + abs(bar.get_height())*0.04,
                        f"{v:.4f}", ha="center", fontsize=7.5, fontweight="bold")

    fig.suptitle("Métricas — modelo atemporal (validação cruzada out-of-fold)\n"
                 "N = 246 municípios  |  avaliação: 5-fold KFold", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "atemporal_metricas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ atemporal_metricas.png")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_atemporal():
    vcfg, pcfg = load_config()
    n_jobs  = pcfg.get("n_processadores", 4)
    n_iter  = pcfg.get("hparam_n_iter",   50)
    n_folds = pcfg.get("atemporal_cv_folds", 5)
    scoring = "neg_mean_absolute_error"

    print(f"\n  Agregação: {pcfg.get('atemporal_aggregation','pct_meses_com_casos')}  |  "
          f"CV: {n_folds}-fold  |  n_iter: {n_iter}  |  n_jobs: {n_jobs}")

    cs, target_col, feature_cols = build_cross_section(pcfg)
    X = cs[feature_cols].values
    y = cs[target_col].values

    regressors = get_regressors(y)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    results, predictions, models_fitted, search_rows = [], {}, {}, []

    for name, (est, param_dist) in regressors.items():
        print(f"\n  [{name}] RandomizedSearchCV n_iter={n_iter} cv={n_folds}...",
              flush=True)
        search = RandomizedSearchCV(
            est, param_dist, n_iter=n_iter,
            scoring=scoring, cv=cv,
            n_jobs=n_jobs, random_state=42, refit=True,
        )
        search.fit(X, y)
        best  = search.best_estimator_
        mae_cv = -search.best_score_
        print(f"    → MAE-CV: {mae_cv:.5f}  |  {search.best_params_}")

        # Out-of-fold predictions (avaliação honesta)
        y_oof = np.zeros_like(y, dtype=float)
        for tr_idx, val_idx in cv.split(X, y):
            fold_est = clone(best)
            fold_est.fit(X[tr_idx], y[tr_idx])
            y_oof[val_idx] = fold_est.predict(X[val_idx])

        results.append(evaluate(y, y_oof, name))
        predictions[name] = y_oof
        models_fitted[name] = best

        search_rows.append({"modelo": name, "MAE_cv": round(mae_cv, 6),
                             **{f"param_{k}": v for k, v in search.best_params_.items()}})

        joblib.dump(best, OUT_MODELS / f"atemporal_{name}.pkl")

        if hasattr(best, "feature_importances_"):
            (pd.DataFrame({"variavel": feature_cols,
                           "importancia": best.feature_importances_})
               .sort_values("importancia", ascending=False)
               .to_csv(OUT_TABLES / f"atemporal_{name}_importancia.csv", index=False))
        if hasattr(best, "coef_"):
            coef = best.coef_ if hasattr(best.coef_, "__len__") else [best.coef_]
            (pd.DataFrame({"variavel": feature_cols[:len(coef)], "coeficiente": coef})
               .sort_values("coeficiente", key=abs, ascending=False)
               .to_csv(OUT_TABLES / f"atemporal_{name}_coeficientes.csv", index=False))

    results_df = pd.DataFrame(results)
    pd.DataFrame(search_rows).to_csv(OUT_TABLES / "atemporal_hparam_results.csv", index=False)

    # Ranking de municípios pelo melhor modelo (menor MAE out-of-fold)
    best_name = results_df.loc[results_df["MAE"].idxmin(), "modelo"]
    cs["pred_atemporal"] = predictions[best_name]
    cs["rank_real"]      = cs[target_col].rank(ascending=False).astype(int)
    cs["rank_modelo"]    = cs["pred_atemporal"].rank(ascending=False).astype(int)
    cs["diferenca_rank"] = (cs["rank_real"] - cs["rank_modelo"]).abs()

    cs.sort_values("rank_real")[
        ["rank_real","rank_modelo","diferenca_rank","cod_mun","municipio",
         target_col,"pred_atemporal"] + feature_cols
    ].to_csv(OUT_TABLES / "atemporal_municipios_ranqueados.csv", index=False)

    # Salvar métricas
    results_df.to_csv(OUT_TABLES / "atemporal_metricas.csv", index=False)

    # Plots
    print()
    plot_metricas_resumo(results_df)
    plot_scatter(cs, predictions, target_col, results_df, pcfg)
    plot_importancias(models_fitted, feature_cols, results_df)
    plot_ranking(cs, target_col, best_name, predictions[best_name], results_df)
    plot_mapa_risco(cs, target_col, predictions[best_name], best_name)

    # Resumo no console
    print("\n" + "="*60)
    print(f"MODELO ATEMPORAL — alvo: {target_col}")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nMelhor modelo (menor MAE out-of-fold): {best_name}")

    top10 = cs.nlargest(10, target_col)[
        ["rank_real","municipio", target_col, "pred_atemporal", "rank_modelo"]]
    print(f"\nTop 10 municípios de maior risco REAL:")
    print(top10.to_string(index=False))

    top10_pred = cs.nlargest(10, "pred_atemporal")[
        ["rank_modelo","municipio","pred_atemporal", target_col, "rank_real"]]
    print(f"\nTop 10 municípios de maior risco PREDITO:")
    print(top10_pred.to_string(index=False))

    print(f"\nRanking completo salvo em: outputs/tables/atemporal_municipios_ranqueados.csv")
    print("Gráficos em: outputs/figures/atemporal_*.png")

    return results_df, cs


if __name__ == "__main__":
    run_atemporal()
