"""
plot_all.py — Geração completa de gráficos diagnósticos
=========================================================
  1. Métricas globais comparativas
  2. Predicted vs Actual (scatter + linha de identidade)
  3. Distribuição de resíduos
  4. Curvas ROC (binarizado: taxa > 0)
  5. Importância de variáveis — 3 modelos lado a lado
  6. SHAP aproximado (permutation-based, TreeExplainer manual para RF/GBM)
  7. Partial Dependence Plots das top-5 features
  8. Série temporal: previsão vs real nos 3 maiores municípios
  9. Calibração: distribuição de erros por faixa de caso real
 10. Mapa de calor correlação entre features
"""
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import roc_curve, auc, mean_absolute_error, r2_score
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGS   = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Paleta por nome de modelo (extensível — novos modelos recebem cor automática)
_PALETTE = ["#E67E22", "#2980B9", "#27AE60", "#8E44AD", "#C0392B", "#16A085"]

def _get_model_info():
    """Lê os nomes reais dos modelos do CSV de predições (não hardcoded)."""
    preds = pd.read_csv(TABLES / "predicoes_teste.csv")
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    labels    = [c.replace("pred_", "").replace("_", " ") for c in pred_cols]
    colors    = _PALETTE[:len(pred_cols)]
    return pred_cols, labels, colors

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130,
})

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130,
})

def load_config():
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def impute(df, feature_cols):
    df = df.sort_values(["cod_mun","date"]).copy()
    for col in feature_cols:
        if col == "has_uti":
            df[col] = df[col].fillna(0); continue
        df[col] = df.groupby("cod_mun")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
    return df

def get_Xy():
    cfg = load_config()
    target = cfg.get("model_target", "srag_taxa_casos")
    exclude = set(cfg.get("model_features_exclude",[])) | {"cod_mun","date","municipio",target}
    panel = pd.read_csv(ROOT/"data/processed/panel_monthly.csv", parse_dates=["date"])
    feature_cols = [c for c in panel.columns if c not in exclude and panel[c].dtype != object]
    mp = panel[panel[target].notna()].copy()
    mp = impute(mp, feature_cols)
    with open(ROOT/"config/periods.yaml") as f:
        pcfg = yaml.safe_load(f)
    n_hold = pcfg.get("test_holdout_months", 6)
    test_dates = set(sorted(mp["date"].unique())[-n_hold:])
    train = mp[~mp["date"].isin(test_dates)]
    test  = mp[ mp["date"].isin(test_dates)]
    return train, test, feature_cols, target, mp

def retrain_models(train, test, feature_cols, target):
    """Carrega os modelos já treinados do disco (outputs/models/*.pkl).
    NÃO repete a busca de hiperparâmetros — apenas carrega e prediz."""
    import joblib
    Xte, yte = test[feature_cols].values, test[target].values
    Xtr, ytr = train[feature_cols].values, train[target].values

    models_dir = ROOT / "outputs" / "models"
    pkls = sorted(models_dir.glob("*.pkl")) if models_dir.exists() else []

    if not pkls:
        print("  AVISO: nenhum modelo salvo em outputs/models/ — rode run_pipeline.py primeiro")
        return {}, Xtr, ytr, Xte, yte

    models_dict = {}
    for pkl in pkls:
        name = pkl.stem
        try:
            est = joblib.load(pkl)
            pred = est.predict(Xte)
            models_dict[name] = (est, None, pred)
            print(f"  Carregado: {name}")
        except Exception as e:
            print(f"  ERRO ao carregar {pkl.name}: {e}")

    return models_dict, Xtr, ytr, Xte, yte

# ══════════════════════════════════════════════════════════════════════════════
# 1. MÉTRICAS GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════
def plot_metrics():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    df = pd.read_csv(TABLES / "metricas_modelos_global.csv")

    # Métricas absolutas (linha 1) + correlacionais (linha 2)
    row1_metrics = ["MAE", "RMSE", "R2"]
    row2_metrics = ["MAPE_pct", "sMAPE_pct", "Pearson_r", "Spearman_rho"]
    row2_metrics = [m for m in row2_metrics if m in df.columns]

    fig, axes = plt.subplots(2, max(len(row1_metrics), len(row2_metrics)),
                              figsize=(16, 7))

    for ax, m in zip(axes[0], row1_metrics):
        vals  = df[m].values
        bars  = ax.bar(MODEL_LABELS, vals, color=MODEL_COLORS, width=0.55, edgecolor="white")
        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(MODEL_LABELS)))
        ax.set_xticklabels(MODEL_LABELS, rotation=18, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + abs(v)*0.02 if v >= 0 else bar.get_height() - abs(v)*0.12,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if m == "R2":
            ax.axhline(0, color="black", lw=0.8, ls="--")
    for ax in axes[0][len(row1_metrics):]:
        ax.set_visible(False)

    labels2 = {"MAPE_pct": "MAPE (%)\n(só onde há casos)",
                "sMAPE_pct": "sMAPE (%)\n(simétrico, inclui zeros)",
                "Pearson_r": "Pearson r\n(correlação linear pred vs real)",
                "Spearman_rho": "Spearman ρ\n(correlação de rank)"}

    for ax, m in zip(axes[1], row2_metrics):
        vals  = df[m].values
        bars  = ax.bar(MODEL_LABELS, vals, color=MODEL_COLORS, width=0.55, edgecolor="white")
        ax.set_title(labels2.get(m, m), fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(MODEL_LABELS)))
        ax.set_xticklabels(MODEL_LABELS, rotation=18, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height() + abs(v)*0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if m in ("Pearson_r", "Spearman_rho"):
            ax.axhline(0, color="black", lw=0.8, ls="--")
    for ax in axes[1][len(row2_metrics):]:
        ax.set_visible(False)

    fig.suptitle("Métricas de avaliação — conjunto de teste (fora da amostra)\n"
                 "Linha 1: erro absoluto  |  Linha 2: erro relativo e correlação",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGS/"01_metricas_globais.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 01_metricas_globais.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PREDICTED VS ACTUAL
# ══════════════════════════════════════════════════════════════════════════════
def plot_pred_vs_actual():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    preds = pd.read_csv(TABLES / "predicoes_teste.csv")
    yte   = preds["y_true"].values
    n = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(4*n + 2, 5))
    if n == 1:
        axes = [axes]
    for ax, col, label, color in zip(axes, MODEL_NAMES, MODEL_LABELS, MODEL_COLORS):
        yp  = preds[col].values
        lim = max(yte.max(), yp.max()) * 1.05
        ax.scatter(yte, yp, alpha=0.35, s=12, color=color, edgecolors="none")
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        mae = mean_absolute_error(yte, yp)
        r2  = r2_score(yte, yp)
        ax.set_title(f"{label}\nMAE={mae:.6f}  R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("Real (taxa casos/pop)")
        ax.set_ylabel("Predito")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.suptitle("Predito vs Real — conjunto de teste", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGS/"02_pred_vs_actual.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 02_pred_vs_actual.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUIÇÃO DE RESÍDUOS
# ══════════════════════════════════════════════════════════════════════════════
def plot_residuals():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    preds = pd.read_csv(TABLES / "predicoes_teste.csv")
    yte   = preds["y_true"].values
    n = len(MODEL_NAMES)
    fig, axes = plt.subplots(2, n, figsize=(4*n + 2, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col_i, (col, label, color) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS)):
        yp  = preds[col].values
        res = yp - yte
        ax = axes[0, col_i]
        ax.scatter(yp, res, alpha=0.3, s=10, color=color, edgecolors="none")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Predito"); ax.set_ylabel("Resíduo (pred − real)")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax = axes[1, col_i]
        ax.hist(res, bins=60, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.axvline(np.median(res), color="red", lw=1, ls="-",
                   label=f"Mediana={np.median(res):.5f}")
        ax.set_xlabel("Resíduo"); ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)
    fig.suptitle("Análise de resíduos por modelo", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGS/"03_residuos.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 03_residuos.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CURVAS ROC (binarizado: taxa > 0)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    preds  = pd.read_csv(TABLES / "predicoes_teste.csv")
    y_bin  = (preds["y_true"] > 0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0,1],[0,1],"k--",lw=1,label="Aleatório (AUC=0.50)")
    for col, label, color in zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS):
        score = preds[col].values
        fpr, tpr, _ = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  (AUC={roc_auc:.3f})")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.set_title("Curvas ROC — classificação: houve caso de SRAG?\n(taxa > 0 = positivo)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGS/"04_curvas_roc.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 04_curvas_roc.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. IMPORTÂNCIA DE VARIÁVEIS — 3 modelos
# ══════════════════════════════════════════════════════════════════════════════
def plot_importances():
    _, model_labels, model_colors = _get_model_info()
    imp_data = []
    for label, color in zip(model_labels, model_colors):
        name = label.replace(" ", "_")
        for suffix, coef_col in [("_importancia.csv", "importancia"),
                                  ("_coeficientes.csv", "coeficiente")]:
            f = TABLES / f"{name}{suffix}"
            if f.exists():
                d = pd.read_csv(f)
                if coef_col not in d.columns:
                    d[coef_col] = d.iloc[:,1].abs()
                imp_data.append((label, color, d.rename(columns={coef_col:"importancia"})))
                break

    feature_labels = {
        "cobertura_aps": "Cob. APS",
        "pct_pop_60mais": "% Pop 60+",
        "pct_pop_menor12": "% Pop <12a",
        "has_uti": "Tem UTI",
        "cadunico_pct_pop": "CadÚnico %pop",
        "caged_taxa_pop": "CAGED /1000hab",
        "pib_per_capita": "PIB per capita",
        "vacina_bcg": "Vacina BCG",
        "vacina_dtp": "Vacina DTP",
        "vacina_pentavalente": "Pentavalente",
        "vacina_pneumococica": "Pneumocócica",
        "vacina_tetraviral": "Tetraviral",
        "vacina_triplice_viral_d1": "Tríplice D1",
        "vacina_triplice_viral_d2": "Tríplice D2",
        "mapbiomas_pct_farming": "MB Agropec.",
        "mapbiomas_pct_forest": "MB Floresta",
        "mapbiomas_pct_non_forest_natural": "MB Cerrado",
        "mapbiomas_pct_non_vegetated": "MB Urbano",
        "mapbiomas_pct_water": "MB Água",
    }

    if not imp_data:
        print("  (importâncias não encontradas — pulando gráfico 05)")
        return

    N = 15
    fig, axes = plt.subplots(1, len(imp_data), figsize=(5*len(imp_data)+1, 6))
    if len(imp_data) == 1:
        axes = [axes]
    for ax, (label, color, df) in zip(axes, imp_data):
        top = df.nlargest(N, "importancia").copy()
        top["var_label"] = top["variavel"].map(lambda v: feature_labels.get(v, v))
        top = top.sort_values("importancia")
        ax.barh(top["var_label"], top["importancia"], color=color, edgecolor="white")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importância")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Importância das variáveis — 3 modelos (top 15)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGS/"05_importancia_variaveis.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 05_importancia_variaveis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP APROXIMADO (permutation-based, por modelo)
# ══════════════════════════════════════════════════════════════════════════════
def plot_shap_approx(models_dict, Xte, yte, feature_cols):
    """
    Sem a lib shap, usamos permutation importance com n_repeats=30 para
    estimar a contribuição média de cada feature no conjunto de teste.
    O gráfico de 'beeswarm' é aproximado por um strip plot colorido pelo
    valor da feature (normalizado), análogo ao SHAP summary plot.
    """
    feature_labels = {
        "cobertura_aps": "Cob. APS (%)",
        "pct_pop_60mais": "% Pop 60+",
        "pct_pop_menor12": "% Pop <12a",
        "has_uti": "Tem UTI (0/1)",
        "cadunico_pct_pop": "CadÚnico %pop",
        "caged_taxa_pop": "CAGED /1000hab",
        "pib_per_capita": "PIB per capita",
        "vacina_bcg": "Vacina BCG",
        "vacina_dtp": "Vacina DTP",
        "vacina_pentavalente": "Pentavalente",
        "vacina_pneumococica": "Pneumocócica",
        "vacina_tetraviral": "Tetraviral",
        "vacina_triplice_viral_d1": "Tríplice D1",
        "vacina_triplice_viral_d2": "Tríplice D2",
        "mapbiomas_pct_farming": "MB Agropec.",
        "mapbiomas_pct_forest": "MB Floresta",
        "mapbiomas_pct_non_forest_natural": "MB Cerrado",
        "mapbiomas_pct_non_vegetated": "MB Urbano",
        "mapbiomas_pct_water": "MB Água",
    }

    cmap = LinearSegmentedColormap.from_list("shap_cmap", ["#3498db","#e74c3c"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (name, (model, scaler, _)) in zip(axes, models_dict.items()):
        N_TOP = min(12, len(feature_cols))
        Xeval = scaler.transform(Xte) if scaler else Xte
        perm = permutation_importance(model, Xeval, yte, n_repeats=30,
                                      random_state=42, n_jobs=-1)
        imp_mean = perm.importances_mean
        imp_std  = perm.importances_std

        order = np.argsort(imp_mean)[-N_TOP:]
        top_feats  = [feature_cols[i] for i in order]
        top_labels = [feature_labels.get(f, f) for f in top_feats]
        top_imp    = imp_mean[order]
        top_std    = imp_std[order]

        # Beeswarm-like: para cada feature, strip plot colorido pelo valor normalizado
        Xeval_df = pd.DataFrame(Xeval if scaler else Xte, columns=feature_cols)
        for yi, (feat, label) in enumerate(zip(top_feats, top_labels)):
            vals = Xeval_df[feat].values
            vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
            norm_vals = np.clip((vals - vmin) / (vmax - vmin + 1e-9), 0, 1)
            # Jitter vertical
            jitter = np.random.RandomState(42).uniform(-0.3, 0.3, len(vals))
            ax.scatter(top_imp[yi] * norm_vals * 0.8 + np.random.RandomState(0).normal(0, top_std[yi]*0.2, len(vals)),
                       yi + jitter,
                       c=norm_vals, cmap=cmap, alpha=0.35, s=8, edgecolors="none")

        ax.set_yticks(range(N_TOP))
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.set_xlabel("Importância de permutação (redução de MAE)")
        ax.set_title(f"{name}\n(SHAP aprox.)", fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", lw=0.8)

    # Barra de cor
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.03, pad=0.04)
    cbar.set_label("Valor da feature\n(azul=baixo, vermelho=alto)", fontsize=8)
    fig.suptitle("SHAP aproximado — contribuição de cada feature\n"
                 "(permutation importance × valor, conjunto de teste)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGS/"06_shap_aproximado.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 06_shap_aproximado.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PARTIAL DEPENDENCE PLOTS — top 5 features do melhor modelo com importâncias
# ══════════════════════════════════════════════════════════════════════════════
def plot_pdp(models_dict, Xtr, feature_cols):
    # Usar o primeiro modelo com feature_importances_ (RF, GBM ou XGBoost)
    tree_name = next(
        (n for n, (m, _, _) in models_dict.items() if hasattr(m, "feature_importances_")),
        None
    )
    if tree_name is None:
        print("  (PDP pulado — nenhum modelo com feature_importances_)")
        return

    tree_model = models_dict[tree_name][0]
    _, model_labels, model_colors = _get_model_info()
    color = model_colors[list(models_dict.keys()).index(tree_name)] if tree_name in list(models_dict.keys()) else "#2980B9"

    # Importâncias do modelo escolhido
    imp_file = TABLES / f"{tree_name}_importancia.csv"
    if not imp_file.exists():
        print(f"  (PDP pulado — {imp_file.name} não encontrado)")
        return
    imp_df = pd.read_csv(imp_file)
    top5 = imp_df.nlargest(5, "importancia")["variavel"].tolist()
    top5_idx = [feature_cols.index(f) for f in top5 if f in feature_cols]
    if not top5_idx:
        print("  (PDP pulado — features não encontradas)")
        return

    feature_labels = {
        "cobertura_aps": "Cob. APS (%)", "pct_pop_60mais": "% Pop 60+",
        "pct_pop_menor12": "% Pop <12a", "has_uti": "Tem UTI",
        "cadunico_pct_pop": "CadÚnico %pop", "caged_taxa_pop": "CAGED /1000hab",
        "pib_per_capita": "PIB per capita",
        "mapbiomas_pct_farming": "MB Agropec.", "mapbiomas_pct_forest": "MB Floresta",
        "mapbiomas_pct_non_forest_natural": "MB Cerrado",
        "mapbiomas_pct_non_vegetated": "MB Urbano", "mapbiomas_pct_water": "MB Água",
        "meteo_temp_media": "Temp. média", "meteo_temp_amplitude": "Amplitude térm.",
        "meteo_precipitacao": "Precip.", "meteo_umidade": "Umidade",
        "fogo_pct_territorio": "% Queimado",
    }
    feature_names = [feature_labels.get(f, f) for f in feature_cols]

    n = len(top5_idx)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    PartialDependenceDisplay.from_estimator(
        tree_model, Xtr, top5_idx,
        feature_names=feature_names,
        ax=axes,
        line_kw={"color": color, "lw": 2},
        n_jobs=-1,
    )
    fig.suptitle(f"Partial Dependence Plots — {tree_name.replace('_',' ')}\n(top 5 features por importância)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGS/"07_partial_dependence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 07_partial_dependence.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SÉRIE TEMPORAL — 3 maiores municípios
# ══════════════════════════════════════════════════════════════════════════════
def plot_time_series():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    preds = pd.read_csv(TABLES/"predicoes_teste.csv", parse_dates=["date"])
    panel = pd.read_csv(ROOT/"data/processed/panel_monthly.csv", parse_dates=["date"])
    panel = panel[panel["srag_taxa_casos"].notna()]

    top3 = (panel.groupby("municipio")["srag_taxa_casos"].sum()
            .nlargest(3).index.tolist())

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, mun in zip(axes, top3):
        full = panel[panel.municipio == mun][["date","srag_taxa_casos"]].sort_values("date")
        ax.fill_between(full["date"], full["srag_taxa_casos"], alpha=0.15,
                        color="gray", label="_nolegend_")
        ax.plot(full["date"], full["srag_taxa_casos"],
                color="gray", lw=1.5, label="Real (treino + teste)")
        test_mun = preds[preds.municipio == mun].sort_values("date")
        if not test_mun.empty:
            ax.plot(test_mun["date"], test_mun["y_true"], "k-", lw=2.5,
                    label="Real (teste)")
            for col, label, color in zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS):
                ax.plot(test_mun["date"], test_mun[col], "--", color=color,
                        lw=1.8, label=label)
            ax.axvline(test_mun["date"].min(), color="red", ls=":", lw=1.2,
                       label="Início do teste")
        ax.set_title(mun.title(), fontsize=10, fontweight="bold")
        ax.set_ylabel("Taxa SRAG\n(casos/100k)")
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Data")
    fig.suptitle("Série temporal: real vs predito — 3 municípios com mais casos",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGS/"08_serie_temporal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 08_serie_temporal.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. CALIBRAÇÃO — erro por faixa de taxa real
# ══════════════════════════════════════════════════════════════════════════════
def plot_calibration():
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS = _get_model_info()
    preds = pd.read_csv(TABLES/"predicoes_teste.csv")
    preds = preds[preds["y_true"] > 0].copy()   # só onde houve caso
    if len(preds) < 20:
        print("  (calibração pulada — poucos positivos no teste)")
        return
    preds["faixa"] = pd.qcut(preds["y_true"], q=5, duplicates="drop",
                              labels=["Q1\n(baixo)","Q2","Q3","Q4","Q5\n(alto)"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(preds["faixa"].cat.categories))
    w = 0.25
    for i, (col, label, color) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS)):
        errs = preds.groupby("faixa", observed=True).apply(
            lambda g: mean_absolute_error(g["y_true"], g[col]))
        ax.bar(x + i*w, errs.values, w, label=label, color=color, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(preds["faixa"].cat.categories)
    ax.set_xlabel("Quintil da taxa real de SRAG (somente onde houve caso)")
    ax.set_ylabel("MAE (casos/100k)")
    ax.set_title("Calibração: erro médio por faixa de intensidade real")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS/"09_calibracao_faixas.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 09_calibracao_faixas.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CORRELAÇÃO ENTRE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_correlation(train, feature_cols):
    short = {
        "cobertura_aps": "APS", "pct_pop_60mais": "Pop60+",
        "pct_pop_menor12": "Pop<12", "has_uti": "UTI",
        "cadunico_pct_pop": "CadÚnico", "caged_taxa_pop": "CAGED",
        "pib_per_capita": "PIB/cap", "vacina_bcg": "BCG",
        "vacina_dtp": "DTP", "vacina_pentavalente": "Penta",
        "vacina_pneumococica": "Pneumo", "vacina_tetraviral": "Tetra",
        "vacina_triplice_viral_d1": "TríplD1","vacina_triplice_viral_d2": "TríplD2",
        "mapbiomas_pct_farming": "Agropec","mapbiomas_pct_forest": "Floresta",
        "mapbiomas_pct_non_forest_natural": "Cerrado",
        "mapbiomas_pct_non_vegetated": "Urbano","mapbiomas_pct_water": "Água",
    }
    X = train[feature_cols].copy()
    X.columns = [short.get(c, c[:8]) for c in X.columns]
    corr = X.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = LinearSegmentedColormap.from_list("corr", ["#2980b9","white","#c0392b"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45,
                                                          ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if abs(v) > 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Correlação de Spearman entre features (conjunto de treino)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGS/"10_correlacao_features.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ 10_correlacao_features.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n=== Gerando gráficos diagnósticos ===\n")
    # Plots que só precisam dos CSVs salvos (sem retreinar)
    plot_metrics()
    plot_pred_vs_actual()
    plot_residuals()
    plot_roc()
    plot_importances()
    plot_time_series()
    plot_calibration()

    # Plots que precisam retreinar os modelos em memória
    print("\nRe-treinando modelos para plots avançados...")
    train, test, feature_cols, target, _ = get_Xy()
    models_dict, Xtr, ytr, Xte, yte = retrain_models(train, test, feature_cols, target)
    print("  Modelos prontos.")

    plot_shap_approx(models_dict, Xte, yte, feature_cols)
    plot_pdp(models_dict, Xtr, feature_cols)
    plot_feature_correlation(train, feature_cols)

    print(f"\n✓ Todos os gráficos salvos em {FIGS}")
    figs = sorted(FIGS.glob("0*.png"))
    print(f"  {len(figs)} arquivos gerados:")
    for f in figs:
        print(f"   {f.name}")

if __name__ == "__main__":
    main()
