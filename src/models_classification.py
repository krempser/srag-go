"""
models_classification.py
-------------------------
Modo de classificação: discretiza srag_taxa_casos em faixas e classifica.

Modelos:
  - Regressão Logística (multi-classe, interpretável)
  - Random Forest Classifier
  - Histogram Gradient Boosting Classifier
  - XGBoost Classifier (se instalado)

Métricas:
  - Acurácia ponderada (weighted accuracy)
  - F1 ponderado (lida melhor com classes desbalanceadas)
  - Matriz de confusão (heatmap)
  - Relatório por classe (precision / recall / F1)

Busca de hiperparâmetros: RandomizedSearchCV + PanelTimeSeriesSplit
(mesmo framework da regressão — configurado em periods.yaml)
"""
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import loguniform, uniform

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"
OUT_MODELS = ROOT / "outputs" / "models"

sys.path.insert(0, str(ROOT / "src"))
from models import (load_config, get_feature_cols, impute_features,
                    PanelTimeSeriesSplit)


# ══════════════════════════════════════════════════════════════════════════════
# Discretização do alvo
# ══════════════════════════════════════════════════════════════════════════════

def discretize_target(y: np.ndarray, pcfg: dict) -> tuple[np.ndarray, list[str]]:
    """
    Converte a taxa contínua em classes ordinais.
    Retorna (y_classes, nomes_das_classes).
    """
    strategy = pcfg.get("classification_strategy", "zero_plus_quantile")
    n_bins   = int(pcfg.get("classification_n_bins", 4))

    if strategy == "custom":
        thresholds = pcfg.get("classification_thresholds")
        if not thresholds:
            raise ValueError("classification_thresholds não definido em periods.yaml")
        thresholds = sorted(thresholds)
        classes = np.zeros(len(y), dtype=int)
        labels  = [f"0 [taxa=0]"]
        for i, t in enumerate(thresholds, 1):
            classes[y > (thresholds[i-2] if i > 1 else 0)] = i
            labels.append(f"{i} [>{thresholds[i-2]:.2e}]" if i > 1 else f"{i} [>0..{t:.2e}]")
        classes[y > thresholds[-1]] = len(thresholds)
        labels.append(f"{len(thresholds)} [>{thresholds[-1]:.2e}]")
        return classes, labels

    elif strategy == "zero_plus_quantile":
        # Classe 0 = zero casos; demais = quantis dos positivos
        n_pos_classes = n_bins - 1
        positivos     = y[y > 0]
        quantis       = np.quantile(positivos, np.linspace(0, 1, n_pos_classes + 1))
        classes = np.zeros(len(y), dtype=int)
        labels  = ["0\n(sem casos)"]
        for i in range(n_pos_classes):
            lo = quantis[i]
            hi = quantis[i + 1]
            mask = (y > lo) if i == 0 else (y > lo) & (y <= hi)
            if i == n_pos_classes - 1:
                mask = y > lo
            classes[mask] = i + 1
            labels.append(f"{i+1}\n(Q{i+1}: >{lo:.2e})")
        return classes, labels

    else:  # "quantile" sobre todos os valores
        quantis = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        classes = np.zeros(len(y), dtype=int)
        labels  = []
        for i in range(n_bins):
            lo, hi = quantis[i], quantis[i + 1]
            mask = (y >= lo) & (y <= hi) if i == n_bins - 1 else (y >= lo) & (y < hi)
            classes[mask] = i
            labels.append(f"{i}\n[{lo:.2e},{hi:.2e}]")
        return classes, labels


# ══════════════════════════════════════════════════════════════════════════════
# Estimadores e espaços de busca
# ══════════════════════════════════════════════════════════════════════════════

def get_classifiers():
    clfs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000,
                               class_weight="balanced", random_state=42),
            {"C": loguniform(1e-3, 100),
             "solver": ["lbfgs", "saga"],
             "penalty": ["l2"]}
        ),
        "Random_Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"n_estimators":    [200, 400, 600],
             "max_depth":       [6, 8, 12, None],
             "min_samples_leaf":[1, 3, 5, 10],
             "max_features":    ["sqrt", "log2", 0.3]}
        ),
        "GradientBoosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
            {"max_depth":       [3, 4, 5, 6],
             "learning_rate":   loguniform(0.01, 0.3),
             "max_iter":        [200, 400, 600],
             "min_samples_leaf":[5, 10, 20],
             "l2_regularization": loguniform(1e-4, 1.0)}
        ),
    }
    if HAS_XGBOOST:
        clfs["XGBoost"] = (
            XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                          tree_method="hist", random_state=42, verbosity=0),
            {"n_estimators":     [200, 400, 600],
             "max_depth":        [3, 4, 5, 6],
             "learning_rate":    loguniform(0.01, 0.3),
             "subsample":        uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.6, 0.4),
             "reg_alpha":        loguniform(1e-4, 10),
             "scale_pos_weight": [1]}
        )
    else:
        print("  XGBoost não instalado (pip install xgboost)")
    return clfs


# ══════════════════════════════════════════════════════════════════════════════
# Métricas
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_clf(y_true, y_pred, label, class_labels):
    acc_w  = accuracy_score(y_true, y_pred,
                            sample_weight=np.ones(len(y_true)))  # acurácia simples
    # Acurácia ponderada: peso inversamente proporcional à frequência da classe
    class_counts = np.bincount(y_true, minlength=len(class_labels))
    weights = 1.0 / np.maximum(class_counts[y_true], 1)
    acc_pond = accuracy_score(y_true, y_pred, sample_weight=weights)
    f1_w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    return {
        "modelo":           label,
        "acuracia":         round(float(acc_w),    4),
        "acuracia_ponderada": round(float(acc_pond), 4),
        "f1_weighted":      round(float(f1_w),    4),
        "f1_macro":         round(float(f1_mac),  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plots — matriz de confusão e métricas
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results: list[dict], cms: dict, class_labels: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5*n + 1, 5))
    if n == 1:
        axes = [axes]
    palette = ["#E67E22","#2980B9","#27AE60","#8E44AD"]

    for ax, (name, cm) in zip(axes, cms.items()):
        # Normalizar por linha (taxa de acerto por classe real)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(name.replace("_"," "), fontsize=10, fontweight="bold")
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                val = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{val}\n({pct:.0%})", ha="center", va="center",
                        fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.7, label="% da classe real")

    fig.suptitle("Matrizes de confusão — classificação de faixas de SRAG\n"
                 "(normalizada por linha: diagonal = taxa de acerto por classe)",
                 fontsize=11)
    plt.tight_layout()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGS / "clf_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ clf_confusion_matrix.png")


def plot_clf_metrics(results_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["acuracia", "acuracia_ponderada", "f1_weighted", "f1_macro"]
    palette = ["#E67E22","#2980B9","#27AE60","#8E44AD"][:len(results_df)]
    labels  = results_df["modelo"].str.replace("_"," ").tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics)+1, 4))
    for ax, m in zip(axes, metrics):
        vals = results_df[m].values
        bars = ax.bar(labels, vals, color=palette, edgecolor="white", width=0.55)
        ax.set_ylim(0, 1.1)
        ax.axhline(1/len(results_df), color="gray", ls="--", lw=0.8,
                   label="Baseline aleatório")
        ax.set_title(m.replace("_","\n"), fontsize=9, fontweight="bold")
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
                    ha="center", fontsize=8, fontweight="bold")
    fig.suptitle("Métricas de classificação — conjunto de teste (fora da amostra)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "clf_metricas.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ clf_metricas.png")


def plot_class_distribution(y_clf_tr, y_clf_te, class_labels, pcfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in [(axes[0], y_clf_tr, "Treino"), (axes[1], y_clf_te, "Teste")]:
        counts = np.bincount(y, minlength=len(class_labels))
        ax.bar(range(len(class_labels)), counts, color="#2980B9", edgecolor="white")
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_ylabel("Nº de observações")
        ax.set_title(f"Distribuição das classes — {title}")
        for i, c in enumerate(counts):
            ax.text(i, c+20, str(c), ha="center", fontsize=8)
    strategy = pcfg.get("classification_strategy","zero_plus_quantile")
    n_bins   = pcfg.get("classification_n_bins", 4)
    fig.suptitle(f"Distribuição das faixas  |  strategy={strategy}  n_bins={n_bins}", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "clf_distribuicao_classes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("✓ clf_distribuicao_classes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_classification():
    vcfg, pcfg = load_config()
    n_jobs     = pcfg.get("n_processadores", 4)
    n_iter     = pcfg.get("hparam_n_iter", 50)
    n_folds    = pcfg.get("hparam_cv_folds", 5)

    panel = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    target, feature_cols = get_feature_cols(panel, vcfg)
    mp = panel[panel[target].notna()].copy()

    if pcfg.get("apenas_municipios_com_casos", False):
        muns  = mp.groupby("cod_mun")[target].max().pipe(lambda s: s[s > 0]).index
        antes = mp["cod_mun"].nunique()
        mp    = mp[mp["cod_mun"].isin(muns)]
        print(f"  Filtro ativo: {len(muns)}/{antes} municípios com casos")

    mp = impute_features(mp, feature_cols)

    n_hold     = pcfg.get("test_holdout_months", 6)
    test_dates = set(sorted(mp["date"].unique())[-n_hold:])
    train = mp[~mp["date"].isin(test_dates)].copy()
    test  = mp[ mp["date"].isin(test_dates)].copy()

    Xtr, ytr_cont = train[feature_cols].values, train[target].values
    Xte, yte_cont = test[feature_cols].values,  test[target].values
    dates_tr = train["date"].astype(str).values

    # Discretizar usando o conjunto de treino como referência
    y_all = np.concatenate([ytr_cont, yte_cont])
    ytr_clf, class_labels = discretize_target(y_all, pcfg)
    ytr_clf = ytr_clf[:len(ytr_cont)]
    yte_clf, _ = discretize_target(yte_cont, pcfg)
    # Garantir que yte_clf usa os mesmos limiares do treino
    yte_clf_base, _ = discretize_target(y_all, pcfg)
    yte_clf = yte_clf_base[len(ytr_cont):]

    print(f"\n  Faixas ({pcfg.get('classification_strategy')}): {class_labels}")
    print(f"  Distribuição treino: { {i: int((ytr_clf==i).sum()) for i in range(len(class_labels))} }")
    print(f"  Distribuição teste:  { {i: int((yte_clf==i).sum()) for i in range(len(class_labels))} }")

    plot_class_distribution(ytr_clf, yte_clf, class_labels, pcfg)

    cv = PanelTimeSeriesSplit(n_splits=n_folds)
    classifiers = get_classifiers()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    results, cms, search_rows = [], {}, []

    for name, (est, param_dist) in classifiers.items():
        print(f"\n  [{name}] RandomizedSearchCV n_iter={n_iter} cv={n_folds}...")
        search = RandomizedSearchCV(
            est, param_dist, n_iter=n_iter,
            scoring="f1_weighted",
            cv=cv, n_jobs=n_jobs, random_state=42, refit=True,
        )
        search.fit(Xtr, ytr_clf, groups=dates_tr)
        best = search.best_estimator_
        print(f"    → melhor F1-weighted CV: {search.best_score_:.4f}")

        y_pred = best.predict(Xte)
        results.append(evaluate_clf(yte_clf, y_pred, name, class_labels))
        cms[name] = confusion_matrix(yte_clf, y_pred, labels=range(len(class_labels)))

        # Relatório completo por classe
        report = classification_report(yte_clf, y_pred,
                                        target_names=[f"Faixa {i}" for i in range(len(class_labels))],
                                        zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(OUT_TABLES / f"clf_{name}_report.csv")

        joblib.dump(best, OUT_MODELS / f"clf_{name}.pkl")
        search_rows.append({"modelo": name, "f1_weighted_cv": round(search.best_score_, 4),
                             **{f"param_{k}": v for k, v in search.best_params_.items()}})

    pd.DataFrame(search_rows).to_csv(OUT_TABLES / "clf_hparam_results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_TABLES / "clf_metricas_global.csv", index=False)

    pd.Series(class_labels, name="label").to_csv(OUT_TABLES / "clf_class_labels.csv", index=True)

    plot_confusion_matrices(results, cms, class_labels)
    plot_clf_metrics(results_df)

    print("\n=== Métricas de classificação (teste) ===")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_classification()
