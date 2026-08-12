"""eval_variables.py — avaliação prévia das variáveis (completude, correlação, VIF)"""
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"

ID_COLS = {"cod_mun", "date", "municipio"}


def load_config():
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_vif(df_features):
    X = df_features.dropna()
    cols = X.columns.tolist()
    if len(X) < 10 or len(cols) < 2:
        return pd.DataFrame(columns=["variavel", "VIF"])
    rows = []
    for col in cols:
        y = X[col].values
        others = X.drop(columns=[col]).values
        try:
            r2 = LinearRegression().fit(others, y).score(others, y)
            vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        except Exception:
            vif = np.nan
        rows.append({"variavel": col, "VIF": round(vif, 1) if np.isfinite(vif) else vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def run_variable_evaluation():
    cfg = load_config()
    panel = pd.read_csv(PROCESSED_DIR / "panel_monthly.csv", parse_dates=["date"])

    target = cfg.get("model_target", "srag_taxa_casos")
    exclude = set(cfg.get("model_features_exclude", [])) | ID_COLS | {target}
    feature_cols = [c for c in panel.columns if c not in exclude and panel[c].dtype != object]

    model_panel = panel[panel[target].notna()].copy()

    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)
    if pcfg.get("apenas_municipios_com_casos", False):
        muns_com_caso = (model_panel.groupby("cod_mun")[target]
                         .max().pipe(lambda s: s[s > 0]).index)
        model_panel = model_panel[model_panel["cod_mun"].isin(muns_com_caso)]

    # 1) Missingness
    miss = model_panel[feature_cols].isna().mean().mul(100).round(1) \
           .rename("pct_faltante_janela_modelagem")

    # 2) Descritiva
    desc = model_panel[feature_cols].describe().T[["mean","std","min","50%","max"]]
    desc.columns = ["media","desvio_padrao","minimo","mediana","maximo"]

    # 3) Correlação Spearman com o alvo
    corr_rows = []
    for col in feature_cols:
        sub = model_panel[[col, target]].dropna()
        rho = sub[col].corr(sub[target], method="spearman") if len(sub) > 30 else np.nan
        corr_rows.append({"variavel": col, f"spearman_vs_{target}": round(rho, 3) if pd.notna(rho) else np.nan})
    corr_df = pd.DataFrame(corr_rows).set_index("variavel")

    summary = pd.concat([miss, desc, corr_df], axis=1).reset_index().rename(columns={"index":"variavel"})

    # 4) VIF (excluir has_uti que é binária)
    vif_cols = [c for c in feature_cols if c != "has_uti"]
    vif_df = compute_vif(model_panel[vif_cols])

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLES / "avaliacao_previa_variaveis.csv", index=False)
    vif_df.to_csv(OUT_TABLES / "vif_variaveis.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    s = summary.set_index("variavel")["pct_faltante_janela_modelagem"].sort_values()
    ax.barh(s.index, s.values, color="#c0392b")
    ax.set_xlabel("% faltante na janela de modelagem")
    ax.set_title("Completude das variáveis (features do modelo)")
    plt.tight_layout(); fig.savefig(OUT_FIGS / "completude_variaveis.png", dpi=130); plt.close(fig)

    corr_col = f"spearman_vs_{target}"
    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols)*0.35)))
    s = summary.set_index("variavel")[corr_col].dropna().sort_values()
    colors = ["#2980b9" if v >= 0 else "#c0392b" for v in s.values]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Correlação de Spearman com {target}")
    ax.set_title("Correlação (univariada) de cada feature com o alvo")
    plt.tight_layout(); fig.savefig(OUT_FIGS / "correlacao_variaveis_alvo.png", dpi=130); plt.close(fig)

    return summary, vif_df


if __name__ == "__main__":
    summary, vif_df = run_variable_evaluation()
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    print("\nVIF:"); print(vif_df.to_string(index=False))
