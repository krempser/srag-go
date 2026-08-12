"""
simulate.py — Simulação de cenários climáticos de SRAG
=======================================================
Usa os modelos treinados para projetar mudança no risco de SRAG
dado novos valores de variáveis climáticas e ambientais.

ENTRADA (escolha um por variável):
  Valor absoluto:    --umidade 35.0
  Delta percentual:  --delta_umidade_pct -20   (% sobre média histórica do município)
  Delta absoluto:    --delta_umidade_abs -10   (unidade sobre média histórica)

USO:
  python simulate.py --experiment S06 --delta_umidade_pct -20 --delta_precipitacao_pct -30

IMPORTANTE — INTERPRETAÇÃO DOS RESULTADOS:
  O modelo inclui termos autorregressivos (lags de SRAG dos meses anteriores)
  que explicam ~70-80% da variância. As variáveis climáticas explicam os ~20-30%
  restantes. Por isso:
  - A variação absoluta entre municípios no cenário simulado é pequena (~1-5%)
  - O que o modelo captura de clima é o EFEITO RELATIVO entre municípios:
    quais se tornam proporcionalmente mais vulneráveis com a mudança climática
  - Para simulação puramente climática sem confundimento temporal, use o
    modelo atemporal (outputs/models/atemporal_*.pkl) ou crie um experimento
    sem lags (remova lag_* de TEMPORAL_FEATURES em run_experiments.py)
"""

import sys, json, yaml, joblib, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from scipy.stats import spearmanr

ROOT      = Path(__file__).resolve().parents[0]
SRC       = ROOT / "src"
PROCESSED = ROOT / "data" / "processed"
EXP_BASE  = ROOT / "outputs" / "experiments"
SIM_OUT   = ROOT / "outputs" / "simulation"
sys.path.insert(0, str(SRC))

VAR_NAMES = {
    "umidade":        "meteo_umidade",
    "temp_media":     "meteo_temp_media",
    "temp_min":       "meteo_temp_min",
    "temp_max":       "meteo_temp_max",
    "temp_amplitude": "meteo_temp_amplitude",
    "precipitacao":   "meteo_precipitacao",
    "fogo":           "fogo_pct_territorio",
    "pct_farming":    "mapbiomas_pct_farming",
    "pct_forest":     "mapbiomas_pct_forest",
    "pct_cerrado":    "mapbiomas_pct_non_forest_natural",
    "pct_urbano":     "mapbiomas_pct_non_vegetated",
    "pct_agua":       "mapbiomas_pct_water",
}

# ── Carregamento ──────────────────────────────────────────────────────────────

def find_best_experiment(exp_cfg):
    s = pd.read_csv(EXP_BASE / "summary_all_experiments.csv")
    return s.groupby("scenario_id")["Spearman_rho"].max().idxmax()


def load_experiment(exp_id):
    d = EXP_BASE / exp_id
    feats   = joblib.load(d / "selected_features.pkl")
    medians = pd.read_csv(d/"feature_medians.csv", index_col=0, header=None).squeeze()
    models  = {p.stem.replace("model_",""): joblib.load(p)
               for p in sorted(d.glob("model_*.pkl"))}
    metrics = pd.read_csv(d / "metrics.csv")
    best    = metrics.loc[metrics["Spearman_rho"].idxmax(), "modelo"]
    with open(d / "scenario.json") as f:
        scenario = json.load(f)
    return feats, medians, models, best, scenario


# ── Painel de simulação ───────────────────────────────────────────────────────

def build_panel(selected_features, medians):
    """
    Constrói painel com 1 linha por município usando medianas históricas.
    Lags recebem o valor do baseline histórico do município
    (melhor estimativa do nível recente sem usar dados futuros).
    Features temporais fixas → cenário futuro pós-COVID.
    """
    panel    = pd.read_csv(PROCESSED/"panel_monthly.csv", parse_dates=["date"])
    muns     = panel[["cod_mun","municipio","populacao"]].drop_duplicates("cod_mun")
    obs      = panel[panel["srag_taxa_casos"].notna()].copy()

    cols_ok  = [f for f in selected_features if f in obs.columns]
    feat_med = obs.groupby("cod_mun")[cols_ok].median()

    for f in selected_features:
        if f not in feat_med.columns:
            feat_med[f] = medians.get(f, 0.0)
        else:
            feat_med[f] = feat_med[f].fillna(medians.get(f, feat_med[f].median()))

    sim = muns.set_index("cod_mun").join(feat_med, how="left").reset_index()
    sim["populacao"] = sim["populacao"].fillna(sim["populacao"].median())

    # Features temporais fixas
    if "year_trend"  in selected_features: sim["year_trend"]  = 1.0
    if "fase_covid"  in selected_features: sim["fase_covid"]  = 2.0
    if "month_sin"   in selected_features: sim["month_sin"]   = 0.0
    if "month_cos"   in selected_features: sim["month_cos"]   = 0.0

    for f in selected_features:
        if f not in sim.columns:
            sim[f] = medians.get(f, 0.0)

    return sim


def apply_overrides(sim, selected_features, o_abs, o_pct, o_delta):
    obs     = pd.read_csv(PROCESSED/"panel_monthly.csv")
    obs     = obs[obs["srag_taxa_casos"].notna()]
    applied = []

    for feat, val in o_abs.items():
        if feat in selected_features:
            sim[feat] = float(val)
            applied.append(f"{feat.split('_',1)[-1]}={val:.1f}")

    for feat, pct in o_pct.items():
        if feat in selected_features:
            if feat in obs.columns:
                hm = obs.groupby("cod_mun")[feat].mean()
                sim[feat] = sim["cod_mun"].map(hm).fillna(
                    sim[feat]) * (1 + pct/100)
            else:
                sim[feat] = sim[feat] * (1 + pct/100)
            applied.append(f"{feat.split('_',1)[-1]} {pct:+.1f}%")

    for feat, d in o_delta.items():
        if feat in selected_features:
            if feat in obs.columns:
                hm = obs.groupby("cod_mun")[feat].mean()
                sim[feat] = sim["cod_mun"].map(hm).fillna(sim[feat]) + d
            else:
                sim[feat] = sim[feat] + d
            applied.append(f"{feat.split('_',1)[-1]} {d:+.2f}")

    return sim, applied


def predict(sim, models, best, feats):
    X = sim[feats].values
    for name, model in models.items():
        try:
            sim[f"taxa_{name}"] = np.clip(model.predict(X), 0, None)
            sim[f"casos_{name}"] = (sim[f"taxa_{name}"] * sim["populacao"]).round().astype(int)
        except Exception as e:
            print(f"  AVISO {name}: {e}")
    if f"taxa_{best}" in sim.columns:
        sim["taxa_predita"]    = sim[f"taxa_{best}"]
        sim["casos_esperados"] = sim[f"casos_{best}"]
        sim["rank_risco"]      = sim["taxa_predita"].rank(ascending=False).astype(int)
    return sim


# ── GeoJSON ───────────────────────────────────────────────────────────────────

def load_geojson():
    for p in [ROOT/"data/external/ibge_municipios_go.geojson",
              ROOT/"data/external/goias_municipios.geojson"]:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            gj = json.load(f)
        if len(gj.get("features",[])) > 50:
            return gj
    return None


def choropleth(ax, gj, lookup, cmap_name, norm, title, cbar_label, fig):
    try: cmap = matplotlib.colormaps[cmap_name]
    except: cmap = plt.get_cmap(cmap_name)

    patches, colors = [], []
    for feat in gj.get("features",[]):
        props = feat.get("properties",{})
        cod6  = str(props.get("id") or props.get("codarea") or
                    props.get("CD_MUN") or "").strip()[:6]
        val   = lookup.get(cod6, np.nan)
        geom  = feat.get("geometry",{})
        rings = ([geom["coordinates"][0]] if geom["type"]=="Polygon"
                 else [p[0] for p in geom["coordinates"]])
        for ring in rings:
            try:
                patches.append(MplPolygon(np.array(ring, dtype=float)))
                colors.append(float(val))
            except: continue

    if not patches:
        ax.text(0.5,0.5,"Sem dados para o mapa",ha="center",va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return

    arr = np.array(colors, dtype=float)
    # Substituir NaN pelo limite inferior para não sumir no mapa
    arr_plot = np.where(np.isnan(arr), norm.vmin, arr)

    pc = PatchCollection(patches, array=arr_plot, cmap=cmap, norm=norm,
                         edgecolor="white", linewidth=0.15)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, pad=4)
    fig.colorbar(pc, ax=ax, shrink=0.65, label=cbar_label, format="%.4f")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plots(baseline, simulado, exp_id, scenario_lbl, applied_desc, out_dir):
    df = baseline[["cod_mun","municipio","populacao","taxa_predita"]].copy()
    df = df.rename(columns={"taxa_predita":"taxa_baseline"})
    df["taxa_simulada"]    = simulado["taxa_predita"].values
    df["delta_abs"]        = df["taxa_simulada"] - df["taxa_baseline"]
    df["delta_pct"]        = (df["delta_abs"] /
                              df["taxa_baseline"].replace(0,np.nan) * 100)
    df["casos_baseline"]   = (df["taxa_baseline"] * df["populacao"]).round().astype(int)
    df["casos_simulados"]  = (df["taxa_simulada"] * df["populacao"]).round().astype(int)
    df["delta_casos"]      = df["casos_simulados"] - df["casos_baseline"]
    df.to_csv(out_dir/"projecao_delta.csv", index=False)

    pct = df["delta_pct"].dropna()
    gj  = load_geojson()

    # ── 1. Dois mapas lado a lado ──────────────────────────────────────────
    if gj:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        # Mapa risco simulado — por quantil (sempre mostra variação)
        lookup_sim = df.set_index("cod_mun")["taxa_simulada"].to_dict()
        vals_sim   = [v for v in lookup_sim.values() if pd.notna(v)]
        v1, v2 = np.percentile(vals_sim, 5), np.percentile(vals_sim, 95)
        norm1 = mcolors.Normalize(vmin=v1, vmax=v2)
        choropleth(axes[0], gj, lookup_sim, "YlOrRd", norm1,
                   "Risco SRAG simulado", "Taxa SRAG/pop (P5-P95)", fig)

        # Mapa variação — normalização pelos quantis do delta real
        lookup_d = df.set_index("cod_mun")["delta_pct"].to_dict()
        dvals    = [v for v in lookup_d.values() if pd.notna(v)]
        p5, p95  = np.percentile(dvals, 10), np.percentile(dvals, 90)
        absmax   = max(abs(p5), abs(p95), 0.5)  # mínimo de ±0.5% para colormap funcionar
        norm2 = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        choropleth(axes[1], gj, lookup_d, "RdBu_r", norm2,
                   "Variação no risco (%)\nvermelho=aumento, azul=redução",
                   "Variação % (P10-P90)", fig)

        fig.suptitle(f"Simulação climática — {exp_id}\n{applied_desc}", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir/"mapas_simulacao.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("✓ mapas_simulacao.png")

    # ── 2. Ranking municípios ──────────────────────────────────────────────
    n   = 15
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, data, cor, titulo, sign in [
        (axes[0], df.nlargest(n,"delta_pct"),   "#C0392B", f"Top {n} maior AUMENTO",  1),
        (axes[1], df.nsmallest(n,"delta_pct"),  "#2980B9", f"Top {n} maior REDUÇÃO",  -1),
    ]:
        data = data.sort_values("delta_pct", ascending=(sign < 0))
        vals = data["delta_pct"].values
        ax.barh(range(len(data)), vals, color=cor, edgecolor="white", height=0.65)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["municipio"].str.title().values, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Variação (%)", fontsize=9)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        # Espaço extra na direita/esquerda para os rótulos
        lim = max(abs(vals)) * 1.6 if len(vals) else 1
        ax.set_xlim(-lim if sign < 0 else 0,
                     lim if sign > 0 else 0)
        for i, (v, casos) in enumerate(zip(vals, data["delta_casos"])):
            ax.text(v + sign * abs(vals).max() * 0.03, i,
                    f"{v:+.1f}% / {casos:+d} casos",
                    va="center",
                    ha="left" if sign > 0 else "right",
                    fontsize=7)

    fig.suptitle(f"Impacto climático — {exp_id}  |  {applied_desc}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir/"ranking_variacao.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ ranking_variacao.png")

    # ── 3. Scatter + histograma ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cores = ["#C0392B" if d > 0 else "#2980B9" for d in df["delta_abs"]]
    ax.scatter(df["taxa_baseline"], df["taxa_simulada"],
               c=cores, alpha=0.6, s=22, edgecolors="none")
    lim = max(df[["taxa_baseline","taxa_simulada"]].max()) * 1.05
    ax.plot([0,lim],[0,lim],"k--",lw=1,alpha=0.4, label="Sem variação")
    sr = spearmanr(df["taxa_baseline"], df["taxa_simulada"])[0]
    ax.text(0.04, 0.96, f"ρ = {sr:.4f}", transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    if sr > 0.99:
        ax.text(0.5, 0.06,
                "ρ ≈ 1: lags dominam (~70-80% da variância).\n"
                "Efeito climático real mas pequeno neste modelo.\n"
                "Para simulação climática pura: use modelo atemporal.",
                transform=ax.transAxes, fontsize=7, ha="center",
                color="darkorange",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    ax.set_xlabel("Taxa baseline"); ax.set_ylabel("Taxa simulada")
    ax.set_title("Baseline vs Simulado")
    ax.legend(fontsize=8)

    ax = axes[1]
    counts, edges, patches_h = ax.hist(pct.clip(-30,30), bins=25, edgecolor="white",
                                        color="gray")
    for patch, left in zip(patches_h, edges[:-1]):
        patch.set_facecolor("#C0392B" if left >= 0 else "#2980B9")
    ax.axvline(0,          color="black",  lw=1.5, ls="--")
    ax.axvline(pct.mean(), color="red",    lw=1.2, ls="-",
               label=f"Média {pct.mean():+.1f}%")
    ax.axvline(pct.median(), color="orange", lw=1.2, ls="-",
               label=f"Mediana {pct.median():+.1f}%")
    ax.set_xlabel("Variação no risco (%)")
    ax.set_ylabel("Nº municípios")
    ax.set_title(f"Distribuição  |  ↑{(pct>0).sum()} municípios  "
                 f"↓{(pct<0).sum()} municípios")
    ax.legend(fontsize=8)
    fig.suptitle(applied_desc, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir/"variacao_distribuicao.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("✓ variacao_distribuicao.png")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    with open(ROOT/"config/experiments.yaml", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    exp_id = args.experiment or find_best_experiment(exp_cfg)
    print(f"\n{'='*55}")
    print(f"  SIMULAÇÃO SRAG | {exp_id}")
    print(f"{'='*55}")

    feats, medians, models, best, scenario = load_experiment(exp_id)
    print(f"  Modelos: {list(models.keys())}  |  Melhor: {best}")
    print(f"  Features ({len(feats)}): {feats}")

    o_abs, o_pct, o_delta = {}, {}, {}
    for short, feat in VAR_NAMES.items():
        if getattr(args, short,              None) is not None: o_abs[feat]   = getattr(args, short)
        if getattr(args, f"delta_{short}_pct", None) is not None: o_pct[feat] = getattr(args, f"delta_{short}_pct")
        if getattr(args, f"delta_{short}_abs", None) is not None: o_delta[feat]= getattr(args, f"delta_{short}_abs")

    print("\n  Construindo baseline...")
    bl = build_panel(feats, medians)
    bl = predict(bl.copy(), models, best, feats)

    print("  Construindo cenário simulado...")
    sim = build_panel(feats, medians)
    sim, applied = apply_overrides(sim, feats, o_abs, o_pct, o_delta)
    sim = predict(sim, models, best, feats)

    applied_desc = ", ".join(applied) if applied else "baseline histórico"
    print(f"  Aplicado: {applied_desc}")

    SIM_OUT.mkdir(parents=True, exist_ok=True)
    sim.sort_values("rank_risco").to_csv(SIM_OUT/"projecao_casos.csv", index=False)

    df_var = plots(bl, sim, exp_id, scenario.get("label","?"),
                   applied_desc, SIM_OUT)

    print(f"\n  Top 10 municípios:")
    print(sim.nsmallest(10,"rank_risco")[
        ["rank_risco","municipio","taxa_predita","casos_esperados"]
    ].to_string(index=False))
    print(f"\n  Variação média: {df_var['delta_pct'].mean():+.1f}%")
    print(f"  ↑ {(df_var['delta_pct']>0).sum()} municípios  "
          f"↓ {(df_var['delta_pct']<0).sum()} municípios")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=None)
    for short in VAR_NAMES:
        p.add_argument(f"--{short}",           type=float, default=None)
        p.add_argument(f"--delta_{short}_pct", type=float, default=None)
        p.add_argument(f"--delta_{short}_abs", type=float, default=None)
    run(p.parse_args())
