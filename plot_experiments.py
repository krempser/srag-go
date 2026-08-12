"""
plot_experiments.py — Figuras para artigo/apresentação
=======================================================
Produz 4 figuras limpas e publicáveis:
  fig1_metricas.png          — heatmap comparativo de métricas
  fig2_importancia.png       — importância das variáveis (sem baseline)
  fig3_shap.png              — SHAP aproximado por permutation importance
  fig4_mapas_quintil.png     — mapas por quintil de risco (sempre mostra variação)

USO:
  python plot_experiments.py --cenarios S05,S06,S07
"""

import sys, json, joblib, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from sklearn.inspection import permutation_importance

ROOT     = Path(__file__).resolve().parents[0]
SRC      = ROOT / "src"
EXP_BASE = ROOT / "outputs" / "experiments"
OUT      = ROOT / "outputs" / "figures"
PROC     = ROOT / "data" / "processed"
sys.path.insert(0, str(SRC))

# Rótulos legíveis para cada feature
LABELS = {
    "baseline_srag_municipio":         "Baseline SRAG",
    "year_trend":                      "Tendência temporal",
    "month_sin":                       "Sazonalidade (sen)",
    "month_cos":                       "Sazonalidade (cos)",
    "fase_covid":                      "Fase pandêmica",
    "meteo_umidade":                   "Umidade relativa (%)",
    "meteo_temp_media":                "Temperatura média (°C)",
    "meteo_temp_min":                  "Temperatura mínima (°C)",
    "meteo_temp_max":                  "Temperatura máxima (°C)",
    "meteo_temp_amplitude":            "Amplitude térmica (°C)",
    "meteo_precipitacao":              "Precipitação (mm)",
    "fogo_pct_territorio":             "Queimadas (% território)",
    "mapbiomas_pct_forest":            "Cobertura florestal (%)",
    "mapbiomas_pct_non_forest_natural":"Cerrado/campo natural (%)",
    "mapbiomas_pct_non_vegetated":     "Área urbanizada (%)",
    "mapbiomas_pct_water":             "Corpos d'água (%)",
    "mapbiomas_pct_farming":           "Agropecuária (%)",
    "pct_pop_60mais":                  "Pop. ≥60 anos (%)",
    "pib_per_capita":                  "PIB per capita",
    "cadunico_pct_pop":                "CadÚnico (% pop.)",
    "caged_taxa_pop":                  "Saldo empregos/hab.",
    "cobertura_aps":                   "Cobertura APS (%)",
}

PALETTE = ["#E67E22","#2980B9","#27AE60","#8E44AD"]

plt.rcParams.update({
    "font.family":"DejaVu Sans",
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.alpha":0.25,
})


def short_label(sid, scen):
    lbl = scen.get("label","").replace("seasonal_humidity","Sazonal").replace("monthly","Mensal")
    lbl = lbl.replace("social_and_environmental","Soc+Amb").replace("environmental_only","Amb")
    lbl = lbl.replace("cases_only","c/casos").replace("all","todos")
    lbl = lbl.replace("__"," | ")
    return f"{sid}\n{lbl}"


def load_geojson():
    for p in [ROOT/"data/external/ibge_municipios_go.geojson",
              ROOT/"data/external/goias_municipios.geojson"]:
        if not p.exists(): continue
        with open(p, encoding="utf-8") as f:
            gj = json.load(f)
        if len(gj.get("features",[])) > 50:
            return gj
    return None


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Heatmap de métricas
# ══════════════════════════════════════════════════════════════════════════════

def fig1_metricas():
    summary = pd.read_csv(EXP_BASE/"summary_all_experiments.csv")
    best = (summary.loc[summary.groupby("scenario_id")["Spearman_rho"].idxmax()]
            .sort_values("Spearman_rho", ascending=False)
            .reset_index(drop=True))

    metrics    = ["Spearman_rho","R2","Pearson_r","MAPE_pct","MAE"]
    m_labels   = ["Spearman ρ","R²","Pearson r","MAPE (%)","MAE"]
    better_hi  = [True, True, True, False, False]  # True = maior é melhor

    # Rótulos curtos para linhas
    row_labels = []
    for _, row in best.iterrows():
        lbl = row["scenario_label"].replace("seasonal_humidity","Saz").replace("monthly","Men")
        lbl = lbl.replace("social_and_environmental","S+A").replace("environmental_only","Amb")
        lbl = lbl.replace("cases_only","só casos").replace("all","todos").replace("__"," ")
        row_labels.append(f"{row['scenario_id']}  {lbl}\n(melhor: {row['modelo']})")

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6),
                              gridspec_kw={"wspace":0.05})

    for ax, metric, mlbl, hi in zip(axes, metrics, m_labels, better_hi):
        vals = best[metric].values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if abs(vmax-vmin) < 1e-9: vmin, vmax = vmin-0.01, vmax+0.01

        cmap = matplotlib.colormaps["RdYlGn"] if hi else matplotlib.colormaps["RdYlGn_r"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for i, (v, lbl) in enumerate(zip(vals, row_labels)):
            color = cmap(norm(v)) if not np.isnan(v) else (0.9,0.9,0.9,1)
            ax.barh(i, 1, color=color, edgecolor="white", height=0.9)
            txt_color = "white" if norm(v) < 0.3 or norm(v) > 0.7 else "black"
            ax.text(0.5, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_color)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(best)-0.5)
        ax.set_xticks([])
        ax.set_title(mlbl, fontsize=10, fontweight="bold", pad=8)
        ax.invert_yaxis()

        if ax is axes[0]:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7.5)
        else:
            ax.set_yticks([])

    fig.suptitle("Comparativo de métricas — melhor modelo por cenário\n"
                 "treino: 2019-2023  |  teste: 2024  |  verde = melhor desempenho",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    p = OUT/"fig1_metricas.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Importância de variáveis (excluindo baseline para escala legível)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_importancia(cenarios):
    EXCLUIR = {"baseline_srag_municipio"}  # domina a escala; mostrado separado

    n = len(cenarios)
    fig, axes = plt.subplots(1, n, figsize=(6.5*n, 10),
                              gridspec_kw={"wspace":0.5})
    if n == 1: axes = [axes]

    baseline_vals = {}

    for ax, sid, color in zip(axes, cenarios, PALETTE):
        imp_file = EXP_BASE/sid/"feature_importance.csv"
        if not imp_file.exists():
            ax.text(0.5,0.5,"sem dados", ha="center", transform=ax.transAxes)
            continue

        df = pd.read_csv(imp_file)
        df = df.groupby("feature")["importance"].mean().reset_index()

        # Guardar baseline para anotação separada
        bl_row = df[df["feature"]=="baseline_srag_municipio"]
        if not bl_row.empty:
            baseline_vals[sid] = bl_row["importance"].values[0]

        # Remover baseline e features com label undefined
        df = df[~df["feature"].isin(EXCLUIR)].copy()
        df["label"] = df["feature"].map(lambda f: LABELS.get(f, f))
        df = df[df["importance"] > 0].sort_values("importance")

        bars = ax.barh(df["label"], df["importance"],
                       color=color, edgecolor="white", height=0.7)
        for bar, v in zip(bars, df["importance"]):
            ax.text(v + df["importance"].max()*0.02, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, df["importance"].max() * 1.3)

        with open(EXP_BASE/sid/"scenario.json") as f:
            scen = json.load(f)
        metrics = pd.read_csv(EXP_BASE/sid/"metrics.csv")
        best    = metrics.loc[metrics["Spearman_rho"].idxmax()]
        bl_note = f"(Baseline SRAG excluído: {baseline_vals.get(sid, 0):.3f})" \
                  if sid in baseline_vals else ""
        ax.set_title(
            f"{sid} — {scen.get('label','').replace('__', chr(10))}\n"
            f"{best['modelo']}  ρ={best['Spearman_rho']:.3f}  R²={best['R2']:.3f}\n"
            f"{bl_note}",
            fontsize=8, fontweight="bold")
        ax.set_xlabel("Importância média (Gini)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    # Legenda explicando o que foi excluído
    fig.text(0.5, 0.01,
             "Baseline SRAG município (média histórica) foi excluído desta figura por dominar a escala.\n"
             "Ver Fig. 1 para métricas completas. Valores mostrados são médias sobre os modelos com feature_importances_.",
             ha="center", fontsize=7.5, style="italic", color="gray")

    fig.suptitle("Importância das variáveis — variáveis ambientais, climáticas e sociais\n"
                 "(baseline histórico de SRAG excluído para melhor visualização das demais)",
                 fontsize=11)
    plt.tight_layout(rect=[0,0.04,1,0.96])
    p = OUT/"fig2_importancia.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — SHAP aproximado (permutation importance, sem baseline)
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_test(sid):
    import yaml
    with open(ROOT/"config/variables.yaml") as f: vcfg = yaml.safe_load(f)
    with open(ROOT/"config/periods.yaml")   as f: pcfg = yaml.safe_load(f)
    with open(EXP_BASE/sid/"scenario.json") as f: scen = json.load(f)

    feats  = joblib.load(EXP_BASE/sid/"selected_features.pkl")
    panel  = pd.read_csv(PROC/"panel_monthly.csv", parse_dates=["date"])
    target = vcfg.get("model_target","srag_taxa_casos")
    obs    = panel[panel[target].notna()].copy()

    from models import impute_features
    avail = [f for f in feats if f in obs.columns]
    obs   = impute_features(obs, avail)

    if scen.get("municipality_filter") == "cases_only":
        muns = obs.groupby("cod_mun")[target].max().pipe(lambda s: s[s>0]).index
        obs  = obs[obs["cod_mun"].isin(muns)].copy()

    if scen.get("temporal_grouping") == "seasonal_humidity":
        from seasonal_grouping import build_seasonal_panel
        obs, _, _ = build_seasonal_panel(obs, target, avail, pcfg.get("seasonal",{}))
        obs["year"] = pd.to_datetime(obs["date"]).dt.year
        test = obs[obs["year"] == obs["year"].max()].copy()
    else:
        n_hold = pcfg.get("test_holdout_months",6)
        td     = set(sorted(obs["date"].unique())[-n_hold:])
        test   = obs[obs["date"].isin(td)].copy()

    for f in feats:
        if f not in test.columns: test[f] = 0.0
    return test[feats].fillna(0).values, test[target].values, feats



def fig3_pdp(cenarios):
    """
    Partial Dependence Plots — mostram como a predição muda quando o valor
    de uma variável varia de baixo a alto, mantendo todas as outras fixas.
    Responde: "umidade alta aumenta ou diminui o risco de SRAG?"
    """
    from sklearn.inspection import partial_dependence

    CLIMA_FEATURES = [
        "meteo_umidade","meteo_temp_media","meteo_precipitacao",
        "meteo_temp_amplitude","mapbiomas_pct_non_forest_natural",
        "mapbiomas_pct_farming","fogo_pct_territorio",
        "mapbiomas_pct_non_vegetated",
    ]

    for sid in cenarios:
        try:
            feats    = joblib.load(EXP_BASE/sid/"selected_features.pkl")
            metrics  = pd.read_csv(EXP_BASE/sid/"metrics.csv")
            best_mod = metrics.loc[metrics["Spearman_rho"].idxmax(),"modelo"]
            model    = joblib.load(EXP_BASE/sid/f"model_{best_mod}.pkl")
            X_te, y_te, feats = rebuild_test(sid)

            feats_plot = [(f, LABELS.get(f,f)) for f in CLIMA_FEATURES if f in feats]
            if not feats_plot:
                print(f"  [{sid}] sem features climáticas disponíveis")
                continue

            n  = len(feats_plot)
            nc = min(4, n)
            nr = (n + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(4.5*nc, 3.5*nr+1.5), squeeze=False)
            flat = axes.flatten()

            print(f"  [{sid}] partial dependence ({n} features)...", flush=True)
            for i, (feat, lbl) in enumerate(feats_plot):
                ax  = flat[i]
                idx = feats.index(feat)
                res = partial_dependence(model, X_te, [idx], kind="average", grid_resolution=50)
                xs  = res["grid_values"][0]
                ys  = res["average"][0]

                ax.plot(xs, ys, color="#C0392B", lw=2.5)
                ax.fill_between(xs, ys, ys.mean(), alpha=0.15, color="#C0392B")
                ax.axhline(ys.mean(), color="gray", lw=1, ls="--")
                ax.set_xlabel(lbl, fontsize=9)
                ax.set_ylabel("Taxa SRAG/pop\n(predita)", fontsize=8)
                ax.tick_params(labelsize=8)
                delta = ys[-1] - ys[0]
                arrow = "risco SOBE com a variavel" if delta > 0 else "risco CAI com a variavel"
                cor   = "#C0392B" if delta > 0 else "#2980B9"
                ax.text(0.05, 0.90, arrow, transform=ax.transAxes, fontsize=7,
                        color=cor, bbox=dict(boxstyle="round", facecolor="white",
                                             alpha=0.85, edgecolor=cor))

            for j in range(n, len(flat)):
                flat[j].set_visible(False)

            with open(EXP_BASE/sid/"scenario.json") as f:
                scen = json.load(f)
            best = metrics.loc[metrics["Spearman_rho"].idxmax()]
            titulo = (f"Partial Dependence Plots --- {sid}\n"
                      f"{scen.get('label','').replace('__',' | ')}\n"
                      f"Modelo: {best['modelo']}  rho={best['Spearman_rho']:.3f}  R2={best['R2']:.3f}\n"
                      "Cada curva: efeito de UMA variavel na predicao (demais fixas na media)")
            fig.suptitle(titulo, fontsize=9, y=1.02)
            plt.tight_layout()
            p = OUT/f"fig3_pdp_{sid}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  OK {p.name}")
        except Exception as e:
            print(f"  ERRO {sid}: {e}")
            import traceback; traceback.print_exc()




def fig4_mapas_quintil(cenarios):
    gj = load_geojson()
    if gj is None:
        print("  GeoJSON não encontrado — fig4 pulada")
        return

    panel = pd.read_csv(PROC/"panel_monthly.csv", parse_dates=["date"])
    obs   = panel[panel["srag_taxa_casos"].notna()]
    hist  = obs.groupby("cod_mun")["srag_taxa_casos"].mean()

    # Quintis da taxa histórica real
    quintil_num = pd.qcut(hist, q=5, labels=False, duplicates="drop")
    # Remapear para 0-4 mesmo que alguns bins tenham sido colapsados
    unique_vals = sorted(quintil_num.dropna().unique())
    remap = {v: i for i, v in enumerate(unique_vals)}
    quintil_num = quintil_num.map(remap)

    cmap5 = matplotlib.colormaps.get_cmap("YlOrRd").resampled(5)
    cores_quintil = [mcolors.to_hex(cmap5(i/4)) for i in range(5)]

    n = len(cenarios)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 9))
    if n == 1: axes = [axes]

    for ax, sid in zip(axes, cenarios):
        patches, colors = [], []
        top5_muns = hist.nlargest(5)

        for feat in gj.get("features",[]):
            props = feat.get("properties",{})
            cod6  = str(props.get("id") or props.get("codarea") or
                        props.get("CD_MUN") or "").strip()[:6]
            q     = quintil_num.get(cod6)
            geom  = feat.get("geometry",{})
            rings = ([geom["coordinates"][0]] if geom["type"]=="Polygon"
                     else [p[0] for p in geom["coordinates"]])
            for ring in rings:
                try:
                    patches.append(MplPolygon(np.array(ring, dtype=float)))
                    colors.append(int(q) if q is not None else -1)
                except: continue

        if not patches: continue
        arr  = np.array(colors, dtype=float)
        norm = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], 5)
        pc   = PatchCollection(patches, array=np.where(arr<0,0,arr),
                               cmap=cmap5, norm=norm,
                               edgecolor="white", linewidth=0.2)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()

        with open(EXP_BASE/sid/"scenario.json") as f:
            scen = json.load(f)
        metrics = pd.read_csv(EXP_BASE/sid/"metrics.csv")
        best    = metrics.loc[metrics["Spearman_rho"].idxmax()]
        ax.set_title(
            f"{sid}\n{scen.get('label','').replace('__',chr(10))}\n"
            f"ρ={best['Spearman_rho']:.3f}  R²={best['R2']:.3f}",
            fontsize=9, fontweight="bold", pad=6)

        # Colorbar com quintis
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=cores_quintil[i],
                         label=["Q1 Muito baixo","Q2 Baixo","Q3 Médio",
                                "Q4 Alto","Q5 Muito alto"][i])
                   for i in range(5)]
        ax.legend(handles=handles, loc="lower left", fontsize=7,
                  title="Risco histórico\nde SRAG", title_fontsize=7.5,
                  framealpha=0.9)

    # Nota sobre os municípios de maior risco
    top5_str = ", ".join([f"{obs.groupby('cod_mun')['srag_taxa_casos'].mean().nlargest(5).index[i]}"
                          for i in range(min(5, len(obs.groupby('cod_mun'))))])

    fig.suptitle("Estratificação de risco estrutural de SRAG — Goiás\n"
                 "Coloração por quintil da taxa histórica (Q1=menor risco, Q5=maior risco)",
                 fontsize=11)
    plt.tight_layout()
    p = OUT/"fig4_mapas_quintil.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    cenarios = [s.strip() for s in args.cenarios.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n=== plot_experiments.py | cenários: {cenarios} ===")
    print("[1/4] Heatmap de métricas...")
    fig1_metricas()
    print("[2/4] Importância de variáveis (sem baseline)...")
    fig2_importancia(cenarios)
    print("[3/4] Partial Dependence Plots...")
    fig3_pdp(cenarios)
    print("[4/4] Mapas por quintil...")
    fig4_mapas_quintil(cenarios)
    print(f"\nSaídas: {OUT}/fig1_metricas.png  fig2_importancia.png  fig3_shap.png  fig4_mapas_quintil.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cenarios", default="S05,S06,S07")
    run(p.parse_args())
