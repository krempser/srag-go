"""
evaluate_shape.py — Avaliação por município ("shape") com mapa coroplético.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT       = Path(__file__).resolve().parents[1]
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"
EXTERNAL   = ROOT / "data" / "external"

GEOJSON_CANDIDATES = [
    EXTERNAL / "ibge_municipios_go.geojson",  # baixado pelo fire script (246 municípios)
    EXTERNAL / "goias_municipios.geojson",     # GeoJSON do usuário
]




def _resolve_geojson():
    for p in GEOJSON_CANDIDATES:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            n = len(json.load(f).get("features", []))
        if n > 50:
            return p
    return None


def per_municipio_metrics(preds_df):
    model_cols = [c for c in preds_df.columns if c.startswith("pred_")]
    rows = []
    for (cod_mun, municipio), g in preds_df.groupby(["cod_mun", "municipio"]):
        row = {"cod_mun": cod_mun, "municipio": municipio,
               "n_meses_teste": len(g), "casos_reais_total": g["y_true"].sum()}
        for col in model_cols:
            err = g[col] - g["y_true"]
            row[f"MAE_{col}"]  = err.abs().mean()
            row[f"RMSE_{col}"] = (err**2).mean()**0.5
            row[f"vies_{col}"] = err.mean()
            mask = g["y_true"] > 0
            row[f"MAPE_{col}"] = ((err[mask].abs() / g["y_true"][mask]).mean() * 100
                                   if mask.sum() > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def try_plot_choropleth(metric_df, value_col, title, out_path):
    geojson_path = _resolve_geojson()
    if geojson_path is None:
        return False

    try:
        cmap = matplotlib.colormaps["OrRd"]
    except AttributeError:
        cmap = plt.get_cmap("OrRd")

    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    n_feat = len(gj.get("features", []))
    print(f"  GeoJSON municipal: {n_feat} features ({geojson_path.name})")

    lookup = metric_df.set_index("cod_mun")[value_col].to_dict()
    vmin, vmax = metric_df[value_col].min(), metric_df[value_col].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 10))
    patches, colors = [], []

    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        raw_id = str(props.get("id") or props.get("codarea") or
                     props.get("codigo_ibge") or props.get("CD_MUN") or
                     props.get("CD_GEOCODM") or "")
        cod6 = raw_id.strip()[:6]
        val  = lookup.get(cod6, np.nan)

        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        coords    = geom.get("coordinates", [])

        if geom_type == "Polygon":
            rings = [coords[0]]
        elif geom_type == "MultiPolygon":
            rings = [poly[0] for poly in coords]
        else:
            continue

        for ring in rings:
            try:
                patches.append(Polygon(np.array(ring, dtype=float)))
                colors.append(val)
            except Exception:
                continue

    if not patches:
        print("  Nenhum polígono extraído — verificar GeoJSON")
        plt.close(fig)
        return False

    pc = PatchCollection(patches, array=np.array(colors, dtype=float),
                         cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=8)
    fig.colorbar(pc, ax=ax, shrink=0.6, label=value_col)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_ranked_bar(metric_df, value_col, title, out_path, n=15):
    s = metric_df[["municipio", value_col]].dropna().sort_values(value_col)
    top_good = s.head(n).sort_values(value_col, ascending=False)
    top_bad  = s.tail(n).sort_values(value_col, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].barh(top_good["municipio"], top_good[value_col], color="#27ae60")
    axes[0].set_title(f"{n} menores erros")
    axes[0].set_xlabel(value_col)

    axes[1].barh(top_bad["municipio"], top_bad[value_col], color="#c0392b")
    axes[1].set_title(f"{n} maiores erros")
    axes[1].set_xlabel(value_col)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_shape_evaluation():
    preds_df = pd.read_csv(OUT_TABLES / "predicoes_teste.csv")
    preds_df["municipio"] = preds_df["municipio"].astype(str)
    model_cols = [c for c in preds_df.columns if c.startswith("pred_")]

    metric_df = per_municipio_metrics(preds_df)

    # Ordenar pelo MAE do primeiro modelo disponível
    first_mae = f"MAE_{model_cols[0]}"
    metric_df = metric_df.sort_values(first_mae)
    metric_df.to_csv(OUT_TABLES / "avaliacao_por_municipio.csv", index=False)

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    used_real_map = False
    for pred_col in model_cols:
        mae_col = f"MAE_{pred_col}"
        label   = pred_col.replace("pred_", "")
        title   = f"MAE por município — {label.replace('_', ' ')} (teste)"
        ok = try_plot_choropleth(metric_df, mae_col, title,
                                  OUT_FIGS / f"mapa_erro_{label}.png")
        used_real_map = used_real_map or ok
        plot_ranked_bar(metric_df, mae_col, title,
                        OUT_FIGS / f"ranking_erro_{label}.png")

    return metric_df, used_real_map


if __name__ == "__main__":
    metric_df, used_real_map = run_shape_evaluation()
    print("Mapa real:", used_real_map)
    print(metric_df.head(5).to_string())
