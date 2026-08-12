"""
build_panel.py — VERSÃO EXPANDIDA
-----------------------------------
Inclui engenharia de features temporais que aumentam o poder preditivo:
  - Lags do alvo (memória epidemiológica)
  - Sazonalidade cíclica (sin/cos do mês)
  - Indicador de fase pandêmica COVID
  - Opção de transformação logarítmica do alvo

Todas as novas features são configuráveis em periods.yaml.
"""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_variable, _norm_cod_mun

ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"
PROCESSED  = ROOT / "data" / "processed"


def load_config():
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        variables_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        periods_cfg = yaml.safe_load(f)
    return variables_cfg, periods_cfg


def build_full_date_range(periods_cfg):
    start = pd.Timestamp(periods_cfg["date_start"] + "-01")
    end   = pd.Timestamp(periods_cfg["date_end"]   + "-01")
    return pd.date_range(start, end, freq="MS")


def build_municipio_master(raw_dir):
    df = pd.read_csv(raw_dir / "pop mensal.csv", encoding="latin1", sep=";")
    df["cod_mun"] = df["cod_mun"].map(_norm_cod_mun)
    out = (df[["cod_mun","Município"]].drop_duplicates()
           .rename(columns={"Município":"municipio"}))
    out["municipio"] = out["municipio"].str.strip()
    return out.reset_index(drop=True)


def series_to_monthly_grid(long_df, full_dates, cod_muns, interpolate):
    wide = long_df.pivot_table(index="cod_mun", columns="date",
                               values="value", aggfunc="mean")
    wide = wide.reindex(index=cod_muns)
    all_cols = sorted(set(wide.columns).union(full_dates))
    wide = wide.reindex(columns=all_cols)
    if interpolate:
        wide = wide.interpolate(method="linear", axis=1, limit_area=None)
        wide = wide.ffill(axis=1).bfill(axis=1)
    wide = wide.reindex(columns=full_dates)
    long_out = wide.stack().rename("value").reset_index()
    long_out.columns = ["cod_mun","date","value"]
    return long_out


def compute_derived_variables(panel, derived_cfg):
    for name, spec in derived_cfg.items():
        t = spec["type"]
        if t == "ratio":
            num = panel[spec["numerator"]] * spec.get("numerator_scale", 1)
            den = panel[spec["denominator"]].replace(0, np.nan)
            panel[name] = num / den * spec.get("multiplier", 1)
        elif t == "binary":
            s = panel[spec["source"]].fillna(0) if spec.get("uti_null_means_zero") \
                else panel[spec["source"]]
            panel[name] = (s > 0).astype(float)
        elif t == "temporal_cyclic":
            panel["_month"] = pd.to_datetime(panel["date"]).dt.month
            period    = spec.get("period", 12)
            component = spec.get("component", "sin")
            angle = 2 * np.pi * panel["_month"] / period
            panel[name] = np.sin(angle) if component == "sin" else np.cos(angle)
            panel.drop(columns=["_month"], inplace=True)
        elif t == "temporal_phase":
            panel["_year"] = pd.to_datetime(panel["date"]).dt.year
            phases = spec.get("phases", {})
            mapping = {}
            for phase_idx, (phase_name, years) in enumerate(phases.items()):
                for y in years:
                    mapping[y] = phase_idx
            panel[name] = panel["_year"].map(mapping).fillna(-1).astype(int)
            panel.drop(columns=["_year"], inplace=True)
    return panel


def add_lag_features(panel, target_col, lag_months, periods_cfg):
    """
    Adiciona lags do alvo como features. Crucial para capturar memória
    epidemiológica: o melhor preditor do SRAG este mês é o SRAG do mês passado.

    IMPORTANTE: os lags são calculados DENTRO do grupo município para não
    vazar informação de um município para outro.
    """
    if not lag_months:
        return panel

    panel = panel.sort_values(["cod_mun","date"]).copy()
    for lag in lag_months:
        col_name = f"{target_col}_lag{lag}"
        panel[col_name] = (panel.groupby("cod_mun")[target_col]
                           .shift(lag))
        print(f"  Lag {lag} meses: '{col_name}' "
              f"(NaN: {panel[col_name].isna().sum()})")
    return panel


def apply_log_transform(panel, target_col, epsilon=1e-7):
    """
    log(taxa + ε) como alvo — reduz o peso de outliers extremos (picos COVID)
    e torna a distribuição mais simétrica, favorecendo regressão linear e GBM.
    Guarda a escala original em uma coluna separada para back-transformation.
    """
    log_col = f"log_{target_col}"
    panel[log_col] = np.log(panel[target_col] + epsilon)
    print(f"  Log-transform: '{log_col}' = log({target_col} + {epsilon})")
    print(f"    Original: min={panel[target_col].min():.5f}  "
          f"max={panel[target_col].max():.5f}  "
          f"zeros={( panel[target_col]==0).sum()}")
    print(f"    Log:      min={panel[log_col].min():.2f}  "
          f"max={panel[log_col].max():.2f}")
    return panel


def build_panel():
    variables_cfg, periods_cfg = load_config()
    full_dates  = build_full_date_range(periods_cfg)
    mun_master  = build_municipio_master(RAW_DIR)
    cod_muns    = mun_master["cod_mun"].tolist()

    panel_pieces, coverage_rows = [], []

    all_specs = {}
    for name, spec in variables_cfg["target"].items():
        all_specs[name] = (spec, False)
    for name, spec in variables_cfg["variables"].items():
        all_specs[name] = (spec, spec.get("interpolate", False))

    for name, (spec, interp_flag) in all_specs.items():
        long_df  = load_variable(RAW_DIR, spec)
        native_min = long_df["date"].min()
        native_max = long_df["date"].max()
        n_mun_native = long_df["cod_mun"].nunique()
        grid = series_to_monthly_grid(long_df, full_dates, cod_muns,
                                      interpolate=interp_flag)
        grid["variable"] = name
        panel_pieces.append(grid)

        pct_miss = np.nan
        if pd.notna(native_min) and pd.notna(native_max):
            w = grid[(grid["date"] >= native_min) & (grid["date"] <= native_max)]
            pct_miss = 100 * w["value"].isna().mean()

        coverage_rows.append({
            "variable": name,
            "tipo": "primária",
            "categoria": spec.get("category"),
            "fonte_arquivo": spec["file"],
            "frequencia_original": "anual→interpolada" if interp_flag else "nativa",
            "data_min_nativa": native_min,
            "data_max_nativa": native_max,
            "n_municipios_com_dado": n_mun_native,
            "pct_faltante_na_janela_nativa": round(pct_miss,1) if pd.notna(pct_miss) else np.nan,
            "entra_no_modelo": name not in variables_cfg.get("model_features_exclude",[]),
            "descricao": spec.get("descricao"),
        })

    long_all = pd.concat(panel_pieces, ignore_index=True)

    for target_name in variables_cfg["target"]:
        obs = long_all[(long_all.variable==target_name) & long_all.value.notna()]
        if obs.empty: continue
        tmin, tmax = obs["date"].min(), obs["date"].max()
        mask = ((long_all.variable==target_name) & long_all.value.isna()
                & (long_all.date>=tmin) & (long_all.date<=tmax))
        long_all.loc[mask, "value"] = 0.0

    panel_wide = long_all.pivot_table(
        index=["cod_mun","date"], columns="variable", values="value")
    panel_wide = panel_wide.reset_index().merge(mun_master, on="cod_mun", how="left")

    # ── Variáveis derivadas (taxas, binárias, temporais) ─────────────────────
    derived_cfg = variables_cfg.get("derived_variables", {})
    if derived_cfg:
        print(f"  Calculando {len(derived_cfg)} variáveis derivadas...")
        panel_wide = compute_derived_variables(panel_wide, derived_cfg)

    # ── Features de engenharia temporal ──────────────────────────────────────
    target_col = variables_cfg.get("model_target","srag_taxa_casos")
    eng_cfg    = periods_cfg.get("temporal_engineering", {})

    # 1. Lags do alvo
    lag_months = eng_cfg.get("lag_months", [])
    if lag_months and target_col in panel_wide.columns:
        print(f"\n  Adicionando lags: {lag_months} meses...")
        panel_wide = add_lag_features(panel_wide, target_col, lag_months, periods_cfg)

    # 2. Sazonalidade cíclica
    if eng_cfg.get("add_seasonality", False):
        print("  Adicionando sazonalidade cíclica (sin/cos do mês)...")
        panel_wide["date_dt"]   = pd.to_datetime(panel_wide["date"])
        month                   = panel_wide["date_dt"].dt.month
        panel_wide["month_sin"] = np.sin(2 * np.pi * month / 12)
        panel_wide["month_cos"] = np.cos(2 * np.pi * month / 12)
        panel_wide.drop(columns=["date_dt"], inplace=True)

    # 3. Fase pandêmica COVID
    if eng_cfg.get("add_covid_phase", False):
        print("  Adicionando indicador de fase pandêmica...")
        year = pd.to_datetime(panel_wide["date"]).dt.year
        panel_wide["fase_covid"] = np.select(
            [year.isin([2020,2021,2022]),
             year.isin([2023,2024,2025])],
            [1, 2],
            default=0  # pré-COVID / fora da janela
        ).astype(float)

    # 4. Transformação log do alvo
    if eng_cfg.get("log_transform_target", False) and target_col in panel_wide.columns:
        print("  Aplicando log-transform ao alvo...")
        panel_wide = apply_log_transform(panel_wide, target_col)

    # ── Atualizar coverage com variáveis derivadas/engenharia ─────────────────
    new_cols = [c for c in panel_wide.columns
                if c not in [r["variable"] for r in coverage_rows]
                and c not in ("cod_mun","date","municipio")]
    for col in new_cols:
        coverage_rows.append({
            "variable": col, "tipo": "engenharia",
            "categoria": "engenharia_temporal",
            "entra_no_modelo": True,
        })

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["tipo","categoria","variable"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel_wide.to_csv(PROCESSED / "panel_monthly.csv", index=False)
    coverage_df.to_csv(PROCESSED / "coverage_report.csv", index=False)
    long_all.to_csv(PROCESSED / "panel_long.csv", index=False)

    return panel_wide, coverage_df, variables_cfg, periods_cfg


if __name__ == "__main__":
    panel, coverage, _, _ = build_panel()
    print(f"\nPainel: {panel.shape}")
    new_cols = [c for c in panel.columns
                if any(x in c for x in ["lag","month_sin","month_cos","covid","log_"])]
    if new_cols:
        print(f"Novas features temporais: {new_cols}")
        print(panel[new_cols].describe().T[["mean","std","min","max"]])
