"""
seasonal_grouping.py
---------------------
Agrupa meses em fases de umidade (ou outra variável meteorológica),
substituindo a dimensão mensal por fases sazonais.

ALGORITMO:
  1. Calcula a média da variável de umidade por MÊS DO ANO (jan..dez),
     agregando municípios e anos.
  2. Divide os 12 meses em N fases usando quartis da série mensal anual.
  3. Cria mapeamento {mês_calendário → fase} e rotula cada linha do painel.
  4. Agrega o painel por (cod_mun, ano, fase), retornando um painel com
     N×anos observações por município em vez de 12×anos.

RESULTADO: painel com mesma estrutura do mensal, mas com coluna 'fase_umidade'
no lugar de 'month'. Pode ser usado diretamente pelos modelos.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_humidity_phases(panel: pd.DataFrame, humidity_col: str,
                            n_phases: int = 4,
                            phase_labels: list[str] | None = None) -> dict[int, int]:
    """
    Calcula a umidade média por mês do ano e atribui cada mês a uma fase.

    Retorna: dict {1..12 → 0..n_phases-1}
    """
    if humidity_col not in panel.columns:
        raise ValueError(
            f"Coluna '{humidity_col}' não encontrada no painel. "
            f"Rode fetch_meteorologico.py primeiro."
        )

    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["date"]).dt.month

    # Média da umidade por mês do calendário (agrega municípios e anos)
    monthly_humidity = (panel.groupby("month")[humidity_col]
                        .mean().sort_index())

    if phase_labels is None:
        phase_labels = [f"Fase_{i+1}" for i in range(n_phases)]

    # Atribuir fase por quartil
    quantiles = np.quantile(monthly_humidity.values,
                            np.linspace(0, 1, n_phases + 1))
    month_to_phase = {}
    for month, hum in monthly_humidity.items():
        for i in range(n_phases):
            lo, hi = quantiles[i], quantiles[i + 1]
            if hum <= hi or i == n_phases - 1:
                month_to_phase[month] = i
                break

    # Log do mapeamento
    phase_months = {i: [] for i in range(n_phases)}
    month_names = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    for month, phase in month_to_phase.items():
        phase_months[phase].append(month_names[month - 1])

    print(f"\n  Mapeamento meses → fases de umidade ({humidity_col}):")
    for i, label in enumerate(phase_labels):
        meses = ", ".join(phase_months[i])
        hums  = [monthly_humidity[m] for m in month_to_phase
                 if month_to_phase[m] == i]
        print(f"    Fase {i} [{label}]: {meses}  "
              f"(umidade média: {np.mean(hums):.1f}%)")

    return month_to_phase, phase_labels


def build_seasonal_panel(panel: pd.DataFrame,
                         target_col: str,
                         feature_cols: list[str],
                         exp_cfg: dict) -> pd.DataFrame:
    """
    Transforma o painel mensal em painel sazonal baseado em fases de umidade.

    Retorna DataFrame com (cod_mun, municipio, ano, fase, fase_label, target, features).
    """
    humidity_col  = exp_cfg.get("humidity_variable", "meteo_umidade")
    n_phases      = exp_cfg.get("n_phases", 4)
    phase_labels  = exp_cfg.get("phase_labels")
    agg_func      = exp_cfg.get("aggregation", "mean")

    panel = panel.copy()
    panel["date"]  = pd.to_datetime(panel["date"])
    panel["month"] = panel["date"].dt.month
    panel["year"]  = panel["date"].dt.year

    month_to_phase, phase_labels = compute_humidity_phases(
        panel, humidity_col, n_phases, phase_labels
    )

    panel["fase"]       = panel["month"].map(month_to_phase)
    panel["fase_label"] = panel["fase"].map(dict(enumerate(phase_labels)))

    # Agregar por município × ano × fase
    cols_to_agg = [c for c in [target_col] + feature_cols if c in panel.columns]
    agg_dict    = {c: agg_func for c in cols_to_agg}

    seasonal = (panel.groupby(["cod_mun", "municipio", "year", "fase", "fase_label"],
                               as_index=False)
                .agg(agg_dict))

    # Criar coluna 'date' sintética: primeiro dia do ano (para compatibilidade)
    seasonal["date"] = pd.to_datetime(seasonal["year"].astype(str) + "-01-01")

    print(f"\n  Painel sazonal: {len(seasonal)} linhas "
          f"({seasonal['cod_mun'].nunique()} municípios × "
          f"{seasonal['year'].nunique()} anos × "
          f"{n_phases} fases)")

    return seasonal, month_to_phase, phase_labels


def plot_humidity_profile(panel: pd.DataFrame, humidity_col: str,
                          month_to_phase: dict, phase_labels: list[str],
                          out_path: Path):
    """Plota o perfil médio de umidade por mês com cores por fase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if humidity_col not in panel.columns:
        return

    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["date"]).dt.month
    monthly = panel.groupby("month")[humidity_col].mean()

    month_names = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    palette = ["#F39C12","#E67E22","#2980B9","#1A5276"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for month, hum in monthly.items():
        phase = month_to_phase[month]
        color = palette[phase % len(palette)]
        ax.bar(month, hum, color=color, alpha=0.85, edgecolor="white")
        ax.text(month, hum + 0.5, month_names[month-1], ha="center",
                fontsize=8, color="gray")

    # Legenda de fases
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[i], label=phase_labels[i])
               for i in range(len(phase_labels))]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              title="Fases de umidade")

    ax.set_xlabel("Mês do ano")
    ax.set_ylabel(f"{humidity_col} (%)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Perfil de umidade média por mês — definição das fases sazonais\n"
                 "(média sobre todos os municípios e anos disponíveis)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.name}")
