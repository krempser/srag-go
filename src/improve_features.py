"""
improve_features.py
--------------------
Adiciona ao painel as features que faltam para empurrar R² de 0.26 → 0.5+:

  1. baseline_srag_municipio
     Taxa histórica média de SRAG por município — calculada SOMENTE sobre os
     dados de TREINO para evitar data leakage. É a feature mais preditiva
     para dados de painel epidemiológico: "municípios que historicamente
     tiveram mais casos tendem a continuar tendo mais casos". Tipicamente
     explica 20-35% da variância por si só.

  2. year_trend
     Ano como variável numérica normalizada — captura a tendência de aumento
     de notificação, evolução do sistema de vigilância, e o "novo normal"
     pós-COVID. Simples e eficaz.

  3. lag_within_seasonal
     No modo sazonal, o lag temporal equivalente é "a taxa nessa mesma fase
     de umidade no ano anterior". Captura a persistência inter-anual do risco.

COMO INCORPORAR:
  Este script adiciona essas features AO PAINEL JÁ PROCESSADO (panel_monthly.csv).
  Rode após build_panel.py:

    python src/build_panel.py
    python src/improve_features.py

  Depois rode os experimentos normalmente.

IMPACTO ESPERADO:
  - baseline_srag_municipio:  +0.15 a +0.25 R²
  - year_trend:               +0.03 a +0.08 R²
  - lag_within_seasonal:      +0.05 a +0.10 R² (no modo sazonal)
  - Total estimado:           R² 0.26 → 0.45-0.60
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def add_municipality_baseline(panel: pd.DataFrame,
                              target_col: str,
                              test_holdout_months: int = 6,
                              seasonal_mode: bool = False) -> pd.DataFrame:
    """
    Adiciona 'baseline_srag_municipio': taxa histórica MÉDIA de SRAG por
    município, calculada SOMENTE sobre dados de treino.

    COMO É CALCULADO (sem leakage):
      Modo mensal: exclui os últimos test_holdout_months meses do painel.
        → baseline = média mensal de 2019-01 a 2024-06 (se holdout=6)
        → testado em 2024-07 a 2024-12 (que nunca entrou no cálculo)

      Modo sazonal: exclui o último ano completo do painel.
        → baseline = média anual de 2019 a 2023
        → testado em 2024 (que nunca entrou no cálculo)

    POR QUE NÃO É LEAKAGE:
      O baseline é uma propriedade do MUNICÍPIO, não do período testado.
      É análogo ao uso de 'Standardized Incidence Ratio' em epidemiologia:
      a taxa esperada de referência é calculada sobre período histórico
      e usada para comparar com o período atual.
      O modelo aprende: "municípios com baseline alto tendem a ter alta
      incidência em qualquer fase climática" — isso é a pergunta de
      estratificação de risco, não previsão do futuro.
    """
    panel = panel.sort_values(["cod_mun", "date"]).copy()
    obs = panel[panel[target_col].notna()].copy()

    if seasonal_mode and "year" in obs.columns:
        # Modo sazonal: excluir o último ano inteiro
        last_year = obs["year"].max()
        obs_train = obs[obs["year"] < last_year]
        mode_desc = f"anos < {last_year}"
    else:
        # Modo mensal: excluir os últimos N meses
        all_dates   = sorted(obs["date"].unique())
        train_dates = set(all_dates[:-test_holdout_months])
        obs_train   = obs[obs["date"].isin(train_dates)]
        mode_desc   = f"excluindo últimos {test_holdout_months} meses"

    baseline = (obs_train
                .groupby("cod_mun")[target_col]
                .mean()
                .rename("baseline_srag_municipio"))

    global_median = baseline.median()
    # Remover coluna anterior se já existir (evita _x/_y em re-execuções)
    if "baseline_srag_municipio" in panel.columns:
        panel = panel.drop(columns=["baseline_srag_municipio"])
    panel = panel.merge(baseline.reset_index(), on="cod_mun", how="left")
    panel["baseline_srag_municipio"] = panel["baseline_srag_municipio"].fillna(global_median)

    n_municip = baseline.notna().sum()
    print(f"  baseline_srag_municipio ({mode_desc}): {n_municip} municípios")
    print(f"    min={baseline.min():.5f}  median={baseline.median():.5f}  "
          f"max={baseline.max():.5f}")
    return panel


def add_year_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona 'year_trend': ano normalizado [0, 1] sobre o intervalo observado.
    Captura tendência de notificação crescente, efeito COVID, transição pós-pandemia.
    """
    panel = panel.copy()
    year = pd.to_datetime(panel["date"]).dt.year.astype(float)
    y_min, y_max = year.min(), year.max()
    panel["year_trend"] = (year - y_min) / (y_max - y_min + 1e-9)
    print(f"  year_trend: {int(y_min)}={panel['year_trend'].min():.2f}  "
          f"→  {int(y_max)}={panel['year_trend'].max():.2f}")
    return panel


def add_seasonal_lag(panel: pd.DataFrame,
                     target_col: str,
                     humidity_variable: str = "meteo_umidade",
                     n_phases: int = 4) -> pd.DataFrame:
    """
    Para o modo sazonal: adiciona 'srag_lag_same_phase_prev_year'.
    = taxa de SRAG do mesmo município, na mesma fase de umidade, no ano anterior.

    Esta é a versão sazonal do lag temporal — "quanto SRAG houve aqui, nessa
    estação climática, no ano passado?" É análoga ao lag-12 no modelo mensal
    mas respeitando a estrutura de fases de umidade.

    Requer que o painel já tenha a coluna 'fase' gerada pelo seasonal_grouping.
    Se 'fase' não existir, o lag é calculado por mês do calendário como fallback.
    """
    panel = panel.copy()
    panel["_year"] = pd.to_datetime(panel["date"]).dt.year

    if "fase" in panel.columns:
        group_cols = ["cod_mun", "fase"]
        print("  srag_lag_same_phase_prev_year: usando fases de umidade")
    else:
        panel["_month"] = pd.to_datetime(panel["date"]).dt.month
        group_cols = ["cod_mun", "_month"]
        print("  srag_lag_same_phase_prev_year: fallback para mês calendário")

    # Para cada município × fase, o lag é o valor do ano anterior
    pivot = panel.groupby(group_cols + ["_year"])[target_col].mean().reset_index()
    pivot["_year_next"] = pivot["_year"] + 1
    lag_map = pivot.rename(columns={target_col: "srag_lag_same_phase_prev_year",
                                    "_year_next": "_year_lag"})

    # Juntar no painel original
    merge_cols = group_cols + ["_year"]
    if "srag_lag_same_phase_prev_year" in panel.columns:
        panel = panel.drop(columns=["srag_lag_same_phase_prev_year"])
    panel = panel.merge(
        lag_map[group_cols + ["_year_lag", "srag_lag_same_phase_prev_year"]]
        .rename(columns={"_year_lag": "_year"}),
        on=group_cols + ["_year"],
        how="left"
    )

    n_valid = panel["srag_lag_same_phase_prev_year"].notna().sum()
    print(f"    {n_valid} linhas com lag válido (primeiros anos ficam NaN → imputados)")

    panel.drop(columns=["_year"] + (["_month"] if "_month" in panel.columns else []),
               inplace=True)
    return panel


def run():
    import yaml
    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)
    with open(ROOT / "config" / "variables.yaml", encoding="utf-8") as f:
        vcfg = yaml.safe_load(f)

    target_col = vcfg.get("model_target", "srag_taxa_casos")
    n_hold     = pcfg.get("test_holdout_months", 6)
    hum_var    = pcfg.get("seasonal", {}).get("humidity_variable", "meteo_umidade")

    print("\n=== improve_features.py ===")
    print(f"  Carregando panel_monthly.csv...")
    panel = pd.read_csv(PROCESSED / "panel_monthly.csv", parse_dates=["date"])
    print(f"  Shape inicial: {panel.shape}")

    print("\n[1/3] Baseline histórico por município...")
    if target_col in panel.columns:
        panel = add_municipality_baseline(panel, target_col, n_hold)
    else:
        print(f"  AVISO: target '{target_col}' não encontrado — pulando baseline")

    print("\n[2/3] Tendência temporal (year_trend)...")
    panel = add_year_trend(panel)

    print("\n[3/3] Lag sazonal (mesmo período, ano anterior)...")
    if target_col in panel.columns:
        panel = add_seasonal_lag(panel, target_col, hum_var)
    else:
        print(f"  AVISO: target '{target_col}' não encontrado — pulando lag sazonal")

    print(f"\n  Shape final: {panel.shape}")
    new_cols = ["baseline_srag_municipio", "year_trend",
                "srag_lag_same_phase_prev_year"]
    present = [c for c in new_cols if c in panel.columns]
    print(f"  Novas colunas: {present}")

    panel.to_csv(PROCESSED / "panel_monthly.csv", index=False)
    print(f"\n✓ panel_monthly.csv atualizado com {len(present)} novas features")
    print(f"\nPróximo passo: atualize feature_sets.py para incluir as novas colunas")
    print(f"e rode: python run_experiments.py --fast")


if __name__ == "__main__":
    run()
