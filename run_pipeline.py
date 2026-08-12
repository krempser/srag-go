#!/usr/bin/env python3
"""
run_pipeline.py
Modos (config/periods.yaml):
  atemporal: true  → agrega município×mês para 1 linha/município (mediana)
                     e faz regressão cross-sectional de risco estrutural
  model_mode: classification → classifica faixas de intensidade
  model_mode: regression     → prediz taxa contínua com painel temporal
"""
import sys, time, yaml
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import build_panel, eval_variables, evaluate_shape, plot_all


def main():
    t0 = time.time()
    with open(ROOT / "config" / "periods.yaml", encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)

    atemporal = pcfg.get("atemporal", False)
    mode      = pcfg.get("model_mode", "regression")

    print(f"\n{'='*55}")
    if atemporal:
        agg = pcfg.get("atemporal_aggregation", "median")
        print(f"  Pipeline SRAG  |  modo: ATEMPORAL ({agg})")
    else:
        print(f"  Pipeline SRAG  |  modo: {mode.upper()}")
    print(f"{'='*55}")

    print("\n[1] Construindo painel mensal...")
    panel, coverage, _, _ = build_panel.build_panel()
    print(f"    painel: {panel.shape[0]} linhas × {panel.shape[1]} colunas")

    print("\n[2] Avaliação prévia das variáveis...")
    summary, _ = eval_variables.run_variable_evaluation()
    print(f"    {len(summary)} variáveis avaliadas")

    if atemporal:
        import models_atemporal
        print("\n[3] Modelo atemporal (risco estrutural por município)...")
        results_df, cs = models_atemporal.run_atemporal()
        print(f"\nConcluído em {time.time()-t0:.1f}s.")
        print("Saídas:")
        print("  outputs/tables/atemporal_municipios_ranqueados.csv")
        print("  outputs/figures/atemporal_ranking_municipios.png")
        print("  outputs/figures/atemporal_mapa_risco.png")
        print("  outputs/figures/atemporal_scatter.png")
        print("  outputs/figures/atemporal_importancias.png")

    elif mode == "classification":
        import models_classification
        print("\n[3] Classificação de faixas de SRAG...")
        models_classification.run_classification()
        preds_csv = ROOT / "outputs" / "tables" / "predicoes_teste.csv"
        if preds_csv.exists():
            print("\n[4] Avaliação por município (shape)...")
            evaluate_shape.run_shape_evaluation()
        print(f"\nConcluído em {time.time()-t0:.1f}s.")

    else:  # regression temporal
        import models
        print("\n[3] Regressão temporal (busca de hiperparâmetros)...")
        results_df, preds_df, _ = models.run_models()
        print(results_df.to_string(index=False))
        print("\n[4] Avaliação por município (shape)...")
        evaluate_shape.run_shape_evaluation()
        print("\n[5] Gerando gráficos diagnósticos...")
        plot_all.main()
        print(f"\nConcluído em {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
