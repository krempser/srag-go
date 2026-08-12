"""
convert_mapbiomas.py
--------------------
Converte a planilha MapBiomas (aba COVERAGE_10.1) para dois CSVs no formato
do pipeline (cod_mun, ano_mes, valor):

  data/processed/mapbiomas_classes_abs.csv  — hectares por classe level 1
  data/processed/mapbiomas_classes_pct.csv  — % do território total

USO:
  python src/convert_mapbiomas.py --file data/raw/mapbiomas.xlsx [--uf GO]

O script aceita qualquer UF ou "BR" para todo o Brasil — é a única
adaptação necessária para escalar para o Brasil inteiro.

ATENÇÃO — junção de nomes:
  O MapBiomas não fornece código IBGE na aba COVERAGE_10.1. A junção é
  feita pelo NOME do município normalizado (sem acento, maiúsculo). Isso
  pode causar falhas em municípios com grafia ambígua (ex: "Barão de
  Goiás" vs "BARAO DE GOIAS"). O script imprime todos os casos não
  associados para revisão manual.
"""
import argparse
import unicodedata
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"


def normalize_name(s: str) -> str:
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_municipio_lookup(raw_dir: Path) -> pd.DataFrame:
    """Lê pop mensal.csv como lista mestre de municípios GO + normaliza nomes."""
    df = pd.read_csv(raw_dir / "pop mensal.csv", encoding="latin1", sep=";")
    df["cod_mun"] = df["cod_mun"].astype(str).str.strip().str.zfill(6)
    df = df[["cod_mun", "Município"]].drop_duplicates()
    df["nome_norm"] = df["Município"].map(normalize_name)
    return df.set_index("nome_norm")


def load_mapbiomas(filepath: Path, uf_filter: str | None = "GO") -> pd.DataFrame:
    print(f"Lendo {filepath}  (aba COVERAGE_10.1)…")
    df = pd.read_excel(filepath, sheet_name="COVERAGE_10.1")
    print(f"  Shape bruto: {df.shape}  |  Colunas: {df.columns.tolist()[:10]}…")

    # Filtrar UF
    if uf_filter and uf_filter.upper() != "BR":
        col_uf = "state_acronym" if "state_acronym" in df.columns else "state"
        df = df[df[col_uf].str.upper() == uf_filter.upper()].copy()
        print(f"  Após filtro UF={uf_filter}: {len(df)} linhas")

    # Manter apenas class_level_1 e agregar sub-classes
    if "class_level_1" not in df.columns:
        raise ValueError("Coluna 'class_level_1' não encontrada. Verificar aba COVERAGE_10.1")

    # Identificar colunas de ano (numéricas >= 1985)
    year_cols = [c for c in df.columns if str(c).isdigit() and int(str(c)) >= 1985]
    if not year_cols:
        raise ValueError("Nenhuma coluna de ano encontrada na planilha.")
    print(f"  Anos encontrados: {min(year_cols)}–{max(year_cols)} ({len(year_cols)} anos)")

    id_cols = ["municipality", "class_level_1"]
    agg_df = df.groupby(id_cols, as_index=False)[year_cols].sum()
    return agg_df, year_cols


def convert(filepath: Path, uf_filter: str = "GO"):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mun_lookup = build_municipio_lookup(RAW_DIR)
    agg_df, year_cols = load_mapbiomas(filepath, uf_filter)

    # Normalizar nomes e juntar cod_mun
    agg_df["nome_norm"] = agg_df["municipality"].map(normalize_name)
    matched = agg_df.join(mun_lookup[["cod_mun"]], on="nome_norm", how="left")
    miss = matched[matched["cod_mun"].isna()]["municipality"].unique()
    if len(miss):
        print(f"\n⚠  {len(miss)} municípios sem cod_mun correspondente (verificar grafia):")
        for m in sorted(miss):
            print(f"    {m}")
    print(f"\n  Municípios associados: {matched['cod_mun'].notna().sum()} / {len(matched)}")

    matched = matched.dropna(subset=["cod_mun"])

    # Formato longo: (cod_mun, class_level_1, ano, valor_ha)
    long = matched.melt(
        id_vars=["cod_mun", "class_level_1"],
        value_vars=year_cols,
        var_name="ano",
        value_name="valor_ha"
    )
    long["ano"] = long["ano"].astype(int)

    # MapBiomas é anual → gerar uma linha por mês (ano-01 a ano-12)
    # (a interpolação mensal real é feita pelo pipeline via interpolate: true no YAML)
    long["ano_mes"] = long["ano"].astype(str) + "-01"

    # ------ Arquivo 1: valores absolutos (hectares) ------
    abs_wide = long.pivot_table(
        index=["cod_mun", "ano_mes"],
        columns="class_level_1",
        values="valor_ha",
        aggfunc="sum"
    ).reset_index()
    abs_wide.columns.name = None
    # Renomear colunas numéricas de classe para "ha_<classe>"
    rename_abs = {c: f"ha_{c}" for c in abs_wide.columns if str(c).isdigit()}
    abs_wide = abs_wide.rename(columns=rename_abs)
    abs_wide.to_csv(PROCESSED_DIR / "mapbiomas_classes_abs.csv", index=False)
    print(f"\n✓ mapbiomas_classes_abs.csv  → {abs_wide.shape}")

    # ------ Arquivo 2: porcentagem sobre o território total ------
    ha_cols = [c for c in abs_wide.columns if str(c).startswith("ha_")]
    pct_wide = abs_wide.copy()
    total = pct_wide[ha_cols].sum(axis=1).replace(0, np.nan)
    for col in ha_cols:
        class_id = col.replace("ha_", "")
        pct_wide[f"pct_{class_id}"] = pct_wide[col] / total * 100
    pct_wide = pct_wide[["cod_mun", "ano_mes"] + [c for c in pct_wide.columns if c.startswith("pct_")]]
    pct_wide.to_csv(PROCESSED_DIR / "mapbiomas_classes_pct.csv", index=False)
    print(f"✓ mapbiomas_classes_pct.csv  → {pct_wide.shape}")

    # Resumo das classes encontradas
    classes = sorted(long["class_level_1"].unique())
    print(f"\nClasses nível 1 encontradas: {classes}")
    print("(Descomente os blocos correspondentes em config/variables.yaml para incluí-las no modelo)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Caminho para a planilha MapBiomas (.xlsx)")
    parser.add_argument("--uf", default="GO", help="Sigla do estado, ou BR para todo o Brasil")
    args = parser.parse_args()
    convert(Path(args.file), args.uf)
