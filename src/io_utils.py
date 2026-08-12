"""
io_utils.py
-----------
Cada função `parse_<formato>` lê UM arquivo bruto e devolve um DataFrame
"longo" padronizado com as colunas:

    cod_mun   (str, código de 6 dígitos, padrão DATASUS/IBGE sem dígito verificador)
    date      (Timestamp, sempre o dia 1 do mês)
    value     (float)

O nome da variável e metadados (categoria, descrição, etc.) são anexados
depois, em build_panel.py, a partir do config/variables.yaml.

Qualquer novo formato de arquivo que apareça no futuro só precisa de uma
nova função aqui + uma referência a ela em PARSERS no final do arquivo.
"""
import re
import io
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

MONTH_MAP_PT = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

YEAR_COL_RE = re.compile(r"^(19|20)\d{2}$")


def _ibge7_to_datasus6(code) -> str:
    """Converte código IBGE de 7 dígitos (com dígito verificador) para o
    código de 6 dígitos usado pelo DATASUS (remove o dígito verificador,
    que é sempre o último). Já recebe 6 dígitos -> mantém como está."""
    s = str(code).strip().split(".")[0]
    if len(s) == 7:
        return s[:6]
    return s.zfill(6)


def _norm_cod_mun(code) -> str:
    s = str(code).strip().split(".")[0]
    if len(s) == 7:
        s = s[:6]
    return s.zfill(6)


def _parse_numeric(series: pd.Series, decimal: str = ".", strip_pct: bool = False,
                    missing_token: str | None = None) -> pd.Series:
    s = series.astype(str).str.strip()
    if missing_token is not None:
        s = s.replace(missing_token, np.nan)
    if strip_pct:
        s = s.str.replace("%", "", regex=False)
    s = s.str.strip()
    if decimal == ",":
        # remove separador de milhar '.' e troca decimal ',' por '.'
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_pt_month_colname(colname: str):
    """Aceita 'jul/09', 'JUL/2009', '2010/Jan', '2010/jan' etc."""
    parts = re.split(r"[/_-]", str(colname).strip())
    if len(parts) != 2:
        return None
    a, b = parts
    if a.isdigit() and not b.isdigit():
        year_tok, mon_tok = a, b
    elif b.isdigit() and not a.isdigit():
        mon_tok, year_tok = a, b
    else:
        return None
    mon_tok = mon_tok.lower()[:3]
    if mon_tok not in MONTH_MAP_PT:
        return None
    month = MONTH_MAP_PT[mon_tok]
    year = int(year_tok)
    if year < 100:
        year += 2000
    try:
        return pd.Timestamp(year=year, month=month, day=1)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# PARSERS
# ----------------------------------------------------------------------------

def parse_wide_year(path, spec) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=spec.get("encoding", "utf-8"), sep=spec.get("sep", ","))
    cod_col = spec["cols"]["cod_mun"]
    year_cols = [c for c in df.columns if YEAR_COL_RE.match(str(c).strip())]
    long = df.melt(id_vars=[cod_col], value_vars=year_cols, var_name="ano", value_name="value")
    long["cod_mun"] = long[cod_col].map(_norm_cod_mun)
    long["value"] = _parse_numeric(long["value"], decimal=spec.get("decimal", "."),
                                    strip_pct=spec.get("strip_pct", False))
    long["date"] = pd.to_datetime(long["ano"].astype(int).astype(str) + "-01-01")
    return long[["cod_mun", "date", "value"]].dropna(subset=["cod_mun"])


def parse_wide_month(path, spec) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=spec.get("encoding", "utf-8"), sep=spec.get("sep", ","))
    cod_col = spec["cols"]["cod_mun"]
    month_cols = []
    col_to_date = {}
    for c in df.columns:
        d = _parse_pt_month_colname(c)
        if d is not None:
            month_cols.append(c)
            col_to_date[c] = d
    long = df.melt(id_vars=[cod_col], value_vars=month_cols, var_name="col", value_name="value")
    long["cod_mun"] = long[cod_col].map(_norm_cod_mun)
    long["date"] = long["col"].map(col_to_date)
    long["value"] = _parse_numeric(long["value"], decimal=spec.get("decimal", "."),
                                    strip_pct=spec.get("strip_pct", False))
    return long[["cod_mun", "date", "value"]].dropna(subset=["cod_mun", "date"])


def parse_wide_month_matrix_footer(path, spec) -> pd.DataFrame:
    """Para arquivos como 'leitos de uti...csv': primeiras 2 colunas são
    cod_mun e nome; há linhas de rodapé/nota no final que devem ser
    descartadas (identificadas por cod_mun não numérico)."""
    df = pd.read_csv(path, encoding=spec.get("encoding", "utf-8"), sep=spec.get("sep", ";"))
    cod_idx = spec.get("cod_mun_col", 0)
    cod_col = df.columns[cod_idx]
    df = df[pd.to_numeric(df[cod_col], errors="coerce").notna()].copy()
    month_cols, col_to_date = [], {}
    for c in df.columns:
        d = _parse_pt_month_colname(c)
        if d is not None:
            month_cols.append(c)
            col_to_date[c] = d
    long = df.melt(id_vars=[cod_col], value_vars=month_cols, var_name="col", value_name="value")
    long["cod_mun"] = long[cod_col].map(_norm_cod_mun)
    long["date"] = long["col"].map(col_to_date)
    long["value"] = _parse_numeric(long["value"], missing_token=spec.get("missing_token"))
    return long[["cod_mun", "date", "value"]].dropna(subset=["cod_mun", "date"])


def parse_long_month(path, spec) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=spec.get("encoding", "utf-8"), sep=spec.get("sep", ","))
    c = spec["cols"]
    out = pd.DataFrame()
    out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    out["date"] = pd.to_datetime(df[c["data"]], format=spec.get("date_format"),
                                 errors="coerce")   # datas inválidas (ex: mês 13) → NaT, descartadas abaixo
    out["value"] = _parse_numeric(df[c["valor"]], decimal=spec.get("decimal", "."),
                                   strip_pct=spec.get("strip_pct", False))
    out = out.dropna(subset=["cod_mun", "date"])
    out["date"] = out["date"].values.astype("datetime64[M]")
    return out


def parse_long_weekly(path, spec) -> pd.DataFrame:
    """SRAG: semana epidemiológica + ano -> aproximação de mês.
    Aproximação: data = 1o de janeiro do ano + (semana-1)*7 dias; o mês
    dessa data é usado como mês de referência. É uma aproximação919 (semanas
    não se alinham perfeitamente a meses), documentada no relatório."""
    df = pd.read_csv(path, encoding=spec.get("encoding", "utf-8"), sep=spec.get("sep", ";"))
    c = spec["cols"]
    out = pd.DataFrame()
    out["cod_mun"] = df[c["cod_mun"]].map(lambda x: _norm_cod_mun(x) if pd.notna(x) else np.nan)
    ano = pd.to_numeric(df[c["ano"]], errors="coerce")
    semana = pd.to_numeric(df[c["semana"]], errors="coerce").clip(upper=52)
    approx_date = pd.to_datetime(ano.astype("Int64").astype(str) + "-01-01", errors="coerce") + \
        pd.to_timedelta((semana - 1) * 7, unit="D")
    out["date"] = approx_date.values.astype("datetime64[M]")
    out["value"] = pd.to_numeric(df[c["valor"]], errors="coerce")
    out = out.dropna(subset=["cod_mun", "date"])
    agg = spec.get("agg", "sum")
    out = out.groupby(["cod_mun", "date"], as_index=False)["value"].agg(agg)
    return out


def parse_xlsx_long(path, spec) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=spec.get("sheet", 0))
    c = spec["cols"]
    out = pd.DataFrame()
    out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    out["date"] = pd.to_datetime(df[c["ano"]].astype(int).astype(str) + "-01-01")
    out["value"] = pd.to_numeric(df[c["valor"]], errors="coerce")
    return out.dropna(subset=["cod_mun", "date"])


def parse_xlsx_long_ym(path, spec) -> pd.DataFrame:
    df = pd.read_excel(path)
    c = spec["cols"]
    out = pd.DataFrame()
    if spec.get("cod_mun_is_7digit"):
        out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    else:
        out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    ano = df[c["ano"]].astype(int).astype(str)
    mes = df[c["mes"]].astype(int).astype(str).str.zfill(2)
    out["date"] = pd.to_datetime(ano + "-" + mes + "-01")
    out["value"] = pd.to_numeric(df[c["valor"]], errors="coerce")
    return out.dropna(subset=["cod_mun", "date"]).groupby(["cod_mun", "date"], as_index=False)["value"].sum()


def parse_xlsx_long_date(path, spec) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=spec.get("sheet", 0))
    c = spec["cols"]
    out = pd.DataFrame()
    out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    out["date"] = pd.to_datetime(df[c["data"]], format=spec.get("date_format"))
    out["date"] = out["date"].values.astype("datetime64[M]")
    out["value"] = pd.to_numeric(df[c["valor"]], errors="coerce")
    return out.dropna(subset=["cod_mun", "date"])


def parse_xlsx_csv_embedded(path, spec) -> pd.DataFrame:
    """Para .xlsx cuja célula/coluna única contém um CSV completo como texto
    (caso do arquivo de CadÚnico fornecido)."""
    raw = pd.read_excel(path, header=None)
    text = "\n".join(raw[0].iloc[1:].astype(str).tolist())
    df = pd.read_csv(io.StringIO(text))
    c = spec["cols"]
    out = pd.DataFrame()
    out["cod_mun"] = df[c["cod_mun"]].map(_norm_cod_mun)
    out["date"] = pd.to_datetime(df[c["data"]], format=spec.get("date_format"))
    out["date"] = out["date"].values.astype("datetime64[M]")
    out["value"] = pd.to_numeric(df[c["valor"]], errors="coerce")
    return out.dropna(subset=["cod_mun", "date"])


PARSERS = {
    "wide_year": parse_wide_year,
    "wide_month": parse_wide_month,
    "wide_month_matrix_footer": parse_wide_month_matrix_footer,
    "long_month": parse_long_month,
    "long_weekly": parse_long_weekly,
    "xlsx_long": parse_xlsx_long,
    "xlsx_long_date": parse_xlsx_long_date,
    "xlsx_long_ym": parse_xlsx_long_ym,
    "xlsx_csv_embedded": parse_xlsx_csv_embedded,
}


def load_variable(raw_dir, spec) -> pd.DataFrame:
    parser_name = spec["parser"]
    if parser_name not in PARSERS:
        raise ValueError(f"Parser desconhecido: {parser_name}. "
                          f"Disponíveis: {list(PARSERS)}")
    raw_dir = Path(raw_dir)
    file_path = spec["file"]
    # Se o path começa com "processed/" resolve em relação a data/, não data/raw/
    if file_path.startswith("processed/"):
        path = raw_dir.parent / file_path
    else:
        path = raw_dir / file_path
    return PARSERS[parser_name](str(path), spec)
