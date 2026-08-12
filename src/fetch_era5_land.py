"""
fetch_era5_land.py
------------------
Baixa dados climáticos mensais históricos para todos os municípios do modelo
usando a API Open-Meteo (https://open-meteo.com/).

Por que Open-Meteo em vez da CDS API do Copernicus?
  - Gratuita, sem cadastro, sem chave de API
  - Usa ERA5-Land como fonte de dados (mesma base, dados idênticos)
  - Retorna JSON direto — só precisa de `requests` (padrão Python)
  - Evita download de arquivos NetCDF pesados (centenas de GB)

PRÉ-REQUISITO:
  pip install requests tqdm
  (se não tiver tqdm: pip install tqdm)

USO:
  python src/fetch_era5_land.py [--uf GO] [--years 2017-2025] [--workers 4]

SAÍDA:
  data/processed/era5_land_go.csv

  Colunas: cod_mun, municipio, ano_mes, temp_media_c, temp_min_c, temp_max_c,
           temp_amplitude_c, precip_mm, umidade_relativa_pct

ESTRATÉGIA:
  1) Busca centroides de todos os municípios GO via API do IBGE (JSON, sem auth)
  2) Para cada município, chama Open-Meteo Archive API com o centroide e baixa
     dados mensais de: temperature_2m_mean/min/max, precipitation_sum,
     relative_humidity_2m_mean  (todos via ERA5-Land)
  3) Agrega para o formato cod_mun × ano_mes
  4) Salva o CSV; descomente os blocos era5_* em config/variables.yaml para
     incluir no modelo.

PARA ESCALAR PARA O BRASIL:
  python src/fetch_era5_land.py --uf BR --workers 8
  (todos os municípios do modelo — pode levar 30-60 min dependendo da conexão)
"""

import sys
import time
import argparse
import gzip
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

# Bounding box para filtrar municípios por UF
UF_STATE_CODES = {
    "GO": "52", "SP": "35", "RJ": "33", "MG": "31", "BA": "29",
    "RS": "43", "PR": "41", "SC": "42", "CE": "23", "PE": "26",
    "PA": "15", "MA": "21", "AM": "13", "MT": "51", "MS": "50",
    "PI": "22", "AL": "27", "SE": "28", "PB": "25", "RN": "24",
    "TO": "17", "ES": "32", "AP": "16", "RO": "11", "RR": "14",
    "AC": "12", "DF": "53",
}

# Open-Meteo Archive API — ERA5-Land
OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=temperature_2m_mean,temperature_2m_min,temperature_2m_max,"
    "precipitation_sum,relative_humidity_2m_mean"
    "&timezone=America%2FSao_Paulo"
)

IBGE_MUN_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"


def calc_dv_ibge(cod6: str) -> str:
    """Calcula o dígito verificador do código IBGE (6→7 dígitos)."""
    pesos = [1, 2, 1, 2, 1, 2]
    s = sum(
        (int(d) * p) // 10 + (int(d) * p) % 10
        for d, p in zip(str(cod6).zfill(6), pesos)
    )
    dv = (10 - (s % 10)) % 10
    return str(cod6).zfill(6) + str(dv)


def fetch_json(url: str, retries: int = 3, timeout: int = 30) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                encoding = r.info().get("Content-Encoding", "")
                if encoding == "gzip" or (raw[:2] == b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def get_municipio_centroids(uf: str, pipeline_muns: pd.DataFrame) -> pd.DataFrame:
    """
    Busca centroides (lat/lon) dos municípios via API IBGE.
    Endpoint: /api/v1/localidades/estados/{uf}/municipios
    Retorna: DataFrame com cod_mun, municipio, lat, lon
    """
    # Verificar se já existe arquivo de centroides salvo (cache local)
    cache_path = ROOT / "data" / "external" / f"centroides_{uf.lower()}.csv"
    if cache_path.exists():
        print(f"  Centroides carregados do cache: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Buscando centroides via API IBGE para UF={uf}...")
    url = IBGE_MUN_URL.format(uf=uf)
    data = fetch_json(url)
    if data is None:
        print("  ERRO: não foi possível acessar a API IBGE. Verifique a conexão.")
        sys.exit(1)

    rows = []
    for item in data:
        cod7 = str(item["id"])
        cod6 = cod7[:6]
        # A API v1 não retorna lat/lon diretamente — usar endpoint de malhas
        rows.append({"cod_mun": cod6, "municipio_ibge": item["nome"], "cod7": cod7})

    mun_df = pd.DataFrame(rows)

    # Para lat/lon, usar endpoint de geocodificação dos municípios
    # GET /api/v3/malhas/municipios/{codmun}?formato=application/vnd.geo+json
    # é muito pesado. Alternativa: endpoint de lista com coordenadas não existe na v1.
    # Usamos o endpoint de localidades com projeção geométrica disponível na v3:
    # GET https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}?formato=application/vnd.geo+json&resolucao=1&qualidade=minima
    # → retorna polígonos simplificados; calculamos centroide do bounding box.
    #
    # Alternativa mais simples: API de municípios com coordenadas do centroide
    # via /api/v1/localidades/municipios (sem filtro de UF, com lat/lon implícito)
    # Infelizmente a API v1 do IBGE não inclui coordenadas.
    #
    # SOLUÇÃO ADOTADA: buscar via Nominatim/OpenStreetMap para cada município.
    # Rate limit: 1 req/s (uso aceitável para 246 municípios ~5 min).

    print(f"  Buscando coordenadas via Nominatim (OpenStreetMap) — ~{len(mun_df)} requisições...")
    coords = []
    for _, row in mun_df.iterrows():
        mun_nome = row["municipio_ibge"]
        query = urllib.parse.quote(f"{mun_nome}, {uf}, Brasil")
        url_nom = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={query}&format=json&limit=1&countrycodes=br"
        )
        result = fetch_json(url_nom, timeout=10)
        if result:
            coords.append({
                "cod_mun": row["cod_mun"],
                "municipio": mun_nome,
                "lat": float(result[0]["lat"]),
                "lon": float(result[0]["lon"]),
            })
        else:
            coords.append({"cod_mun": row["cod_mun"], "municipio": mun_nome,
                           "lat": np.nan, "lon": np.nan})
        time.sleep(1.1)  # respeitar rate limit do Nominatim

    centroids = pd.DataFrame(coords)

    # Filtrar para municípios que estão no pipeline
    centroids = centroids[centroids["cod_mun"].isin(pipeline_muns["cod_mun"])]
    n_miss = centroids["lat"].isna().sum()
    if n_miss:
        print(f"  ⚠ {n_miss} municípios sem coordenadas (serão pulados)")

    # Salvar cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(cache_path, index=False)
    print(f"  Centroides salvos em {cache_path}")
    return centroids


def fetch_climate_for_municipio(row: dict, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Baixa dados climáticos diários do Open-Meteo e agrega para mensal."""
    url = OPEN_METEO_URL.format(
        lat=row["lat"], lon=row["lon"],
        start=start_date, end=end_date
    )
    data = fetch_json(url, retries=4, timeout=30)
    if data is None or "daily" not in data:
        return None

    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_media": daily.get("temperature_2m_mean", [np.nan] * len(daily["time"])),
        "temp_min": daily.get("temperature_2m_min", [np.nan] * len(daily["time"])),
        "temp_max": daily.get("temperature_2m_max", [np.nan] * len(daily["time"])),
        "precip": daily.get("precipitation_sum", [np.nan] * len(daily["time"])),
        "umidade": daily.get("relative_humidity_2m_mean", [np.nan] * len(daily["time"])),
    })

    # Agregar diário → mensal
    df["ano_mes"] = df["date"].dt.strftime("%Y-%m")
    monthly = df.groupby("ano_mes").agg(
        temp_media_c=("temp_media", "mean"),
        temp_min_c=("temp_min", "min"),      # mínima do mês = mínima diária mais fria
        temp_max_c=("temp_max", "max"),      # máxima do mês = máxima diária mais quente
        precip_mm=("precip", "sum"),         # precipitação acumulada mensal
        umidade_relativa_pct=("umidade", "mean"),
    ).reset_index()
    monthly["temp_amplitude_c"] = (monthly["temp_max_c"] - monthly["temp_min_c"]).round(2)
    monthly["temp_media_c"] = monthly["temp_media_c"].round(2)
    monthly["temp_min_c"] = monthly["temp_min_c"].round(2)
    monthly["temp_max_c"] = monthly["temp_max_c"].round(2)
    monthly["precip_mm"] = monthly["precip_mm"].round(1)
    monthly["umidade_relativa_pct"] = monthly["umidade_relativa_pct"].round(1)
    monthly.insert(0, "cod_mun", row["cod_mun"])
    monthly.insert(1, "municipio", row["municipio"])
    return monthly


def load_pipeline_municipios(uf: str) -> pd.DataFrame:
    pop = pd.read_csv(RAW_DIR / "pop mensal.csv", encoding="latin1", sep=";")
    pop["cod_mun"] = pop["cod_mun"].astype(str).str[:6].str.zfill(6)
    muns = pop[["cod_mun", "Município"]].drop_duplicates().copy()
    muns = muns.rename(columns={"Município": "municipio"})
    if uf.upper() != "BR":
        state_code = UF_STATE_CODES.get(uf.upper(), "")
        muns = muns[muns["cod_mun"].str.startswith(state_code)]
    return muns.reset_index(drop=True)


def run(args):
    uf = args.uf.upper()
    years_range = args.years or "2017-2025"
    start_year, end_year = years_range.split("-")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    n_workers = args.workers

    print(f"\n{'='*60}")
    print(f"ERA5-Land via Open-Meteo | UF={uf} | {start_date} → {end_date}")
    print(f"{'='*60}")

    # 1) Lista de municípios do pipeline
    pipeline_muns = load_pipeline_municipios(uf)
    print(f"\nMunicípios no pipeline ({uf}): {len(pipeline_muns)}")

    # 2) Centroides
    centroids = get_municipio_centroids(uf, pipeline_muns)
    valid = centroids.dropna(subset=["lat", "lon"])
    print(f"Municípios com coordenadas válidas: {len(valid)}/{len(centroids)}")
    if valid.empty:
        print("ERRO: nenhuma coordenada disponível.")
        sys.exit(1)

    # 3) Download paralelo via Open-Meteo
    print(f"\nBaixando dados climáticos ({n_workers} workers simultâneos)...")
    rows = valid.to_dict("records")
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(fetch_climate_for_municipio, row, start_date, end_date): row
            for row in rows
        }
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                df_mun = future.result()
                if df_mun is not None:
                    results.append(df_mun)
                else:
                    failed.append(row["municipio"])
            except Exception as e:
                failed.append(row["municipio"])
            # Progresso simples
            if done % 20 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} municípios processados"
                      f" ({len(results)} com dados, {len(failed)} falhas)")

    if not results:
        print("ERRO: nenhum dado obtido.")
        sys.exit(1)

    # 4) Concatenar e salvar
    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(["cod_mun", "ano_mes"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "era5_land_go.csv"
    final.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ {out_path}")
    print(f"  Municípios: {final['cod_mun'].nunique()}")
    print(f"  Período:    {final['ano_mes'].min()} → {final['ano_mes'].max()}")
    print(f"  Linhas:     {len(final)}")
    if failed:
        print(f"\n⚠  {len(failed)} municípios com falha: {failed[:10]}")
    print(f"\n>>> Próximo passo:")
    print(f"    Descomente os blocos era5_* em config/variables.yaml")
    print(f"    e rode: python run_pipeline.py")

    print(f"\nAmostra:")
    print(final.head(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa dados climáticos ERA5-Land via Open-Meteo para municípios do modelo"
    )
    parser.add_argument("--uf", default="GO",
                        help="Sigla do estado (ex: GO, SP) ou BR para todo o Brasil")
    parser.add_argument("--years", default="2017-2025",
                        help="Intervalo de anos, ex: 2017-2025")
    parser.add_argument("--workers", type=int, default=4,
                        help="Requisições simultâneas ao Open-Meteo (padrão: 4)")
    run(parser.parse_args())
