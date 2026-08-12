"""
fetch_meteorologico.py
-----------------------
Baixa dados climáticos mensais para todos os municípios do modelo.

FONTE: NASA POWER API  (https://power.larc.nasa.gov/)
  - Gratuita, sem cadastro, sem chave de API
  - Retorna médias MENSAIS direto — sem precisar agregar dados diários
  - Cobertura global desde 1984, atualizada regularmente
  - Variáveis: temperatura (média/min/max), precipitação, umidade relativa

CENTROIDES (necessários para a API de ponto):
  Estratégia em cascata:
  1. data/external/goias_municipios.geojson  ← se existir, usa o centroide
     do polígono real (muito mais preciso e rápido — recomendado)
  2. IBGE API v3 (malhas municipais simplificadas, calcula centroide do bbox)
     — fallback automático, ~2 min para 246 municípios

USO:
  python src/fetch_meteorologico.py
  python src/fetch_meteorologico.py --uf GO --years 2017-2025 --workers 6

SAÍDA:
  data/processed/meteorologico_go.csv
  Colunas: cod_mun, municipio, ano_mes,
           temp_media_c, temp_min_c, temp_max_c, temp_amplitude_c,
           precip_mm, umidade_relativa_pct

PRÓXIMO PASSO após rodar:
  Descomente os blocos `meteo_*` em config/variables.yaml e rode run_pipeline.py
"""

import sys
import time
import json
import gzip
import argparse
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
EXTERNAL   = ROOT / "data" / "external"
RAW_DIR    = ROOT / "data" / "raw"
GEOJSON_IBGE  = EXTERNAL / "ibge_municipios_go.geojson"   # baixado pelo fire script
GEOJSON_USER  = EXTERNAL / "goias_municipios.geojson"      # GeoJSON do usuário (pode ser estadual)
CACHE_FILE    = EXTERNAL / "centroides_municipios.csv"

UF_CODES = {
    "GO":"52","SP":"35","RJ":"33","MG":"31","BA":"29","RS":"43","PR":"41",
    "SC":"42","CE":"23","PE":"26","PA":"15","MA":"21","AM":"13","MT":"51",
    "MS":"50","PI":"22","AL":"27","SE":"28","PB":"25","RN":"24","TO":"17",
    "ES":"32","AP":"16","RO":"11","RR":"14","AC":"12","DF":"53",
}

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
    "?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M"
    "&community=AG"
    "&longitude={lon}&latitude={lat}"
    "&start={start}&end={end}"
    "&format=JSON"
)


# ── Utilitários HTTP ──────────────────────────────────────────────────────────

def fetch_json(url: str, retries: int = 4, timeout: int = 60) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Accept-Encoding": "gzip, deflate",
                               "User-Agent": "srag-pipeline/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.info().get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── Centroides ────────────────────────────────────────────────────────────────

def centroids_from_geojson(path: Path) -> pd.DataFrame:
    """Calcula centroide do bbox de cada polígono no GeoJSON."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        raw_id = str(props.get("id") or props.get("codigo_ibge") or
                     props.get("codarea") or props.get("CD_MUN") or "")
        cod6 = raw_id.strip()[:6]
        if not cod6:
            continue
        geom = feat.get("geometry", {})
        all_coords = []
        if geom["type"] == "Polygon":
            all_coords = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                all_coords.extend(poly[0])
        if all_coords:
            lons = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            rows.append({"cod_mun": cod6,
                          "lat": (min(lats)+max(lats))/2,
                          "lon": (min(lons)+max(lons))/2})
    df = pd.DataFrame(rows)
    print(f"  GeoJSON → {len(df)} centroides extraídos de {path.name}")
    return df


def centroids_from_ibge(uf: str) -> pd.DataFrame:
    """Fallback: IBGE API v3 (malhas municipais simplificadas → centroide do bbox)."""
    print(f"  Buscando malhas municipais via IBGE API (UF={uf})…")
    url = (f"https://servicodados.ibge.gov.br/api/v3/malhas/estados"
           f"/{uf}?formato=application/vnd.geo%2Bjson"
           f"&resolucao=5&qualidade=minima&intrarregiao=municipio")
    data = fetch_json(url, timeout=120)
    if data is None:
        print("  ERRO: não foi possível acessar a API IBGE. "
              "Coloque data/external/goias_municipios.geojson e tente novamente.")
        sys.exit(1)
    # A resposta usa o mesmo formato GeoJSON; propriedades têm 'codarea'
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        cod6 = str(props.get("codarea", "")).strip()[:6]
        if not cod6:
            continue
        geom = feat.get("geometry", {})
        all_coords = []
        if geom["type"] == "Polygon":
            all_coords = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                all_coords.extend(poly[0])
        if all_coords:
            lons = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            rows.append({"cod_mun": cod6,
                          "lat": (min(lats)+max(lats))/2,
                          "lon": (min(lons)+max(lons))/2})
    df = pd.DataFrame(rows)
    print(f"  IBGE API → {len(df)} centroides")
    return df


def get_centroids(uf: str, pipeline_muns: pd.DataFrame) -> pd.DataFrame:
    # Cache local
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE)
        df = df[df["cod_mun"].isin(pipeline_muns["cod_mun"])]
        if len(df) >= len(pipeline_muns) * 0.9:
            print(f"  Centroides carregados do cache ({len(df)} municípios)")
            return df

    # GeoJSON com polígonos municipais reais (gerado pelo fire script via API IBGE)
    if GEOJSON_IBGE.exists():
        df = centroids_from_geojson(GEOJSON_IBGE)
    elif GEOJSON_USER.exists():
        # Verificar se é municipal (> 50 features) ou estadual
        with open(GEOJSON_USER, encoding="utf-8") as f:
            n = len(json.load(f).get("features", []))
        if n > 50:
            df = centroids_from_geojson(GEOJSON_USER)
        else:
            print(f"  GeoJSON do usuário tem apenas {n} feature(s) — parece estadual. Usando API IBGE.")
            df = centroids_from_ibge(uf)
    else:
        df = centroids_from_ibge(uf)

    # Juntar com nome do município da lista mestre
    df = df.merge(pipeline_muns[["cod_mun","municipio"]], on="cod_mun", how="inner")
    df = df.dropna(subset=["lat","lon"])

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  Cache salvo: {CACHE_FILE}")
    return df


# ── Download NASA POWER ───────────────────────────────────────────────────────

def fetch_nasa_power(row: dict, start_year: int, end_year: int) -> pd.DataFrame | None:
    url = NASA_URL.format(
        lat=round(row["lat"], 4),
        lon=round(row["lon"], 4),
        start=start_year,
        end=end_year,
    )
    data = fetch_json(url, timeout=60)
    if data is None:
        return None
    try:
        params = data["properties"]["parameter"]
    except KeyError:
        return None

    t2m   = params.get("T2M",        {})
    tmin  = params.get("T2M_MIN",    {})
    tmax  = params.get("T2M_MAX",    {})
    prec  = params.get("PRECTOTCORR",{})
    rh    = params.get("RH2M",       {})

    records = []
    for yyyymm in sorted(t2m):
        if len(yyyymm) != 6:
            continue
        mes_num = int(yyyymm[4:])
        if mes_num > 12:          # mês 13 = resumo anual da NASA POWER, ignorar
            continue
        ano_mes = f"{yyyymm[:4]}-{yyyymm[4:]}"
        tm   = t2m.get(yyyymm, np.nan)
        tmi  = tmin.get(yyyymm, np.nan)
        tma  = tmax.get(yyyymm, np.nan)
        pr   = prec.get(yyyymm, np.nan)
        h    = rh.get(yyyymm, np.nan)
        # NASA POWER usa -999 como missing
        tm  = np.nan if tm  == -999 else tm
        tmi = np.nan if tmi == -999 else tmi
        tma = np.nan if tma == -999 else tma
        pr  = np.nan if pr  == -999 else pr
        h   = np.nan if h   == -999 else h
        records.append({
            "cod_mun":             row["cod_mun"],
            "municipio":           row["municipio"],
            "ano_mes":             ano_mes,
            "temp_media_c":        round(tm,  2) if not np.isnan(tm)  else np.nan,
            "temp_min_c":          round(tmi, 2) if not np.isnan(tmi) else np.nan,
            "temp_max_c":          round(tma, 2) if not np.isnan(tma) else np.nan,
            "temp_amplitude_c":    round(tma-tmi, 2) if not (np.isnan(tmi) or np.isnan(tma)) else np.nan,
            "precip_mm":           round(pr,  1) if not np.isnan(pr)  else np.nan,
            "umidade_relativa_pct":round(h,   1) if not np.isnan(h)   else np.nan,
        })
    return pd.DataFrame(records) if records else None


# ── Pipeline principal ────────────────────────────────────────────────────────

def load_pipeline_muns(uf: str) -> pd.DataFrame:
    pop = pd.read_csv(RAW_DIR / "pop mensal.csv", encoding="latin1", sep=";")
    pop["cod_mun"] = pop["cod_mun"].astype(str).str[:6].str.zfill(6)
    muns = (pop[["cod_mun","Município"]].drop_duplicates()
               .rename(columns={"Município":"municipio"}))
    if uf.upper() != "BR":
        code = UF_CODES.get(uf.upper(), "")
        muns = muns[muns["cod_mun"].str.startswith(code)]
    return muns.reset_index(drop=True)


def run(args):
    uf = args.uf.upper()
    years = args.years or "2017-2025"
    start_year, end_year = [int(y) for y in years.split("-")]
    n_workers = args.workers

    print(f"\n{'='*60}")
    print(f"Dados meteorológicos via NASA POWER | UF={uf} | {start_year}–{end_year}")
    print(f"{'='*60}")

    pipeline_muns = load_pipeline_muns(uf)
    print(f"\nMunicípios no pipeline: {len(pipeline_muns)}")

    centroids = get_centroids(uf, pipeline_muns)
    valid = centroids.dropna(subset=["lat","lon"])
    print(f"Municípios com coordenadas válidas: {len(valid)}/{len(pipeline_muns)}")

    print(f"\nBaixando NASA POWER ({n_workers} workers simultâneos)…")
    rows = valid.to_dict("records")
    results, failed = [], []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(fetch_nasa_power, r, start_year, end_year): r
            for r in rows
        }
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                df_mun = future.result()
                if df_mun is not None and not df_mun.empty:
                    results.append(df_mun)
                else:
                    failed.append(row["municipio"])
            except Exception:
                failed.append(row["municipio"])
            if done % 30 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} | OK: {len(results)} | Falha: {len(failed)}")

    if not results:
        print("ERRO: nenhum dado obtido.")
        sys.exit(1)

    final = pd.concat(results, ignore_index=True).sort_values(["cod_mun","ano_mes"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "meteorologico_go.csv"
    final.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"✓ {out}")
    print(f"  Municípios: {final['cod_mun'].nunique()}")
    print(f"  Período:    {final['ano_mes'].min()} → {final['ano_mes'].max()}")
    print(f"  Linhas:     {len(final)}")
    if failed:
        print(f"  Falhas: {len(failed)} — {failed[:5]}")
    print(f"\nPróximo passo: descomente blocos `meteo_*` em config/variables.yaml")
    print(f"e rode: python run_pipeline.py")
    print(final.head(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa dados meteorológicos mensais (NASA POWER) para municípios do modelo"
    )
    parser.add_argument("--uf",      default="GO")
    parser.add_argument("--years",   default="2017-2025")
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())
