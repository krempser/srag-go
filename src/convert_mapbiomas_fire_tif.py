"""
convert_mapbiomas_fire_tif.py
-------------------------------
Processa TIFs MapBiomas Fire e agrega área queimada por município.

REQUISITOS:
  pip install rasterio numpy

USO:
  python src/convert_mapbiomas_fire_tif.py --dir data/raw/fire_tifs/

SAÍDA:
  data/processed/mapbiomas_fire_abs.csv   — hectares queimados por município/mês
  data/processed/mapbiomas_fire_pct.csv   — % do território queimado

POLÍGONOS MUNICIPAIS:
  Baixados automaticamente da API IBGE (malhas municipais de GO) e salvos
  em data/external/ibge_municipios_go.geojson para reutilização.
  Não depende do GeoJSON do usuário (que pode ser estadual).
"""

import sys
import re
import json
import gzip
import time
import argparse
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXTERNAL  = ROOT / "data" / "external"
RAW_DIR   = ROOT / "data" / "raw"

# GeoJSON municipal baixado da API IBGE (cache local)
IBGE_GEOJSON = EXTERNAL / "ibge_municipios_go.geojson"

# Bounding box de Goiás com margem (lon_min, lat_min, lon_max, lat_max)
GO_BBOX = (-53.6, -19.6, -45.4, -10.9)

# Área de um pixel MapBiomas 30m em hectares
PIXEL_HA = 0.09

TIF_PATTERN = re.compile(r"(\d{4})_fire_monthly_(\d+)[-_]", re.IGNORECASE)


# ─── Utilitário HTTP ──────────────────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 120) -> dict | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"Accept-Encoding": "gzip, deflate",
                               "User-Agent": "srag-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.info().get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


# ─── Polígonos municipais via API IBGE ───────────────────────────────────────

def get_municipios_go() -> list[dict]:
    """
    Baixa polígonos municipais de Goiás da API IBGE e salva em cache.
    Retorna lista de {cod_mun, geometry}.
    """
    if IBGE_GEOJSON.exists():
        print(f"  Polígonos municipais carregados do cache: {IBGE_GEOJSON.name}")
        with open(IBGE_GEOJSON, encoding="utf-8") as f:
            gj = json.load(f)
    else:
        print("  Baixando polígonos municipais GO via API IBGE...")
        url = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/GO"
               "?formato=application/vnd.geo%2Bjson"
               "&resolucao=5&qualidade=minima&intrarregiao=municipio")
        gj = fetch_json(url)
        if gj is None:
            print("  ERRO: não foi possível acessar a API IBGE.")
            print("  Verifique sua conexão e tente novamente.")
            sys.exit(1)
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        with open(IBGE_GEOJSON, "w", encoding="utf-8") as f:
            json.dump(gj, f)
        print(f"  Salvo em {IBGE_GEOJSON.name}")

    muns = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        # API IBGE v3 malhas usa 'codarea' com 7 dígitos
        cod = str(props.get("codarea", "")).strip()
        if cod and cod.isdigit():
            muns.append({"cod_mun": cod[:6], "geometry": feat["geometry"]})

    print(f"  {len(muns)} municípios carregados")
    return muns


# ─── Leitura windowed (nunca carrega o raster inteiro) ───────────────────────

def read_goias_window(src):
    import rasterio.windows as rw

    src_crs = src.crs
    bbox = GO_BBOX
    if src_crs and src_crs.to_epsg() != 4326:
        try:
            from rasterio.warp import transform_bounds
            bbox = transform_bounds("EPSG:4326", src_crs, *GO_BBOX)
        except Exception:
            pass

    win = rw.from_bounds(*bbox, transform=src.transform)
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    width   = max(1, min(int(win.width),  src.width  - col_off))
    height  = max(1, min(int(win.height), src.height - row_off))
    win_c   = rw.Window(col_off, row_off, width, height)

    data      = np.array(src.read(1, window=win_c))   # np.array evita o DeprecationWarning
    transform = src.window_transform(win_c)
    nodata    = src.nodata
    return data, transform, nodata


# ─── Zonal stats por município ────────────────────────────────────────────────

def zonal_stats(data, transform, nodata, features) -> dict:
    from rasterio.features import geometry_mask
    nrows, ncols = data.shape
    results = {}
    for feat in features:
        try:
            mask = geometry_mask(
                [feat["geometry"]],
                transform=transform,
                invert=True,
                out_shape=(nrows, ncols),
            )
            pixels = data[mask]
            if nodata is not None:
                pixels = pixels[pixels != nodata]
            results[feat["cod_mun"]] = int(np.sum(pixels > 0)) * PIXEL_HA
        except Exception:
            results[feat["cod_mun"]] = 0.0
    return results


# ─── Processar um TIF ─────────────────────────────────────────────────────────

def process_tif(tif_path: Path, features: list, diag: bool = False) -> pd.DataFrame | None:
    import rasterio

    m = TIF_PATTERN.search(tif_path.name)
    if not m:
        print(f"  IGNORADO (nome não reconhecido): {tif_path.name}")
        return None

    ano, mes = int(m.group(1)), int(m.group(2))
    ano_mes  = f"{ano}-{mes:02d}"

    try:
        with rasterio.open(tif_path) as src:
            if diag:
                print(f"  CRS: {src.crs} | Dtype: {src.dtypes[0]} | "
                      f"Shape: {src.height}×{src.width} | NoData: {src.nodata}")
            data, transform, nodata = read_goias_window(src)
            if diag:
                sample = data[:500, :500]
                print(f"  Valores únicos (amostra): {np.unique(sample)[:10]}")
                print(f"  Janela GO shape: {data.shape}")

        burned = zonal_stats(data, transform, nodata, features)
    except Exception as e:
        print(f"  ERRO: {e}")
        return None

    rows = [{"cod_mun": str(cod), "ano_mes": ano_mes, "area_ha": ha}
            for cod, ha in burned.items()]
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    tif_dir = Path(args.dir)

    print(f"\n{'='*60}")
    print(f"MapBiomas Fire TIF → CSV  |  {tif_dir}")
    print(f"{'='*60}\n")

    try:
        import rasterio
        print(f"rasterio {rasterio.__version__} OK")
    except ImportError:
        print("ERRO: pip install rasterio")
        sys.exit(1)

    print()
    features = get_municipios_go()

    tifs = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.TIF"))
    if not tifs:
        print(f"ERRO: nenhum .tif em {tif_dir}")
        sys.exit(1)
    print(f"\n{len(tifs)} TIFs encontrados")

    print(f"\nDiagnóstico: {tifs[0].name}")
    process_tif(tifs[0], features[:3], diag=True)

    print(f"\nProcessando (sequencial — controle de memória)…")
    results = []
    for i, tif in enumerate(tifs, 1):
        print(f"  [{i}/{len(tifs)}] {tif.name}", end=" ", flush=True)
        df = process_tif(tif, features)
        if df is not None:
            results.append(df)
            print(f"→ {df['area_ha'].sum():,.0f} ha queimados em GO")
        else:
            print("→ ignorado")

    if not results:
        print("Nenhum resultado.")
        sys.exit(1)

    final = (pd.concat(results, ignore_index=True)
               .assign(cod_mun=lambda d: d["cod_mun"].astype(str))
               .sort_values(["cod_mun", "ano_mes"]))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final.to_csv(PROCESSED / "mapbiomas_fire_abs.csv", index=False)
    print(f"\n✓ mapbiomas_fire_abs.csv → {final.shape}")

    # % do território
    abs_cover = PROCESSED / "mapbiomas_classes_abs.csv"
    if abs_cover.exists():
        cover = pd.read_csv(abs_cover)
        cover["cod_mun"] = cover["cod_mun"].astype(str)          # garantir str
        ha_cols = [c for c in cover.columns
                   if c.startswith("ha_") and "not_observed" not in c]
        cover["territorio_ha"] = cover[ha_cols].sum(axis=1)
        cover["ano"] = cover["ano_mes"].str[:4]
        final["ano"] = final["ano_mes"].str[:4]
        territorio = (cover.groupby(["cod_mun", "ano"])["territorio_ha"]
                           .mean().reset_index()
                           .assign(cod_mun=lambda d: d["cod_mun"].astype(str)))
        final = final.merge(territorio, on=["cod_mun", "ano"], how="left")
        final["pct_territorio_queimado"] = (
            final["area_ha"] / final["territorio_ha"].replace(0, np.nan) * 100
        ).round(4)
        pct = final[["cod_mun", "ano_mes", "pct_territorio_queimado"]]
    else:
        pct = final[["cod_mun", "ano_mes"]].copy()
        pct["pct_territorio_queimado"] = np.nan

    pct.to_csv(PROCESSED / "mapbiomas_fire_pct.csv", index=False)
    print(f"✓ mapbiomas_fire_pct.csv → {pct.shape}")

    print(f"\nPeríodo: {final['ano_mes'].min()} → {final['ano_mes'].max()}")
    print(f"Municípios: {final['cod_mun'].nunique()}")
    print("\nTop 5 por área queimada total:")
    print(final.groupby("cod_mun")["area_ha"].sum().nlargest(5))
    print("\nPróximo passo: descomente os blocos `fogo_*` em config/variables.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    run(parser.parse_args())
