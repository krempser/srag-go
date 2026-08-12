"""
feature_sets.py
---------------
Define os grupos de variáveis para cada cenário de experimento.

REGRA FIXA: vacinas e UTI/leitos NUNCA entram em qualquer experimento.
"""

# Vacinas e UTI — SEMPRE excluídas
EXCLUDED_ALWAYS = {
    "vacina_bcg", "vacina_dtp", "vacina_pentavalente",
    "vacina_pneumococica", "vacina_tetraviral",
    "vacina_triplice_viral_d1", "vacina_triplice_viral_d2",
    "has_uti", "leitos_uti",
    # variáveis brutas (substituídas por taxas)
    "populacao", "srag_casos", "srag_obitos", "srag_taxa_obitos",
    "cadunico_pessoas", "caged_saldo_emprego", "pib_municipal",
}

# Variáveis sociais (demográficas + econômicas + saúde pública)
SOCIAL_FEATURES = [
    "pct_pop_60mais",
    "pct_pop_menor12",
    "pib_per_capita",
    "cadunico_pct_pop",
    "caged_taxa_pop",
    "cobertura_aps",
]

# Variáveis ambientais, climáticas e meteorológicas
ENVIRONMENTAL_FEATURES = [
    # MapBiomas — cobertura do solo
    "mapbiomas_pct_farming",
    "mapbiomas_pct_forest",
    "mapbiomas_pct_non_forest_natural",
    "mapbiomas_pct_non_vegetated",
    "mapbiomas_pct_water",
    # Fogo
    "fogo_pct_territorio",
    # Meteorologia — NASA POWER
    "meteo_temp_media",
    "meteo_temp_min",
    "meteo_temp_max",
    "meteo_temp_amplitude",
    "meteo_precipitacao",
    "meteo_umidade",
]

FEATURE_SETS = {
    "social_and_environmental": SOCIAL_FEATURES + ENVIRONMENTAL_FEATURES,
    "environmental_only":       ENVIRONMENTAL_FEATURES,
}


def get_feature_cols(panel_cols: list[str], feature_set: str) -> list[str]:
    """
    Retorna as features disponíveis para o cenário, intersectando com as
    colunas reais do painel (pode faltar meteo ou fogo se não gerados).
    """
    candidates = FEATURE_SETS[feature_set]
    available  = [f for f in candidates
                  if f in panel_cols and f not in EXCLUDED_ALWAYS]
    missing    = [f for f in candidates if f not in panel_cols]
    if missing:
        print(f"  [feature_sets] Variáveis ausentes no painel: {missing}")
        print(f"    → para incluí-las, rode fetch_meteorologico.py / convert_mapbiomas_fire_tif.py")
    return available


# Features de engenharia temporal (geradas por build_panel.py)
# Lags REMOVIDOS — confundem simulação climática e dominam o modelo
TEMPORAL_FEATURES = [
    "month_sin",
    "month_cos",
    "fase_covid",
]

# Atualizar FEATURE_SETS para incluir temporais em ambos os cenários
FEATURE_SETS["social_and_environmental"] = (
    SOCIAL_FEATURES + ENVIRONMENTAL_FEATURES + TEMPORAL_FEATURES
)
FEATURE_SETS["environmental_only"] = (
    ENVIRONMENTAL_FEATURES + TEMPORAL_FEATURES
)


# Features de alta importância derivadas da história do município e do tempo
# srag_lag_same_phase_prev_year REMOVIDO — era um lag, mesmo problema
HIGH_VALUE_FEATURES = [
    "baseline_srag_municipio",
    "year_trend",
]

# Atualizar FEATURE_SETS — estas features entram em TODOS os cenários
for key in FEATURE_SETS:
    existing = FEATURE_SETS[key]
    FEATURE_SETS[key] = existing + [f for f in HIGH_VALUE_FEATURES
                                    if f not in existing]
