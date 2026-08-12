# [Uma Só Saúde] Fatores ambientais, socioeconômicos e assistenciais na predição da síndrome respiratória aguda grave: estudo ecológico em Goiás, 2019–2024

## Este repositório disponibiliza todos os códigos e dados utilizados na elaboração do artigo, permitindo sua completa avaliação, reprodutibilidade de seus experimentos e ampliação de escopo.


# SRAG x Variáveis Socioeconômicas — Goiás (pipeline configurável)

Pipeline de ML para estimar a relação entre variáveis socioeconômicas/de
saúde e casos de Síndrome Respiratória Aguda Grave (SRAG), mensal, por
município.

## Escopo real dos dados (leia antes de tudo)

Os 19 arquivos fornecidos cobrem **exclusivamente o estado de Goiás**
(UF=GO, códigos de município 52xxxx), incluindo a própria variável-alvo (`srag_total.csv`). Este pipeline modela **Goiás**,
e foi desenhado para que, no dia em que dados nacionais equivalentes
existirem, baste apontar `config/periods.yaml` para `uf: BR` e adicionar os
arquivos nacionais ao registro — nenhuma lógica de ingestão/modelagem
precisa ser reescrita.

A variável-alvo (`srag_total.csv`) também só cobre **2019–2024** (semanal).
Fora dessa janela (2017-2018 e 2025) o painel mantém os meses como
ausentes (`NaN`). As demais variáveis têm cada uma sua
própria janela de disponibilidade (ver `outputs/tables/avaliacao_previa_variaveis.csv`
e a coluna `frequencia_original` de `data/processed/coverage_report.csv`).


## Fontes dos dados

Todos os dados utilizados foram obtidos de fontes públicas e abertas de dados, especialmente baseadas no Ministério da Saúde do Brasil, no IBGE e no MapBiomas. Neste repositório, apenas os dados brutos de queimadas livremente disponibilizados pelo MapBiomas não foram aqui fornecidos por limitações de armazenamento, porém, podem ser diretamente recuperados do respectivo projeto: https://plataforma.monitorfogo.mapbiomas.org/ 

## Como rodar

```bash
pip install pandas numpy scikit-learn matplotlib pyyaml openpyxl
python run_pipeline.py
```

Saídas:
- `data/processed/panel_monthly.csv` — painel município x mês, todas as variáveis
- `outputs/tables/` — todas as tabelas de avaliação e métricas
- `outputs/figures/` — todos os gráficos

## Arquitetura

```
config/variables.yaml   # registro de TODAS as variáveis de entrada
config/periods.yaml     # período, frequência, janela de teste
src/io_utils.py          # 1 parser por FORMATO de arquivo (não por arquivo!)
src/build_panel.py        # monta o painel mensal a partir do registro
src/eval_variables.py      # avaliação prévia das variáveis
src/models.py                # os 3 modelos + métricas
src/evaluate_shape.py         # avaliação por município ("shape")
external_layers/              # hook para uso do solo / ar / fogo / emissões
data/external/                 # hook para shapefile/geojson real (mapa)
```

**Para adicionar uma nova variável:** edite `config/variables.yaml` (ver os
comentários no topo do arquivo, a menos
que o formato do arquivo seja realmente novo, caso em que se acrescenta uma
função `parse_<formato>` em `io_utils.py`).

**Para mudar o período/granularidade:** edite `config/periods.yaml`. O
pipeline já é mensal por padrão, mas pode ser apontado para qualquer
`date_start`/`date_end`; a frequência anual->mensal é tratada
automaticamente por interpolação (ver próxima seção).

**Para adicionar cobertura do solo / qualidade do ar / fogo / emissões:**
ver `external_layers/README.md`.

**Para usar um mapana avaliação por
município:** coloque um GeoJSON em `data/external/goias_municipios.geojson`
(ver `data/external/LEIA-ME.txt`). Sem ele, a avaliação por município usa
um gráfico de barras ranqueado.

## Decisões metodológicas explícitas

1. **Semana epidemiológica -> mês**: a base de SRAG é semanal. A
   conversão usa uma aproximação (1º de janeiro do ano + (semana-1)\*7 dias,
   mês resultante) — pequenas distorções perto da virada do mês são
   esperadas e aceitáveis para análise mensal agregada.
2. **Ausência de notificação = zero casos**: dentro da janela real da base
   de SRAG (2019-01 a 2024-12), meses/municípios sem nenhuma linha de
   notificação são tratados como **0 casos**, não como dado faltante.
3. **Interpolação anual -> mensal**: variáveis anuais (PIB, coberturas
   vacinais) são interpoladas linearmente entre os pontos anuais conhecidos
   e, fora da janela observada (ex.: vacinas que só vão até 2022, mas o
   painel pede até 2025), o último valor é repetido (*carry-forward*).
4. **Imputação de variáveis mensais com buracos**: `leitos_uti` ausente ->
   0 (município sem leito de UTI cadastrado); `cobertura_aps` e
   `caged_saldo_emprego` ausentes -> preenchidos por município (ffill/bfill)
   e o resíduo pela mediana global.

## Os 3 métodos de modelagem (e por quê)

O alvo é uma **contagem** (casos de SRAG/mês/município), então os 3 métodos
foram escolhidos para cobrir o espectro linear-interpretável até
não-linear-preditivo, todos nativamente compatíveis com contagens:

1. **Regressão de Poisson regularizada** (`sklearn.linear_model.PoissonRegressor`)
2. **Random Forest Regressor** 
3. **Histogram Gradient Boosting com perda Poisson** 

Avaliação fora da amostra: separação cronológica (sem embaralhar no tempo),
últimos `test_holdout_months` (padrão: 6) meses como teste.




## Execução dos experimentos em cenários:

# Teste rápido (n_iter=5, ~10 min)
python run_experiments.py --fast

# Completo (~2-4h com 24 CPUs)
python run_experiments.py

# Só um cenário específico
python run_experiments.py --scenario S03

# Simulação — cenário de seca severa
python simulate.py --umidade 35.0 --precipitacao 15.0 --temp_media 30.0

# Simulação com CSV por município
python simulate.py --input_csv cenario_2026.csv



