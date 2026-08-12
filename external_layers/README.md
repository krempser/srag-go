# Camadas externas (uso do solo, qualidade do ar, fogo, emissões)

Este projeto já está preparado para receber essas camadas sem mudança de
arquitetura. Veja como adicionar cada uma.

## Passo a passo genérico
1. Baixe/exporte os dados na granularidade município x mês (ou município x ano,
   que o pipeline interpola).
2. Salve um CSV aqui em `external_layers/` com colunas:
   `cod_mun, ano_mes, valor` (cod_mun = código de 6 dígitos, ano_mes = "YYYY-MM").
3. Descomente/adapte o bloco correspondente em `config/variables.yaml`
   (já deixados prontos, comentados, no final do arquivo).
4. Rode `python run_pipeline.py` de novo. Nenhum código precisa mudar.

## Fontes públicas recomendadas por camada

**Cobertura/uso do solo**
- MapBiomas Coleção (mapbiomas.org) — séries anuais de uso e cobertura do
  solo por município, incluindo % área urbana, agropecuária, vegetação nativa.
- TerraClass/INPE para o Cerrado.

**Qualidade do ar / material particulado**
- Copernicus CAMS (Copernicus Atmosphere Monitoring Service) — reanálise
  global de PM2.5, PM10, NO2, ozônio, disponível por coordenada/grade.
- Plataforma SATVeg/INPE ou estações da rede MonitorAr (cobertura limitada
  em GO, a maior parte das estações fica em capitais/RMs).

**Fogo / queimadas**
- INPE BDQueimadas (queimadas.dgi.inpe.br) — focos de calor diários por
  município, satélites de referência (AQUA_M-T), download em CSV.
- INPE Programa Queimadas — área queimada acumulada mensal.

**Emissões de poluentes/GEE**
- SEEG / Observatório do Clima (seeg.eco.br) — emissões de GEE por
  município e setor, série anual.
- Inventários estaduais de emissões (SEMAD-GO, quando disponíveis).

## Chaves
A junção é sempre pela mesma chave (`cod_mun` + mês), e o painel final já é
construído num formato longo intermediário (`data/processed/panel_long.csv`)
antes de virar tabela larga — então adicionar uma variável nova nunca exige
reescrever lógica de junção, só describer o arquivo de origem em
`variables.yaml`.
