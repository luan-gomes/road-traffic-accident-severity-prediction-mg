# road-traffic-accident-severity-prediction-mg
Repositório utilizando no Trabalho de Conclusão de Curso da Especialização em Ciência de Dados da UNIMONTES.

## Sobre este repositório

Este repositório é utilizado na disciplina de Trabalho de Conclusão de Curso, no semestre 2026.2, da Especialização em Ciência de Dados da Universidade Estadual de Montes Claros (UNIMONTES).

## Sobre o projeto

Comparar o desempenho de diferentes modelos de aprendizado de máquina na predição da gravidade de acidentes de trânsito com vítimas ocorridos em Minas Gerais, utilizando dados provenientes da Base de Dados do Painel de Sinistro de Trânsito da Secretaria de Estado de Justiça e Segurança Pública de Minas Gerais, com o propósito de apoiar a priorização do atendimento de emergência às ocorrências. 

## Sobre os dados

**Fonte:** Base de Dados do Painel Acidente de Trânsito da Secretaria de Estado de Justiça e Segurança Pública (Sejusp) de Minas Gerais.
**Link:** [https://drive.google.com/drive/folders/1W_4SsAW8u5bvsaBoHhwInpUr72vES8yw?usp=sharing](https://drive.google.com/drive/folders/1W_4SsAW8u5bvsaBoHhwInpUr72vES8yw?usp=sharing)  
**Data de acesso e coleta de dados:** 06 de Setembro de 2026

**Procedimento de obtenção dos dados:**
A base de dados original está armazenada em uma pasta compartilhada do Google Drive e possui 5 subpastas (também referenciadas como data groups ao longo deste trabalho): Vítimas, Via, Veículo, Ocorrências e Condições. Cada subpasta possui um arquivo `.csv` para cada ano contendo dados de Janeiro a Dezembro do respectivo ano, com exceção, na data da execução desta análise (06 de Setembro de 2026), para o ano atual que possui dois arquivos, um abrangendo o perídio de Janeiro a Junho e outro de Janeiro a Julho. 

- Criar um diretório `data` na raiz deste repositório e, dentro desse novo diretório, criar as seguintes subpastas: `condicoes`, `ocorrencias`, `veiculo`, `via`, `vitimas`.
- Acessar o link da fonte de dados.
- Acessar cada uma das subpastas e baixar o arquivo `.csv` dos anos 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.
- Após o download, mover os arquivos de cada data group para a sua respectiva subpasta dentro de `data`.
- Renomear os arquivos `.csv` para o seguinte padrão `<ANO-NUMÉRICO>.csv`, por exemplo, o arquivo `Vítimas de Acidente de Trânsito - MG - Jan a Jun 2026.csv` deve ser renomeado para `2026.csv`.

## Executar as análises

Recomenda-se Python 3.11, 3.12 ou 3.13 em um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Acessar os notebooks no diretório `./notebooks` e executar as células em ordem sequencial. Ordem recomendada: `eda.ipynb > eda_10_years.ipynb > modeling.ipynb`.

## Estrutura

```text
road-traffic-accident-severity-prediction-mg/
├── data/
│   ├── condicoes/                     # Dados obtidos conforme seção "Sobre os dados" referentes ao data group "Condições"
│   ├── ocorrencias/                   # Dados obtidos conforme seção "Sobre os dados" referentes ao data group "Ocorrências"
│   ├── veiculo/                       # Dados obtidos conforme seção "Sobre os dados" referentes ao data group "Veículo"
│   ├── via/                           # Dados obtidos conforme seção "Sobre os dados" referentes ao data group "Via"
│   └── vitimas/                       # Dados obtidos conforme seção "Sobre os dados" referentes ao data group "Vítimas"
├── notebooks/
│   ├── eda.ipynb               # Análise exploratória considerando 5 anos (2021 - 2025)
│   ├── eda_10_years.ipynb      # Análise exploratória considerando 10 anos (2016 - 2025)
│   └── modeling.ipynb          # Seeleção de atributos, o treinamento dos modelos e a comparação entre os recortes temporais de 5 e de 10 anos
├── src/
│   ├── modeling.py             # Funções de modelagem
│   ├── preprocessing.py        # Funções da análise exploratória
│   └── visualization.py        # Funções da visualização e plotagem de gráficos
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitações

- Parte dos atributos, como a quantidade e a idade das vítimas e as condições da via, vem do boletim de ocorrência e pode não estar disponível no momento do chamado.
- A proporção de graves em 2025 (16,41%) é menor que a dos anos de treino, o que dificulta a comparação direta entre as métricas do teste e as da validação cruzada.
- Os espaços de busca de hiperparâmetros foram limitados: nos modelos com mais de 10 combinações, foram testadas apenas 10 e, em vários modelos, inclusive na Regressão Logística, o melhor valor ficou no limite da grade testada.
- No recorte de 10 anos, apenas os finalistas não lineares foram reajustados.