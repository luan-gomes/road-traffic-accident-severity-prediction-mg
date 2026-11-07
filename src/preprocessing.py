from pathlib import Path
from IPython.display import display
import pandas as pd
from src.constants import (
    COLUMNS_PER_DATA_GROUP,
    DATA_GROUPS,
    YEARS,
    VEHICLES_CATEGORIES,
)

def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Retorna um DataFrame a partir de um path."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: '{path}'")
    
    header = 0 if columns != None else 'infer'
    return pd.read_csv(path, sep=';', header=header, names=columns)

def load_data_group_df_from_data_dir(data_group: str, data_dir: Path, years: tuple[int] = YEARS) -> pd.DataFrame:
    """Carrega todos os arquivos CSV de um data group a partir do diretório de dados do projecto e retorna um único DataFrame."""
    data_group_dir = data_dir / data_group
    if not data_group_dir.exists() or not data_group_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: '{data_group_dir}'") 
    
    if data_group not in COLUMNS_PER_DATA_GROUP.keys():
        raise KeyError(f"Data group '{data_group}' não encontrados no mapeamento de colunas.")
    
    frames = []
    columns = COLUMNS_PER_DATA_GROUP[data_group]
    for year in years:
        year_file_path = data_group_dir / f"{year}.csv"
        frames.append(read_csv(year_file_path, columns))

    return pd.concat(frames, ignore_index=True)

def get_df_per_data_group(data_dir: Path,  years: tuple[int] = YEARS) -> dict[str: pd.DataFrame]:
    """Retorna um dicionário com o DataFrame completo de cada data group."""
    raw_dfs = {}
    for data_group in DATA_GROUPS:
       raw_dfs[data_group] = load_data_group_df_from_data_dir(data_group, data_dir, years)
    return raw_dfs

def display_basic_df_info(data_group: str, df: pd.DataFrame):
    """Exibir informações básicas sobre o DataFrame de um data group."""
    print(f"Data group: {data_group}")
    print(f"Números de linhas: {df.shape[0]}")
    print(f"Número de colunas: {df.shape[1]}")
    print("Colunas disponíveis:")
    display(pd.DataFrame({"Nome da coluna": df.columns, "Tipo da coluna": [str(dtype) for dtype in df.dtypes]}))
    print("Primeiras 5 linhas do DataFrame:")
    display(df.head())

def display_basic_df_info_per_data_group(df_per_data_group: dict[str, pd.DataFrame]):
    """Exibe informações básicas de cada DataFrame dos data groups."""
    for data_group in DATA_GROUPS:
        if data_group not in df_per_data_group.keys():
            raise KeyError(f"Data group '{data_group}' not found in df_per_data_group keys.")
        display_basic_df_info(data_group, df_per_data_group[data_group])

def identify_multiple_rows_per_occurrence(df_per_data_group: dict[str, pd.DataFrame], occurrence_col: str):
    """Verifica e exibe quais DataFrames possuem mais de uma linha para a mesma ocorrência."""
    for data_group in DATA_GROUPS:
        if data_group not in df_per_data_group.keys():
            raise KeyError(f"Data group '{data_group}' não encontrado em df_per_data_group keys.")
        
        df = df_per_data_group[data_group]
        if occurrence_col not in df.columns:
            print(f"[{data_group}]: Ignorado. A coluna '{occurrence_col}' não existe neste dataset.\n")
            continue
        
        total_rows = len(df)
        unique_occurrences = df[occurrence_col].nunique()
        if total_rows > unique_occurrences:
            extra_rows = total_rows - unique_occurrences
            print(f"[{data_group}]: POSSUI múltiplas linhas por ocorrência.")
            print(f"   -> Total de linhas: {total_rows}")
            print(f"   -> Ocorrências únicas: {unique_occurrences}")
            print(f"   -> Linhas adicionais (repetições): {extra_rows}\n")
        else:
            print(f"[{data_group}]: Relação 1:1. Possui apenas uma linha por ocorrência.\n")

def aggregate_vitimas_dataset(df_victims: pd.DataFrame) -> pd.DataFrame:
    """ 
    Agrupa o dataset de vítimas por ocorrência, criando features agregadas e a variável target.
    """
    df = df_victims.copy()
    
    condicoes_graves = ['GRAVES OU INCONSCIENTE', 'FATAL']
    df['is_grave'] = df['vit_condicao_fisica_descricao'].isin(condicoes_graves)
    
    df['is_pedestre'] = df['vit_tipo_envolvimento'] == 'VITIMA E PEDESTRE (TRANSITO)'
    
    # Tratamento de segurança para a coluna de idade (força conversão para número, erros viram NaN)
    df['vit_valor_idade_aparente'] = pd.to_numeric(df['vit_valor_idade_aparente'], errors='coerce', downcast='integer')
    
    df_grouped = df.groupby('numero_ocorrencia_associado').agg(
        qtd_vitimas=('numero_ocorrencia_associado', 'size'),
        has_pedestre_envolvido=('is_pedestre', 'max'),
        acidente_grave=('is_grave', 'max'),
        idade_media_vitimas=('vit_valor_idade_aparente', 'mean'),
        idade_maxima_vitima=('vit_valor_idade_aparente', 'max')
    ).reset_index()
    
    # Converter booleanos (True/False) do 'max' para inteiros (1/0)
    df_grouped['has_pedestre_envolvido'] = df_grouped['has_pedestre_envolvido'].astype(int)
    df_grouped['acidente_grave'] = df_grouped['acidente_grave'].astype(int)

    # Converter idade_maxima_vitima para Int64
    df_grouped['idade_maxima_vitima'] = df_grouped['idade_maxima_vitima'].astype('Int64')
    
    return df_grouped

def aggregate_condicoes_via_dataset(df_condicoes: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa o dataset de condições da via por ocorrência, criando flags de risco 
    relevantes para a predição da gravidade do acidente.
    """
    df = df_condicoes.copy()
        
    # Luminosidade crítica: Noite sem iluminação
    df['is_noite_sem_iluminacao'] = (
        df['con_luminosidade_descricao'] == 'NOITE SEM ILUMINACAO'
    ).astype(int)
    
    # Período noturno geral (com ou sem iluminação)
    df['is_periodo_noturno'] = df['con_luminosidade_descricao'].isin(
        ['NOITE/ILUMINACAO ARTIFICIAL', 'NOITE SEM ILUMINACAO']
    ).astype(int)
    
    # Sinalização precária ou ausente
    sinalizacao_ruim = ['NAO HA', 'IRREGULAR', 'EM MAS CONDICOES']
    df['has_sinalizacao_deficiente'] = df['con_condicao_sinalizacao_descricao'].isin(
        sinalizacao_ruim
    ).astype(int)
    
    # Pista simples (associada a colisões frontais mais graves)
    df['is_pista_simples'] = (
        df['con_tipo_via_descricao'] == 'PISTA SIMPLES'
    ).astype(int)
    
    # Pavimento não pavimentado / terra / cascalho
    pavimento_ruim = ['TERRA', 'CASCALHO (MINERAL)']
    df['is_pista_nao_pavimentada'] = df['con_tipo_pavimento_descricao'].isin(
        pavimento_ruim
    ).astype(int)
    
    df_grouped = df.groupby('numero_ocorrencia_associado').agg(
        is_noite_sem_iluminacao=('is_noite_sem_iluminacao', 'max'),
        is_periodo_noturno=('is_periodo_noturno', 'max'),
        has_sinalizacao_deficiente=('has_sinalizacao_deficiente', 'max'),
        is_pista_simples=('is_pista_simples', 'max'),
        is_pista_nao_pavimentada=('is_pista_nao_pavimentada', 'max')
    ).reset_index()
    
    return df_grouped

def process_ocorrencias_base_dataset(df_ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """
    Processa e filtra o dataset base de ocorrências com base nos valores exatos
    do REDS-MG para Belo Horizonte ou Região Metropolitana (RMBH).
    """
    df = df_ocorrencias.copy()
    
    df['oco_nome_municipio_clean'] = df['oco_nome_municipio'].astype(str).str.strip().str.upper()
    df['oco_rmbh_clean'] = df['oco_rmbh'].astype(str).str.strip().str.upper()
        
    # Fim de semana considerando o padrão exato dos dados ('Sábado', 'Domingo')
    dias_fim_semana = ['sábado', 'domingo']
    df['is_fim_de_semana'] = (
        df['oco_dia_da_semana'].astype(str).str.lower().isin(dias_fim_semana)
    ).astype(int)
    
    # Extração de hora e faixas de horário de risco
    df['hora_fato'] = pd.to_datetime(df['oco_horario_fato'], format='%H:%M:%S', errors='coerce').dt.hour
    df['is_madrugada'] = df['hora_fato'].between(0, 5, inclusive='both').fillna(False).astype(int)
    df['is_horario_pico'] = df['hora_fato'].isin([7, 8, 9, 17, 18, 19]).fillna(False).astype(int)
    
    # Periodo chuvoso em MG (Novembro a Março)
    df['is_periodo_chuvoso_mg'] = df['oco_mes_numerico_fato'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Classificação de vias de maior velocidade
    vias_rapidas = ['AVENIDA', 'RODOVIA', 'ESTRADA', 'VIA', 'VIADUTO']
    df['is_via_rapida'] = (
        df['oco_tipo_logradouro_descricao'].astype(str).str.upper().isin(vias_rapidas)
    ).astype(int)

    # Flag indicando se é o município de BH
    df['is_municipio_bh'] = (df['oco_nome_municipio_clean'] == 'BELO HORIZONTE').astype(int)
    
    # Flag indicando se é Região Metropolitana de BH
    df['is_rmbh'] = (df['oco_rmbh_clean'] == 'SIM').astype(int)

    cols_finais = [
        'numero_ocorrencia_associado',
        'is_municipio_bh',
        'is_rmbh',
        'is_fim_de_semana',
        'hora_fato',
        'is_madrugada',
        'is_horario_pico',
        'is_periodo_chuvoso_mg',
        'is_via_rapida',
        'oco_ano_fato',
        'oco_tipo_logradouro_descricao',
        'oco_natureza_descricao',
        'oco_nome_bairro',
        'oco_nome_municipio',
        'oco_latitude_sirgas2000',
        'oco_longitude_sirgas2000'
    ]
    
    cols_existentes = [c for c in cols_finais if c in df.columns]
    df_processed = df[cols_existentes].copy()
    
    df_processed = df_processed.drop_duplicates(subset=['numero_ocorrencia_associado'])
    
    return df_processed

def aggregate_via_dataset(df_via: pd.DataFrame) -> pd.DataFrame:
    """
    Processa e agrupa o dataset de defeitos/condições da via.
    Converte valores 'Sim'/'Não' para 1/0 e gera colunas sintéticas de defeitos.
    """
    df = df_via.copy()
    
    cols_sim_nao = [
        'via_boa', 'via_buracos', 'via_material_na_pista', 
        'via_ondulada', 'via_rachadura', 'via_sulcos_horizontais', 
        'via_outras_no_historico'
    ]
    
    for col in cols_sim_nao:
        if col in df.columns:
            # Mapeia 'Sim' para 1 e qualquer outro valor ('Não', NaN) para 0
            df[f'{col}_flag'] = (
                df[col].astype(str).str.strip().str.capitalize() == 'Sim'
            ).astype(int)
            
    cols_defeitos_especificos = [
        'via_buracos_flag', 'via_material_na_pista_flag', 
        'via_ondulada_flag', 'via_rachadura_flag', 
        'via_sulcos_horizontais_flag', 'via_outras_no_historico_flag'
    ]
    
    # Soma total de defeitos físicos registrados no local
    df['qtd_defeitos_via'] = df[cols_defeitos_especificos].sum(axis=1)
    
    # Flag geral: 1 se a via NÃO estava boa OU se foi registrado qualquer defeito físico
    df['has_defeito_via'] = (
        (df['via_boa_flag'] == 0) | (df['qtd_defeitos_via'] > 0)
    ).astype(int)
    
    df_grouped = df.groupby('numero_ocorrencia_associado').agg(
        has_via_boa=('via_boa_flag', 'max'),
        has_via_buracos=('via_buracos_flag', 'max'),
        has_material_na_pista=('via_material_na_pista_flag', 'max'),
        has_via_ondulada=('via_ondulada_flag', 'max'),
        has_via_rachadura=('via_rachadura_flag', 'max'),
        has_via_sulcos_horizontais=('via_sulcos_horizontais_flag', 'max'),
        qtd_defeitos_via=('qtd_defeitos_via', 'max'),
        has_defeito_via=('has_defeito_via', 'max')
    ).reset_index()
    
    return df_grouped

def aggregate_veiculos_dataset(veiculos_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa o dataset do grupo veiculo e cria novas colunas"""
    df = veiculos_df.copy()

    df['is_moto'] = df['vei_tipo_veiculo_descricao_longa'].isin(VEHICLES_CATEGORIES['motocicleta']).astype(int)

    df['is_bicicleta'] = df['vei_tipo_veiculo_descricao_longa'].isin(VEHICLES_CATEGORIES['bicicleta']).astype(int)

    df['is_pesado'] = df['vei_tipo_veiculo_descricao_longa'].isin(VEHICLES_CATEGORIES['pesado']).astype(int)

    new_df = df.groupby('numero_ocorrencia_associado').agg(
        qtd_veiculos_envolvidos=('vei_tipo_veiculo_descricao_longa', 'size'),
        has_motocicleta_envolvida=('is_moto', 'any'),
        has_bicicleta_envolvida=('is_bicicleta', 'any'),
        has_veiculo_pesado_envolvido=('is_pesado', 'any')
    ).reset_index()

    return new_df

def build_analytical_base_table(
    df_ocorrencias: pd.DataFrame,
    df_vitimas: pd.DataFrame,
    df_condicoes: pd.DataFrame,
    df_via: pd.DataFrame,
    df_veiculos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orquestra o processamento e merge de todos os datasets de acidentes de trânsito,
    garantindo a integridade da granularidade (1 linha por ocorrência).
    """    
    # 1. Processar a Tabela Base (Ocorrências)
    print("1/4. Processando tabela base de ocorrências...")
    df_base = process_ocorrencias_base_dataset(df_ocorrencias)
    total_base_rows = len(df_base)
    print(f"   -> Tabela base pronta com {total_base_rows} ocorrências únicas.\n")
    
    # 2. Agrupar Datasets Secundários (1 linha por ocorrência)
    print("2/4. Agrupando datasets secundários...")
    df_vit_agg = aggregate_vitimas_dataset(df_vitimas)
    df_cond_agg = aggregate_condicoes_via_dataset(df_condicoes)
    df_via_agg = aggregate_via_dataset(df_via)
    df_veiculos_agg = aggregate_veiculos_dataset(df_veiculos)
    print("   -> Agrupamentos concluídos.\n")
    
    # 3. Executar Joins Encadeados (LEFT JOIN)
    print("3/4. Realizando merges (LEFT JOINs)...")
    df_abt = df_base.merge(df_vit_agg, on='numero_ocorrencia_associado', how='left')
    df_abt = df_abt.merge(df_cond_agg, on='numero_ocorrencia_associado', how='left')
    df_abt = df_abt.merge(df_via_agg, on='numero_ocorrencia_associado', how='left')
    df_abt = df_abt.merge(df_veiculos_agg, on='numero_ocorrencia_associado', how='left')
    
    # Validação de Integridade (Defesa de Código)
    if len(df_abt) != total_base_rows:
        raise ValueError(
            f"ERRO DE INTEGRIDADE: O dataset final possui {len(df_abt)} linhas, "
            f"mas a tabela base possui {total_base_rows}. Houve duplicação de chaves durante o merge!"
        )
    print("   -> Validação concluída: Nenhuma duplicata gerada no merge.\n")
    
    # 4. Tratamento de Valores Nulos (NaN) Pós-Merge
    print("4/4. Tratando valores nulos decorrentes do LEFT JOIN...")
    
    # 4.1. Flags binárias e contagens: Se não houve match no LEFT JOIN, o valor real é 0
    # Exemplo: Se o acidente não aparece na tabela de vítimas, significa 0 vítimas e 0 graves.
    cols_zero_fill = [
        'qtd_vitimas', 'has_pedestre_envolvido', 'acidente_grave',
        'is_noite_sem_iluminacao', 'is_periodo_noturno', 'has_sinalizacao_deficiente',
        'is_pista_simples', 'is_pista_nao_pavimentada',
        'has_via_boa', 'has_via_buracos', 'has_material_na_pista',
        'has_via_ondulada', 'has_via_rachadura', 'has_via_sulcos_horizontais',
        'qtd_defeitos_via', 'has_defeito_via', 'has_motocicleta_envolvida',
        'has_bicicleta_envolvida', 'has_veiculo_pesado_envolvido'
    ]
    
    for col in cols_zero_fill:
        if col in df_abt.columns:
            df_abt[col] = df_abt[col].fillna(0).astype(int)

    print(f"ABT gerada com sucesso! Dimensões finais: {df_abt.shape[0]} linhas x {df_abt.shape[1]} colunas.\n")
    
    return df_abt

def apply_filters(original_df: pd.DataFrame,  years: tuple[int] = YEARS) -> pd.DataFrame:
    """
    Filtra a base para manter apenas:
    - Ocorrências nos anos especificados;
    - Ocorrências com PELO MENOS 1 VÍTIMA.
    """
    filter_condition = (
        (original_df['oco_ano_fato'].isin(years)) &
        (original_df['qtd_vitimas'] > 0)
    )
    return original_df[filter_condition].copy()

def check_missing_values(df: pd.DataFrame, show_only_missing: bool = True) -> pd.DataFrame:
    """Analisa e exibe a quantidade e o percentual de valores faltantes por coluna."""
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / total_rows) * 100
    
    summary_df = pd.DataFrame({
        'coluna': df.columns,
        'tipo_dado': df.dtypes.values,
        'qtd_nulos': missing_count.values,
        'percentual_nulos (%)': missing_percent.round(2).values
    })
    
    if show_only_missing:
        summary_df = summary_df[summary_df['qtd_nulos'] > 0]
        
    summary_df = summary_df.sort_values(by='qtd_nulos', ascending=False).reset_index(drop=True)
    
    return summary_df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza de valores faltantes com base nos resultados da função check_missing_values"""
    df_clean = df.copy()
    
    # 1. Remover linhas sem coordenadas geográficas (caso vá usar em mapa ou modelo)
    df_clean = df_clean.dropna(subset=['oco_latitude_sirgas2000', 'oco_longitude_sirgas2000'])
    
    # 2. Imputar a quantidade de veículos com a moda (se houver nulos)
    if 'qtd_veiculos_envolvidos' in df_clean.columns and df_clean['qtd_veiculos_envolvidos'].isnull().any():
        mode_val = df_clean['qtd_veiculos_envolvidos'].mode()[0]
        df_clean['qtd_veiculos_envolvidos'] = df_clean['qtd_veiculos_envolvidos'].fillna(mode_val)
        
    # 3. Imputar a idade média/máxima restante com a mediana do dataset de vítimas
    cols_idade = ['idade_media_vitimas', 'idade_maxima_vitima']
    for col in cols_idade:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    return df_clean
