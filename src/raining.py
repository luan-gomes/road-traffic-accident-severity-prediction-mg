from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.constants import (
    INMET_HEADER_LINES,
    INMET_COLUMNS,
    INMET_MAX_DISTANCE_KM,
    INMET_NEAREST_STATIONS,
    EARTH_RADIUS_KM,
    LOCAL_TIMEZONE
)


def read_inmet_station_metadata(path: Path) -> dict:
    """Retorna o código, o nome e as coordenadas da estação, contidos nas linhas iniciais do arquivo do INMET."""
    with open(path, encoding='utf-8') as file:
        header_lines = [file.readline() for _ in range(INMET_HEADER_LINES)]

    # Cada linha segue o formato "Chave: valor". Ex.: "Codigo Estacao: A502"
    metadata = dict(line.strip().split(': ', 1) for line in header_lines if ': ' in line)
    return {
        'codigo': metadata['Codigo Estacao'],
        'nome': metadata['Nome'],
        'latitude': float(metadata['Latitude']),
        'longitude': float(metadata['Longitude']),
    }


def read_inmet_precipitation(path: Path) -> pd.Series:
    """Retorna a precipitação horária (mm) de uma estação automática do INMET, indexada pelo horário UTC da medição."""
    df = pd.read_csv(path, sep=';', skiprows=INMET_HEADER_LINES, decimal=',', usecols=[0, 1, 2], dtype={'Hora Medicao': str})

    if list(df.columns) != list(INMET_COLUMNS):
        raise ValueError(
            f"Cabeçalho inesperado no arquivo '{path}'.\n"
            f"   -> Esperado: {list(INMET_COLUMNS)}\n"
            f"   -> Encontrado: {list(df.columns)}"
        )

    horario_utc = pd.to_datetime(df['Data Medicao'] + df['Hora Medicao'], format='%Y-%m-%d%H%M')
    return pd.Series(df['PRECIPITACAO TOTAL, HORARIO(mm)'].to_numpy(), index=horario_utc)


def load_inmet_precipitation_data(inmet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os arquivos das estações automáticas do INMET (um arquivo por estação) e retorna:
    - df_estacoes: uma linha por estação, com código, nome, latitude e longitude;
    - df_precipitacao: precipitação horária (mm), com uma linha por horário UTC e uma coluna por estação.
    """
    if not inmet_dir.exists() or not inmet_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: '{inmet_dir}'")

    estacoes = []
    precipitacao = {}
    for path in sorted(inmet_dir.glob('dados_*_H_*.csv')):
        estacao = read_inmet_station_metadata(path)
        if estacao['codigo'] in precipitacao:
            raise ValueError(f"A estação '{estacao['codigo']}' aparece em mais de um arquivo.")
        estacoes.append(estacao)
        precipitacao[estacao['codigo']] = read_inmet_precipitation(path)

    if not estacoes:
        raise FileNotFoundError(f"Nenhum arquivo do INMET encontrado em '{inmet_dir}'")

    return pd.DataFrame(estacoes), pd.DataFrame(precipitacao)


def get_inmet_record_time(df_ocorrencias: pd.DataFrame) -> pd.Series:
    """
    Retorna o horário (UTC) do registro do INMET que corresponde à hora do acidente.
    Cada registro acumula a chuva da hora anterior. Ex.: um acidente às 14:35 (17:35 UTC) corresponde ao registro das 18:00 UTC.
    Os horários que não existem ou que se repetem na troca do horário de verão (até 2019) ficam nulos.
    """
    horario_local = pd.to_datetime(
        df_ocorrencias['oco_data_fato'] + ' ' + df_ocorrencias['oco_horario_fato'],
        format='%Y-%m-%d %H:%M:%S', errors='coerce'
    )
    horario_utc = horario_local.dt.tz_localize(LOCAL_TIMEZONE, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    return horario_utc.dt.tz_localize(None).dt.floor('h') + pd.Timedelta(hours=1)


def add_is_chovendo(df_ocorrencias: pd.DataFrame, df_estacoes: pd.DataFrame, df_precipitacao: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona o atributo is_chovendo: 1 se a estação do INMET mais próxima do acidente registrou chuva (mais de 0 mm) na hora do acidente e 0 caso contrário.
    Se a estação mais próxima não tem medição naquela hora, usa-se a seguinte, entre as INMET_NEAREST_STATIONS mais próximas e a até INMET_MAX_DISTANCE_KM.
    O atributo fica nulo quando o acidente não tem coordenada ou horário válido, ou quando nenhuma dessas estações tem medição.
    """
    df = df_ocorrencias.copy()

    # 1. Coordenadas (texto com vírgula decimal) e horário do registro do INMET
    latitude = pd.to_numeric(df['oco_latitude_sirgas2000'].astype(str).str.replace(',', '.'), errors='coerce')
    longitude = pd.to_numeric(df['oco_longitude_sirgas2000'].astype(str).str.replace(',', '.'), errors='coerce')
    horario_registro = get_inmet_record_time(df)
    is_valido = latitude.notna() & longitude.notna() & horario_registro.isin(df_precipitacao.index)

    # 2. Estações mais próximas de cada acidente (distância sobre a superfície da Terra)
    arvore = BallTree(np.radians(df_estacoes[['latitude', 'longitude']].to_numpy()), metric='haversine')
    coordenadas = np.radians(np.column_stack([latitude[is_valido], longitude[is_valido]]))
    distancias, estacoes_proximas = arvore.query(coordenadas, k=INMET_NEAREST_STATIONS)
    distancias_km = distancias * EARTH_RADIUS_KM

    # 3. Chuva na hora do acidente: usa a estação mais próxima, dentro do raio, que tem medição naquela hora
    precipitacao = df_precipitacao[df_estacoes['codigo']].to_numpy()
    linhas = df_precipitacao.index.get_indexer(horario_registro[is_valido])
    chuva_mm = np.full(len(linhas), np.nan)
    for i in range(INMET_NEAREST_STATIONS):
        chuva_estacao = precipitacao[linhas, estacoes_proximas[:, i]]
        sem_valor = np.isnan(chuva_mm) & (distancias_km[:, i] <= INMET_MAX_DISTANCE_KM)
        chuva_mm[sem_valor] = chuva_estacao[sem_valor]

    # 4. Atributo final: nulo quando não foi possível obter a chuva
    chuva = pd.Series(np.nan, index=df.index)
    chuva[is_valido] = chuva_mm
    df['is_chovendo'] = (chuva > 0).astype('Int64').where(chuva.notna())

    return df
