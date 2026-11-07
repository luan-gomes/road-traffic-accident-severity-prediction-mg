import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['#1f77b4', '#d62728']

def plot_proporcao_gravidade(df, coluna_feature, target='acidente_grave'):
    # Calcula a proporção
    props = df.groupby(coluna_feature)[target].value_counts(normalize=True).unstack() * 100
    
    # Plota o gráfico de barras empilhadas
    props.plot(kind='bar', stacked=True, color=COLORS, figsize=(8, 5))
    
    plt.title(f'Proporção de Acidentes Graves por {coluna_feature}')
    plt.ylabel('Percentual (%)')
    plt.xlabel(coluna_feature)
    plt.legend(['Não Grave (0)', 'Grave (1)'], title='Gravidade')
    plt.xticks(rotation=0)
    
    # Adiciona os rótulos de porcentagem dentro das barras
    for n, x in enumerate([*props.index.values]):
        for (proportion, y_loc) in zip(props.loc[x], props.loc[x].cumsum()):
            plt.text(x=n, y=(y_loc - proportion) + (proportion / 2), 
                     s=f'{proportion:.1f}%', color="white", fontsize=10, fontweight="bold",
                     ha="center", va="center")
            
    plt.show()


def plot_densidade_numerica(df, coluna_numerica, target='acidente_grave'):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(
        data=df, 
        x=coluna_numerica, 
        hue=target, 
        fill=True, 
        common_norm=False, 
        palette=COLORS,
        alpha=0.5
    )
    plt.title(f'Distribuição de {coluna_numerica} por Gravidade do Acidente')
    plt.xlabel(coluna_numerica)
    plt.ylabel('Densidade')
    plt.show()

def plot_matriz_correlacao(df):
    # Seleciona apenas as colunas numéricas (int, float)
    df_numerico = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])
    
    plt.figure(figsize=(16, 10))
    corr = df_numerico.corr(method='spearman') # Spearman é melhor para variáveis binárias/ordinais
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        annot=False,
        fmt=".2f", 
        linewidths=.5
    )
    plt.title('Matriz de Correlação das Variáveis')
    plt.show()

def plot_mapa_calor_hexagonal(df):
    df_geo = df.dropna(subset=['oco_latitude_sirgas2000', 'oco_longitude_sirgas2000']).copy()
    
    df_geo['lat'] = pd.to_numeric(
        df_geo['oco_latitude_sirgas2000'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    df_geo['lon'] = pd.to_numeric(
        df_geo['oco_longitude_sirgas2000'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    
    # Remove linhas onde a conversão falhou
    df_geo = df_geo.dropna(subset=['lat', 'lon'])
    
    # Filtra apenas os acidentes graves (hotspots)
    df_graves = df_geo[df_geo['acidente_grave'] == 1]
    
    plt.figure(figsize=(10, 8))
    plt.hexbin(
        x=df_graves['lon'], 
        y=df_graves['lat'], 
        gridsize=100, 
        cmap='inferno', 
        mincnt=1
    )
    plt.colorbar(label='Contagem de Acidentes Graves')
    plt.title('Concentração Espacial de Acidentes Graves (Hotspots)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()