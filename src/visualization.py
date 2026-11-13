import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay

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

def plot_taxa_gravidade_hexagonal(df, target='acidente_grave', min_acidentes=30):
    """Mapa hexagonal da TAXA de acidentes graves por célula (média do target), em vez da contagem absoluta."""
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

    plt.figure(figsize=(10, 8))
    plt.hexbin(
        x=df_geo['lon'],
        y=df_geo['lat'],
        C=df_geo[target],
        reduce_C_function=np.mean,
        gridsize=100,
        cmap='inferno',
        mincnt=min_acidentes
    )
    plt.colorbar(label='Proporção de Acidentes Graves')
    plt.title(f'Taxa de Acidentes Graves por Região (células com ao menos {min_acidentes} acidentes)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Cores dos modelos (sempre na mesma ordem, para que cada modelo mantenha a mesma cor entre os gráficos)
MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

def plot_scores_atributos(scores, titulo, xlabel):
    """Barras horizontais com o score de cada atributo (ex.: correlação de Pearson ou F-score), do maior para o menor."""
    scores = scores.sort_values()
    has_negative = (scores < 0).any()
    colors = [COLORS[1] if value >= 0 else COLORS[0] for value in scores]

    plt.figure(figsize=(10, 0.3 * len(scores) + 1.5))
    plt.barh(scores.index, scores.values, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.grid(axis='x', alpha=0.3)

    if has_negative:
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[1]), plt.Rectangle((0, 0), 1, 1, color=COLORS[0])]
        plt.legend(handles, ['Positiva', 'Negativa'], title='Relação com o alvo', loc='lower right')

    plt.tight_layout()
    plt.show()

def plot_desempenho_por_k(resultados_por_modelo):
    """Curva da PR-AUC (média ± desvio entre os folds) em função do número k de atributos selecionados, para cada modelo."""
    plt.figure(figsize=(12, 5))
    for i, (modelo, resultados) in enumerate(resultados_por_modelo.items()):
        color = MODEL_COLORS[i]
        plt.plot(resultados['k'], resultados['pr_auc_media'], marker='o', markersize=4, linewidth=2, color=color, label=modelo)
        plt.fill_between(
            resultados['k'],
            resultados['pr_auc_media'] - resultados['pr_auc_desvio'],
            resultados['pr_auc_media'] + resultados['pr_auc_desvio'],
            color=color,
            alpha=0.2
        )

    plt.title('Desempenho por Número de Atributos Selecionados (k maiores F-scores)')
    plt.xlabel('k (número de atributos)')
    plt.ylabel('PR-AUC (validação cruzada)')
    plt.xticks(resultados['k'])
    plt.grid(alpha=0.3)
    plt.legend(title='Modelo')
    plt.tight_layout()
    plt.show()

def plot_comparacao_modelos(resultados, metrica='pr_auc_media', desvio='pr_auc_desvio', titulo='Comparação dos Modelos', xlabel='PR-AUC (média entre os folds)'):
    """Barras horizontais com uma métrica de cada modelo e, se informado, o desvio entre os folds como barra de erro."""
    resultados = resultados.sort_values(metrica)
    erro = resultados[desvio] if desvio in resultados.columns else pd.Series(0, index=resultados.index)

    plt.figure(figsize=(10, 0.5 * len(resultados) + 1.5))
    plt.barh(resultados.index, resultados[metrica], xerr=erro, color=COLORS[0], capsize=4)
    for y_pos, (value, std) in enumerate(zip(resultados[metrica], erro)):
        plt.text(value + std, y_pos, f'  {value:.3f}', va='center', ha='left', fontsize=9)

    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.xlim(0, (resultados[metrica] + erro).max() * 1.2)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_curvas_precisao_recall(y_true, scores_por_modelo, titulo='Curvas de Precisão e Recall no Conjunto de Teste'):
    """Curvas de precisão x recall de cada modelo, com a PR-AUC na legenda e a linha de referência da taxa de acidentes graves."""
    plt.figure(figsize=(9, 7))
    for i, (modelo, scores) in enumerate(scores_por_modelo.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, linewidth=2, color=MODEL_COLORS[i], label=f'{modelo} (PR-AUC = {ap:.3f})')

    taxa_graves = np.mean(y_true)
    plt.axhline(taxa_graves, color='gray', linestyle='--', linewidth=1.5, label=f'Taxa de graves ({taxa_graves:.1%})')

    plt.title(titulo)
    plt.xlabel('Recall (acidentes graves identificados)')
    plt.ylabel('Precisão')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_matriz_confusao(y_true, y_pred, titulo='Matriz de Confusão'):
    """Matriz de confusão com as quantidades de cada célula."""
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=['Não Grave (0)', 'Grave (1)'],
        cmap='Blues',
        values_format='d'
    )
    plt.title(titulo)
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe real')
    plt.tight_layout()
    plt.show()

def plot_comparacao_recortes(resultados, metrica='valor', titulo='PR-AUC no Teste (2025) por Recorte de Treino', ylabel='PR-AUC'):
    """
    Barras agrupadas por modelo com a métrica de cada recorte de treino e o IC de 95% (bootstrap) como barra de erro.
    Espera um DataFrame com as colunas: 'modelo', 'recorte', a métrica, 'ic_inferior' e 'ic_superior'.
    """
    modelos = list(dict.fromkeys(resultados['modelo']))
    recortes = list(dict.fromkeys(resultados['recorte']))
    largura = 0.8 / len(recortes)
    posicoes = np.arange(len(modelos))

    plt.figure(figsize=(10, 5))
    for i, recorte in enumerate(recortes):
        dados = resultados[resultados['recorte'] == recorte].set_index('modelo').loc[modelos]
        erro = [dados[metrica] - dados['ic_inferior'], dados['ic_superior'] - dados[metrica]]
        plt.bar(posicoes + i * largura, dados[metrica], width=largura, yerr=erro, capsize=4, color=MODEL_COLORS[i], edgecolor='white', linewidth=1.5, label=recorte)

    plt.xticks(posicoes + largura * (len(recortes) - 1) / 2, modelos)
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Recorte de treino', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(recortes))
    plt.tight_layout()
    plt.show()
