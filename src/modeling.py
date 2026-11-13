from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, precision_score, recall_score, f1_score


def clean_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers de idade, descarta variáveis redundantes/multicolineares 
    e elimina colunas de texto de alta cardinalidade e coordenadas brutas.
    """
    df_clean: pd.DataFrame = df.copy()
    
    # Filtra erros de digitação na idade máxima das vítimas
    df_clean = df_clean[(df_clean['idade_maxima_vitima'] >= 0) & (df_clean['idade_maxima_vitima'] <= 110)]
    
    # Colunas textuais (identificadores, alta cardinalidade e coordenadas) 
    text_columns_to_remove: List[str] = [
        'numero_ocorrencia_associado',
        'oco_tipo_logradouro_descricao',
        'oco_natureza_descricao',
        'oco_nome_bairro',
        'oco_nome_municipio',
        'oco_latitude_sirgas2000',
        'oco_longitude_sirgas2000'
    ]
    
    # Colunas colineares/redundantes identificadas na matriz de correlação
    collinear_columns_to_remove: List[str] = [
        'idade_media_vitimas',
        'qtd_defeitos_via',
        'has_via_buracos',
        'has_via_ondulada',
        'has_via_rachadura',
        'has_via_sulcos_horizontais',
        'has_material_na_pista'
    ]
    
    all_columns_to_drop: List[str] = text_columns_to_remove + collinear_columns_to_remove
    
    existing_columns_to_drop: List[str] = [col for col in all_columns_to_drop if col in df_clean.columns]
    
    return df_clean.drop(columns=existing_columns_to_drop)

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte variáveis categóricas (strings) em representações numéricas binárias (One-Hot Encoding).
    """
    df_encoded: pd.DataFrame = df.copy()
    
    # Identifica colunas que são do tipo string/object
    categorical_columns = df_encoded.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Aplica o get_dummies convertendo os booleanos para inteiros (0 e 1)
    # drop_first=True evita a armadilha das variáveis dummy (multicolinearidade perfeita)
    if categorical_columns:
        df_encoded = pd.get_dummies(
            df_encoded, 
            columns=categorical_columns, 
            drop_first=True, 
            dtype=int
        )
        
    return df_encoded

def prepare_train_test_data(
    df: pd.DataFrame, 
    target_column: str = 'acidente_grave', 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza a divisão estratificada dos dados entre os conjuntos de treino e teste.
    """
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.Series = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def train_random_forest_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Treina o modelo baseline Random Forest aplicando peso balanceado para tratar o desbalanceamento de classes.
    """
    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_severity_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Exibe o relatório de classificação, matriz de confusão e curva ROC focando na previsão de acidentes graves.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("=== Relatório de Desempenho ===")
    print(classification_report(y_test, y_pred, target_names=['Leve/Moderado', 'Grave']))
    
    auc: float = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matriz de Confusão
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Reds',
        ax=ax[0],
        xticklabels=['Previsto Leve', 'Previsto Grave'],
        yticklabels=['Real Leve', 'Real Grave']
    )
    ax[0].set_title('Matriz de Confusão')
    
    # Curva ROC
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
    ax[1].set_title(f'Curva ROC (AUC = {auc:.4f})')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_decision_thresholds(
    model: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    thresholds: List[float] = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
) -> pd.DataFrame:
    """
    Avalia o impacto de diferentes limiares de decisão nas métricas operacionais e de negócio 
    para a classe de acidentes graves (1).
    """
    # Extrai as probabilidades da classe positiva (acidente grave)
    y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]
    threshold_results: List[Dict[str, Any]] = []

    for threshold in thresholds:
        # Aplica o ponto de corte customizado
        y_pred_custom: np.ndarray = (y_proba >= threshold).astype(int)
        
        # Extrai os componentes da matriz de confusão
        cm: np.ndarray = confusion_matrix(y_test, y_pred_custom)
        tn, fp, fn, tp = cm.ravel()
        
        precision: float = precision_score(y_test, y_pred_custom, pos_label=1, zero_division=0)
        recall: float = recall_score(y_test, y_pred_custom, pos_label=1, zero_division=0)
        f1: float = f1_score(y_test, y_pred_custom, pos_label=1, zero_division=0)
        
        threshold_results.append({
            'Limiar (Threshold)': threshold,
            'Verdadeiros Graves (TP)': tp,
            'Falsos Leves (FN)': fn,
            'Falsos Alertas (FP)': fp,
            'Precisão (Grave)': round(precision, 4),
            'Recall (Grave)': round(recall, 4),
            'F1-Score (Grave)': round(f1, 4)
        })

    metrics_df: pd.DataFrame = pd.DataFrame(threshold_results)
    return metrics_df