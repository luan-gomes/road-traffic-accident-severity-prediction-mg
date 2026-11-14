import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, cross_val_predict
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from src.constants import (
    OCCURRENCE_ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
)

def split_train_test_by_year(df: pd.DataFrame, train_years: tuple[int], test_year: int, year_column: str = 'oco_ano_fato') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a base em treino (anos de treino) e teste (ano de teste), garantindo que nenhum acidente esteja nos dois conjuntos."""
    df_train = df[df[year_column].isin(train_years)].copy()
    df_test = df[df[year_column] == test_year].copy()

    ids_em_comum = set(df_train[OCCURRENCE_ID_COLUMN]) & set(df_test[OCCURRENCE_ID_COLUMN])
    assert not ids_em_comum, f"{len(ids_em_comum)} ocorrências aparecem no treino e no teste."

    print(f"Treino ({min(train_years)}-{max(train_years)}): {len(df_train)} instâncias | {df_train[TARGET_COLUMN].mean():.2%} de acidentes graves")
    print(f"Teste ({test_year}): {len(df_test)} instâncias | {df_test[TARGET_COLUMN].mean():.2%} de acidentes graves")
    return df_train, df_test

def get_features_and_target(df: pd.DataFrame, features: list[str], target: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna a matriz de atributos (float64, com NaN nos valores faltantes) e o vetor da variável alvo."""
    X = df[list(features)].astype('float64')
    y = df[target].astype(int)
    return X, y

def compute_features_variance(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a variância de cada atributo.
    - Numéricos (não binários): variância e coeficiente de variação (desvio padrão / média), que não depende da unidade.
    - Binários: variância (igual a p * (1 - p)) e o percentual de instâncias com valor 1.
    """
    rows = []
    for col in X.columns:
        values = X[col].dropna()
        is_binary = set(values.unique()) <= {0.0, 1.0}
        rows.append({
            'atributo': col,
            'tipo': 'binário' if is_binary else 'numérico',
            'variancia': values.var(),
            'coef_variacao': np.nan if is_binary else values.std() / values.mean(),
            'percentual_valor_1': values.mean() * 100 if is_binary else np.nan,
        })

    return pd.DataFrame(rows).sort_values(['tipo', 'variancia'], ascending=[False, True]).reset_index(drop=True)

def compute_feature_relevance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcula, para cada atributo, a correlação de Pearson com a variável alvo e o F-score da ANOVA (f_classif), com o p-valor.
    Os valores faltantes são preenchidos com a mediana do próprio conjunto (o de treino), pois o f_classif não aceita NaN.
    """
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    f_scores, p_values = f_classif(X_imputed, y)
    pearson = X_imputed.corrwith(y)

    relevance = pd.DataFrame({
        'pearson': pearson.values,
        'abs_pearson': pearson.abs().values,
        'f_score': f_scores,
        'p_valor': p_values,
    }, index=pd.Index(X.columns, name='atributo'))

    return relevance.sort_values('f_score', ascending=False)

def find_high_correlation_pairs(X: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Retorna os pares de atributos com correlação de Pearson, em módulo, maior ou igual ao limite informado."""
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    pairs = upper.stack()
    pairs.index.names = ['atributo_1', 'atributo_2']
    pairs = pairs.rename('correlacao').reset_index()
    pairs = pairs[pairs['correlacao'].abs() >= threshold]

    return pairs.sort_values('correlacao', key=abs, ascending=False).reset_index(drop=True)

def build_model_pipeline(estimator, scale: bool = False) -> Pipeline:
    """
    Monta o pipeline de um modelo: imputação pela mediana -> seleção dos k atributos de maior F-score -> padronização (opcional) -> modelo.
    Por padrão, k='all' (todos os atributos); o k só é alterado na análise das combinações (evaluate_k_best_features).
    Como tudo fica dentro do pipeline, a mediana, o ranking de atributos e a padronização são calculados apenas com os dados de treino de cada fold.
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
    ]
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', estimator))

    return Pipeline(steps)

def evaluate_k_best_features(estimator, X: pd.DataFrame, y: pd.Series, cv, k_values: list[int] | None = None, scale: bool = False, n_jobs: int = 1) -> pd.DataFrame:
    """
    Avalia, por validação cruzada, o desempenho (PR-AUC) do modelo usando apenas os k atributos de maior F-score, para cada valor de k.
    A coluna 'atributo_adicionado' indica qual atributo entra ao passar de k-1 para k (ranking calculado em todo o conjunto informado).
    """
    if k_values is None:
        k_values = list(range(1, X.shape[1] + 1))

    search = GridSearchCV(
        build_model_pipeline(estimator, scale),
        param_grid={'selector__k': list(k_values)},
        scoring='average_precision',
        cv=cv,
        n_jobs=n_jobs,
    )
    search.fit(X, y)

    ranking = compute_feature_relevance(X, y).index.tolist()
    return pd.DataFrame({
        'k': k_values,
        'atributo_adicionado': [ranking[k - 1] for k in k_values],
        'pr_auc_media': search.cv_results_['mean_test_score'],
        'pr_auc_desvio': search.cv_results_['std_test_score'],
    })

def get_linear_models(random_state: int = RANDOM_STATE) -> dict[str, dict]:
    """
    Retorna o baseline e os modelos lineares. Cada modelo possui:
    - estimator: o modelo do scikit-learn;
    - params: o espaço de busca de hiperparâmetros;
    - scale: se os atributos devem ser padronizados;
    - n_jobs: quantos ajustes da validação cruzada rodam em paralelo (-1 = todos os núcleos).
    """
    return {
        'Dummy (baseline)': {
            'estimator': DummyClassifier(strategy='prior'),
            'params': {},
            'scale': False,
            'n_jobs': -1,
        },
        'Regressão Logística': {
            'estimator': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
            'params': {'model__C': [0.001, 0.01, 0.1, 1, 10]},
            'scale': True,
            'n_jobs': -1,
        },
        'LDA': {
            'estimator': LinearDiscriminantAnalysis(),
            'params': {},
            'scale': True,
            'n_jobs': -1,
        },
        'Ridge': {
            'estimator': RidgeClassifier(class_weight='balanced'),
            'params': {'model__alpha': [0.1, 1, 10, 100, 1000]},
            'scale': True,
            'n_jobs': -1,
        },
        'LinearSVC': {
            'estimator': LinearSVC(class_weight='balanced', dual=False, random_state=random_state),
            'params': {'model__C': [0.001, 0.01, 0.1, 1]},
            'scale': True,
            'n_jobs': -1,
        },
        'SGD': {
            'estimator': SGDClassifier(class_weight='balanced', random_state=random_state),
            'params': {'model__loss': ['hinge', 'log_loss'], 'model__alpha': [0.00001, 0.0001, 0.001, 0.01]},
            'scale': True,
            'n_jobs': -1,
        },
        'Perceptron': {
            'estimator': Perceptron(class_weight='balanced', random_state=random_state),
            'params': {'model__penalty': [None, 'l2', 'l1', 'elasticnet']},
            'scale': True,
            'n_jobs': -1,
        },
        'Nearest Centroid': {
            'estimator': NearestCentroid(),
            'params': {'model__shrink_threshold': [None, 0.1, 0.5, 1.0]},
            'scale': True,
            'n_jobs': -1,
        },
        'Naive Bayes (Bernoulli)': {
            'estimator': BernoulliNB(),
            'params': {'model__alpha': [0.01, 0.1, 1.0]},
            'scale': True,
            'n_jobs': -1,
        },
        'Naive Bayes (Multinomial)': {
            'estimator': MultinomialNB(),
            'params': {'model__alpha': [0.01, 0.1, 1.0]},
            'scale': False,
            'n_jobs': -1,
        },
        'Naive Bayes (Complement)': {
            'estimator': ComplementNB(),
            'params': {'model__alpha': [0.01, 0.1, 1.0]},
            'scale': False,
            'n_jobs': -1,
        },
    }

def get_nonlinear_models(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> dict[str, dict]:
    """
    Retorna os modelos não lineares (baseados em árvores), com o espaço de busca de hiperparâmetros de cada um.
    - scale_pos_weight: razão entre negativos e positivos no treino; no XGBoost, tem o mesmo efeito do class_weight='balanced'.
    A Random Forest usa n_jobs=1 na busca porque já paraleliza as árvores internamente, e várias florestas ao mesmo tempo ocupariam muita memória.
    """

    tree_params = {
        'model__max_depth': [5, 10, 15, 20, None],
        'model__min_samples_leaf': [10, 50, 100, 500],
    }

    return {
        'HistGradientBoosting': {
            'estimator': HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state),
            'params': {
                'model__learning_rate': [0.05, 0.1],
                'model__max_iter': [200, 400],
                'model__max_leaf_nodes': [31, 63, 127],
                'model__min_samples_leaf': [20, 100],
                'model__l2_regularization': [0.0, 1.0],
            },
            'scale': False,
            'n_jobs': -1,
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state),
            'params': {
                'model__max_depth': [10, 15, 20, None],
                'model__min_samples_leaf': [10, 50, 100],
                'model__max_features': ['sqrt', 0.5],
            },
            'scale': False,
            'n_jobs': 1,
        },
        'XGBoost': {
            'estimator': XGBClassifier(tree_method='hist', scale_pos_weight=scale_pos_weight, eval_metric='aucpr', n_jobs=-1, random_state=random_state),
            'params': {
                'model__n_estimators': [200, 400, 800],
                'model__learning_rate': [0.03, 0.1],
                'model__max_depth': [4, 6, 8],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0],
                'model__min_child_weight': [1, 10],
            },
            'scale': False,
            'n_jobs': -1,
        },
         'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=random_state),
            'params': {
                'model__n_estimators': [100, 200, 400],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 1.0],
                'model__min_samples_leaf': [1, 50],
            },
            'scale': False,
            'n_jobs': -1,
        },
        'AdaBoost': {
            'estimator': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=random_state),
            'params': {
                'model__n_estimators': [100, 200, 400],
                'model__learning_rate': [0.1, 0.5, 1.0],
                'model__estimator__max_depth': [1, 2, 3],
            },
            'scale': False,
            'n_jobs': -1,
        },
        'Extra Trees': {
            'estimator': ExtraTreesClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state),
            'params': {
                'model__max_depth': [10, 15, 20, None],
                'model__min_samples_leaf': [10, 50, 100],
                'model__max_features': ['sqrt', 0.5],
            },
            'scale': False,
            'n_jobs': 1,
        },
        'Bagging': {
            'estimator': BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'), n_estimators=100, n_jobs=-1, random_state=random_state),
            'params': {
                'model__max_samples': [0.5, 1.0],
                'model__estimator__max_depth': [15, None],
                'model__estimator__min_samples_leaf': [10, 50, 100],
            },
            'scale': False,
            'n_jobs': 1,
        },
        'Árvore de Decisão': {
            'estimator': DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
            'params': tree_params,
            'scale': False,
            'n_jobs': -1,
        },
        'Extra Tree': {
            'estimator': ExtraTreeClassifier(class_weight='balanced', random_state=random_state),
            'params': tree_params,
            'scale': False,
            'n_jobs': -1,
        },
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'params': {'model__n_neighbors': [5, 25, 50, 100, 200], 'model__weights': ['uniform', 'distance']},
            'scale': True,
            'n_jobs': 1,
        },
        'Radius Neighbors': {
            'estimator': RadiusNeighborsClassifier(outlier_label='most_frequent'),
            'params': {'model__radius': [2.0, 2.5, 3.0], 'model__weights': ['uniform', 'distance']},
            'scale': True,
            'n_jobs': 1,
        },
        'MLP': {
            'estimator': MLPClassifier(random_state=random_state),
            'params': {'model__hidden_layer_sizes': [(50,), (100,), (200,)], 'model__alpha': [0.0001, 0.001, 0.01]},
            'scale': True,
            'n_jobs': -1,
        },
        'QDA': {
            'estimator': QuadraticDiscriminantAnalysis(),
            'params': {'model__reg_param': [0.01, 0.1, 0.5, 0.9]},
            'scale': True,
            'n_jobs': -1,
        },
        'Naive Bayes (Gaussiano)': {
            'estimator': GaussianNB(),
            'params': {'model__var_smoothing': [1e-9, 1e-3, 1e-2, 1e-1]},
            'scale': True,
            'n_jobs': -1,
        },
    }

def tune_models(models: dict[str, dict], X: pd.DataFrame, y: pd.Series, cv, n_iter: int = 10, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict]:
    """
    Ajusta os hiperparâmetros de cada modelo por validação cruzada (mesmos folds para todos) e retorna:
    - uma tabela com a PR-AUC e a ROC-AUC (média e desvio entre os folds) da melhor combinação de cada modelo;
    - um dicionário com o melhor pipeline de cada modelo, já reajustado em todo o conjunto de treino.
    Se o espaço de busca tiver até n_iter combinações, todas são testadas (GridSearchCV); senão, n_iter são sorteadas (RandomizedSearchCV).
    """
    scoring = {'pr_auc': 'average_precision', 'roc_auc': 'roc_auc'}
    rows = []
    best_estimators = {}

    for name, spec in models.items():
        pipeline = build_model_pipeline(spec['estimator'], spec['scale'])
        n_candidates = len(ParameterGrid(spec['params']))

        if n_candidates <= n_iter:
            search = GridSearchCV(pipeline, spec['params'], scoring=scoring, refit='pr_auc', cv=cv, n_jobs=spec['n_jobs'])
        else:
            search = RandomizedSearchCV(pipeline, spec['params'], n_iter=n_iter, scoring=scoring, refit='pr_auc', cv=cv, n_jobs=spec['n_jobs'], random_state=random_state)

        start = time.time()
        search.fit(X, y)
        elapsed_minutes = (time.time() - start) / 60

        best = search.best_index_
        results = search.cv_results_
        rows.append({
            'modelo': name,
            'pr_auc_media': results['mean_test_pr_auc'][best],
            'pr_auc_desvio': results['std_test_pr_auc'][best],
            'roc_auc_media': results['mean_test_roc_auc'][best],
            'roc_auc_desvio': results['std_test_roc_auc'][best],
            'melhores_parametros': search.best_params_,
            'combinacoes_testadas': len(results['params']),
            'tempo_min': round(elapsed_minutes, 1),
        })
        best_estimators[name] = search.best_estimator_
        print(f"[{name}] PR-AUC = {results['mean_test_pr_auc'][best]:.4f} ± {results['std_test_pr_auc'][best]:.4f} ({elapsed_minutes:.1f} min)")

    return pd.DataFrame(rows).set_index('modelo').sort_values('pr_auc_media', ascending=False), best_estimators

def get_model_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    """Retorna o score da classe positiva: a probabilidade (predict_proba) ou, se o modelo não a fornecer, a função de decisão."""
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)

def find_best_threshold(estimator, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int = 1) -> tuple[float, float]:
    """
    Escolhe o limiar de decisão que maximiza o F1 da classe grave nas previsões fora do fold (out-of-fold) do conjunto de treino.
    Retorna o limiar e o F1 obtido com ele. O conjunto de teste não é usado.
    """
    method = 'predict_proba' if hasattr(estimator, 'predict_proba') else 'decision_function'
    oof_predictions = cross_val_predict(estimator, X, y, cv=cv, method=method, n_jobs=n_jobs)
    scores = oof_predictions[:, 1] if method == 'predict_proba' else oof_predictions

    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)

    # O último par (precisão, recall) da curva não possui limiar associado
    best = np.argmax(f1[:-1])
    return thresholds[best], f1[best]

def evaluate_models_on_test(estimators: dict, thresholds: dict[str, float], X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Avalia cada modelo no conjunto de teste e retorna:
    - uma tabela com as métricas que não dependem de limiar (PR-AUC, ROC-AUC) e as calculadas no limiar escolhido no treino;
    - um dicionário com os scores de cada modelo no teste (usados nas curvas e no bootstrap).
    """
    rows = []
    scores_per_model = {}

    for name, estimator in estimators.items():
        scores = get_model_scores(estimator, X_test)
        y_pred = (scores >= thresholds[name]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        rows.append({
            'modelo': name,
            'pr_auc': average_precision_score(y_test, scores),
            'roc_auc': roc_auc_score(y_test, scores),
            'limiar': thresholds[name],
            'precisao': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'verdadeiros_positivos': tp,
            'falsos_positivos': fp,
            'falsos_negativos': fn,
            'verdadeiros_negativos': tn,
        })
        scores_per_model[name] = scores

    return pd.DataFrame(rows).set_index('modelo').sort_values('pr_auc', ascending=False), scores_per_model

def bootstrap_metric_ci(y_true: pd.Series, scores: np.ndarray, metric=average_precision_score, n_boot: int = 1000, random_state: int = RANDOM_STATE) -> dict[str, float]:
    """Calcula uma métrica no teste e o seu intervalo de confiança de 95% por bootstrap (reamostragem das instâncias com reposição)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(random_state)

    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(metric(y_true[idx], scores[idx]))

    return {
        'valor': metric(y_true, scores),
        'ic_inferior': np.percentile(values, 2.5),
        'ic_superior': np.percentile(values, 97.5),
    }

def bootstrap_metric_difference(y_true: pd.Series, scores_a: np.ndarray, scores_b: np.ndarray, metric=average_precision_score, n_boot: int = 1000, random_state: int = RANDOM_STATE) -> dict[str, float]:
    """
    Calcula a diferença (A - B) de uma métrica entre dois modelos avaliados nas mesmas instâncias e o IC de 95% por bootstrap pareado
    (as mesmas instâncias sorteadas são usadas para os dois modelos). Se o intervalo não contém 0, a diferença não é atribuível ao acaso da amostra de teste.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    rng = np.random.default_rng(random_state)

    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        differences.append(metric(y_true[idx], scores_a[idx]) - metric(y_true[idx], scores_b[idx]))

    return {
        'diferenca': metric(y_true, scores_a) - metric(y_true, scores_b),
        'ic_inferior': np.percentile(differences, 2.5),
        'ic_superior': np.percentile(differences, 97.5),
    }
