#QUESTAO 2 - Wine Qualit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df_red = pd.read_csv(r"C:\Users\User\Documents\PROVA-SI-AV\winequality-red.csv", sep=';')
df_white = pd.read_csv(r"C:\Users\User\Documents\PROVA-SI-AV\winequality-white.csv", sep=';')
#print("Dados carregados")

df_red['color'] = 'red'
df_white['color'] = 'white'

df_wine = pd.concat([df_red, df_white], ignore_index=True)

print(f"Dataset combinado - Tintos: {len(df_red)} | Brancos: {len(df_white)} | Total: {len(df_wine)}")

#distribuicao da classe alvo
print("\nDistribuicao da qualidade:")
for score, count in df_wine['quality'].value_counts().sort_index().items():
    print(f"  Score {score}: {count} vinhos")

#pre-processamento
df_wine['color_encoded'] = df_wine['color'].map({'red': 0, 'white': 1})

feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol', 'color_encoded']

X = df_wine[feature_cols]
y = df_wine['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# avaliacao de tres metaestimadores
print("\nComparacao de metaestimadores:")

#usados em aula
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=42)
}

resultados = {}

for nome, modelo in modelos.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', modelo)
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    resultados[nome] = {
        'pipeline': pipeline,
        'cv_media': cv_scores.mean(),
        'cv_desvio': cv_scores.std(),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"  {nome:<15} CV F1: {cv_scores.mean():.4f}  Test F1: {resultados[nome]['f1_weighted']:.4f}")

#selecionar melhor
melhor = max(resultados, key=lambda x: resultados[x]['f1_weighted'])
print(f"\nMelhor modelo: {melhor}")

#otimizacao do melhor modelo
print("\nOtimizacao com GridSearch:")

if melhor == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 15, None],
        'classifier__min_samples_split': [2, 5]
    }
    modelo_base = RandomForestClassifier(random_state=42)
    
elif melhor == 'KNN':
    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance']
    }
    modelo_base = KNeighborsClassifier()
    
else:  # SVM
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    }
    modelo_base = SVC(probability=True, random_state=42)


pipeline_opt = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', modelo_base)
])

grid = GridSearchCV(pipeline_opt, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)

modelo_final = grid.best_estimator_
print(f"Melhores parametros: {grid.best_params_}")
print(f"F1 CV otimizado: {grid.best_score_:.4f}")

#avaliacao final
y_pred_final = modelo_final.predict(X_test)

print(f"\nResultados finais no teste:")
print(f"Accuracy global: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1-Score weighted: {f1_score(y_test, y_pred_final, average='weighted'):.4f}")

# matriz de confusao
print("\nMatriz de Confusao:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# relatorio completo
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred_final, digits=4))

# acuracia por classe
print("Acuracia por classe:")
classes = sorted(y.unique())
print(f"{'Score':<8} {'Amostras':<10} {'Acuracia':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
for i, score in enumerate(classes):
    n = cm[i].sum()
    if n > 0:
        acc = cm[i, i] / n
        prec = precision_score(y_test, y_pred_final, labels=[score], average=None)[0]
        rec = recall_score(y_test, y_pred_final, labels=[score], average=None)[0]
        f1 = f1_score(y_test, y_pred_final, labels=[score], average=None)[0]
        print(f"{score:<8} {n:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

# justificativa
justificativas = {
    'Random Forest': [
        "Maior F1-Score Weighted entre os tres modelos testados",
        "Nao e sensivel a escala das features (normalizacao e redundante, mas inofensiva)",
        "Ensemble de arvores: robusto a outliers em variaveis quimicas",
        "Captura relacoes nao-lineares entre acidez, alcool e qualidade",
        "Feature importances disponiveis para auditoria do modelo",
        "Inferencia rapida e paralelizavel para producao",
    ],
    'KNN': [
        "Maior F1-Score Weighted entre os tres modelos testados",
        "Eficaz quando exemplos similares quimicamente tem qualidade similar",
        "Beneficia-se do StandardScaler: distancias calculadas em escala uniforme",
        "Simples e interpretavel: classificacao por vizinhanca",
        "Sem fase de treino lenta; atualizacao com novos dados e trivial",
        "Limitacao: inferencia mais lenta em datasets grandes",
    ],
    'SVM': [
        "Maior F1-Score Weighted entre os tres modelos testados",
        "StandardScaler essencial: SVM e muito sensivel a escala das features",
        "Kernel RBF captura relacoes nao-lineares entre compostos quimicos",
        "Margem maxima de separacao: bom desempenho em espacos de alta dimensao",
        "Robusto ao desbalanceamento de classes com ajuste de C",
        "Limitacao: tempo de treino alto em datasets grandes",
    ],
}

print(f"\nJustificativa para implantacao ({melhor}):")
for i, ponto in enumerate(justificativas[melhor], 1):
    print(f"  {i}. {ponto}")

# modulo de inferencia
print("\nModulo de inferencia:")

def prever_qualidade(modelo, dados_vinho, nome="Vinho"):
    df = pd.DataFrame([dados_vinho])
    if 'color_encoded' not in df.columns and 'color' in df.columns:
        df['color_encoded'] = df['color'].map({'red': 0, 'white': 1})
    df = df[feature_cols]
    
    pred = modelo.predict(df)[0]
    proba = modelo.predict_proba(df)[0]
    classes_modelo = modelo.named_steps['classifier'].classes_
    
    top3 = np.argsort(proba)[::-1][:3]
    
    print(f"\n{nome}:")
    print(f"  Score predito: {pred}/10 (confianca: {max(proba):.1%})")
    print(f"  Top 3:")
    for idx in top3:
        print(f"    Score {classes_modelo[idx]}: {proba[idx]:.1%}")
    
    return pred

# testes
vinho1 = {
    'fixed acidity': 8.5, 'volatile acidity': 0.5, 'citric acid': 0.3,
    'residual sugar': 2.5, 'chlorides': 0.08, 'free sulfur dioxide': 15,
    'total sulfur dioxide': 120, 'density': 0.997, 'pH': 3.3,
    'sulphates': 0.8, 'alcohol': 12.5, 'color_encoded': 0
}

vinho2 = {
    'fixed acidity': 6.5, 'volatile acidity': 0.25, 'citric acid': 0.35,
    'residual sugar': 8.0, 'chlorides': 0.04, 'free sulfur dioxide': 30,
    'total sulfur dioxide': 150, 'density': 0.994, 'pH': 3.2,
    'sulphates': 0.45, 'alcohol': 10.5, 'color_encoded': 1
}

vinho3 = {
    'fixed acidity': 7.2, 'volatile acidity': 0.28, 'citric acid': 0.5,
    'residual sugar': 1.8, 'chlorides': 0.045, 'free sulfur dioxide': 25,
    'total sulfur dioxide': 110, 'density': 0.995, 'pH': 3.4,
    'sulphates': 0.95, 'alcohol': 13.5, 'color_encoded': 0
}

prever_qualidade(modelo_final, vinho1, "Vinho Tinto Encorpado")
prever_qualidade(modelo_final, vinho2, "Vinho Branco Suave")
prever_qualidade(modelo_final, vinho3, "Vinho Tinto Reserva")

import joblib
joblib.dump(modelo_final, 'wine_quality_model.pkl')
print("\nModelo salvo: wine_quality_model.pkl")