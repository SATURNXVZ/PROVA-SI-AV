# QUESTÃO 1 - Heart Failure Clinical Records

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


print("QUESTÃO 1 - HEART FAILURE CLINICAL RECORDS (CORRIGIDO)")

#1- CARREGAMENTO DOS DADOS
df_heart = pd.read_csv(r"C:\Users\User\Documents\PROVA-SI-AV\heart_failure_clinical_records_dataset.csv")

print("\n1. CARREGAMENTO DOS DADOS")
print("-"*60)
print(f"Total de pacientes: {df_heart.shape[0]}")
print(f"Total de variaveis: {df_heart.shape[1]}")
print(f"\nPrimeiros registros:")
print(df_heart.head())

print(f"\nTipos de dados:")
print(df_heart.dtypes)

print(f"\nVerificacao de valores ausentes:")
print(df_heart.isnull().sum())


#2- ANALISE DAS VARIAVEIS

print("\n2. ANALISE DAS VARIAVEIS BINARIAS")
print("-"*60)

binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
for var in binary_vars:
    counts = df_heart[var].value_counts()
    print(f"\n{var}:")
    print(f"  0: {counts.get(0, 0)} ({counts.get(0, 0)/len(df_heart)*100:.1f}%)")
    print(f"  1: {counts.get(1, 0)} ({counts.get(1, 0)/len(df_heart)*100:.1f}%)")

print("\n3. ANALISE DAS VARIAVEIS CONTINUAS")
print("-"*60)

# ARRUMADO 'time' removida das variaveis analisadas (data leakage)
continuous_vars = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                   'platelets', 'serum_creatinine', 'serum_sodium']

for var in continuous_vars:
    print(f"\n{var}:")
    print(f"  Media: {df_heart[var].mean():.2f}")
    print(f"  Mediana: {df_heart[var].median():.2f}")
    print(f"  Desvio Padrao: {df_heart[var].std():.2f}")
    print(f"  Min: {df_heart[var].min():.2f}")
    print(f"  Max: {df_heart[var].max():.2f}")



# 3- PRE-PROCESSAMENTO

print("\n5. PRE-PROCESSAMENTO")
print("-"*60)

feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                'ejection_fraction', 'high_blood_pressure', 'platelets',
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
target_col = 'DEATH_EVENT'

X = df_heart[feature_cols]
y = df_heart[target_col]

binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
# sex: 0 = feminino, 1 = masculino

numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                    'platelets', 'serum_creatinine', 'serum_sodium']
# Standardsclar ficou para compatibilidade com possíveis modelos futuros,
# mas nao afeta a performance do randomforest

preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), numeric_features),
    ('binary', 'passthrough', binary_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} pacientes")
print(f"Teste: {X_test.shape[0]} pacientes")
print(f"Obitos no treino: {y_train.mean():.1%}")
print(f"Obitos no teste: {y_test.mean():.1%}")
print(f"\nFeatures utilizadas ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  - {f}")
print(f"\nFeature REMOVIDA por data leakage: time")


#4- TREINAMENTO

print("\n6. TREINAMENTO DO MODELO")

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ))
])

rf_model.fit(X_train, y_train)

#Validacao cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

cv_recall = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='recall')
print(f"Cross-validation Recall:   {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")


#5- AVALIACAO
print("\n7. AVALIACAO DO MODELO")
print("-"*60)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {acc:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Recall:    {rec:.4f}  <- foco: minimizar falsos negativos")
print(f"ROC-AUC:   {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusao:")
print(f"  Verdadeiros Negativos (sobreviveu, previsto sobreviveu): {cm[0,0]}")
print(f"  Falsos Positivos      (sobreviveu, previsto obito):      {cm[0,1]}")
print(f"  Falsos Negativos      (obito, previsto sobreviveu):      {cm[1,0]}  <- erro critico")
print(f"  Verdadeiros Positivos (obito, previsto obito):           {cm[1,1]}")

total_obitos_teste = cm[1,0] + cm[1,1]
print(f"\nDe {total_obitos_teste} obitos reais no teste, o modelo identificou {cm[1,1]} ({cm[1,1]/total_obitos_teste:.1%})")

#6-. IMPORTANCIA DAS FEATURES

print("\n8. IMPORTANCIA DAS VARIAVEIS")

importances = rf_model.named_steps['classifier'].feature_importances_
all_features = numeric_features + binary_features

sorted_idx = np.argsort(importances)[::-1]
print(f"{'Rank':<5} {'Feature':<30} {'Importancia':<12}")
print("-"*47)
for i, idx in enumerate(sorted_idx):
    print(f"{i+1:<5} {all_features[idx]:<30} {importances[idx]:.4f}")


# 7- SALVAR MODELO

import joblib
joblib.dump(rf_model, 'heart_failure_model.pkl')
print("\n10. MODELO SALVO")

print("Arquivo 'heart_failure_model.pkl' criado com sucesso!")
