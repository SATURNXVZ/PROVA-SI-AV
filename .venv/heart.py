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

print("="*80)
print("QUESTÃO 1 - HEART FAILURE CLINICAL RECORDS")
print("Analise de predicao de obito por insuficiencia cardiaca")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
base = ".venv/heart_failure_clinical_records_dataset.csv"
df_heart = pd.read_csv(base)

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

# ============================================================================
# 2. ANALISE DAS VARIAVEIS
# ============================================================================
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

continuous_vars = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']

for var in continuous_vars:
    print(f"\n{var}:")
    print(f"  Media: {df_heart[var].mean():.2f}")
    print(f"  Mediana: {df_heart[var].median():.2f}")
    print(f"  Desvio Padrao: {df_heart[var].std():.2f}")
    print(f"  Min: {df_heart[var].min():.2f}")
    print(f"  Max: {df_heart[var].max():.2f}")

# ============================================================================
# 3. JUSTIFICATIVA DO RANDOM FOREST
# ============================================================================
print("\n4. JUSTIFICATIVA DO RANDOM FOREST")
print("-"*60)
print("""
Random Forest foi escolhido porque:
- Nao requer normalizacao dos dados (variaveis em escalas diferentes)
- Lida bem com variaveis binarias e continuas misturadas
- Fornece importancia das features (interpretabilidade medica)
- Resistente a outliers (comum em dados clinicos como CPK)
- Boa performance em datasets pequenos (299 registros)
- Evita overfitting com ensemble de arvores
""")

# ============================================================================
# 4. PRE-PROCESSAMENTO
# ============================================================================
print("\n5. PRE-PROCESSAMENTO")
print("-"*60)

feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                'ejection_fraction', 'high_blood_pressure', 'platelets', 
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
target_col = 'DEATH_EVENT'

X = df_heart[feature_cols]
y = df_heart[target_col]

binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                    'platelets', 'serum_creatinine', 'serum_sodium', 'time']

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

# ============================================================================
# 5. TREINAMENTO
# ============================================================================
print("\n6. TREINAMENTO DO MODELO")
print("-"*60)

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

# Validacao cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 6. AVALIACAO
# ============================================================================
print("\n7. AVALIACAO DO MODELO")
print("-"*60)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusao:")
print(f"  Verdadeiros Negativos: {cm[0,0]}")
print(f"  Falsos Positivos:      {cm[0,1]}")
print(f"  Falsos Negativos:      {cm[1,0]}")
print(f"  Verdadeiros Positivos: {cm[1,1]}")

# ============================================================================
# 7. IMPORTANCIA DAS FEATURES
# ============================================================================
print("\n8. IMPORTANCIA DAS VARIAVEIS")
print("-"*60)

importances = rf_model.named_steps['classifier'].feature_importances_
all_features = numeric_features + binary_features

sorted_idx = np.argsort(importances)[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"{i+1}. {all_features[idx]:<30} {importances[idx]:.4f}")

# ============================================================================
# 8. FUNCAO DE INFERENCIA
# ============================================================================
print("\n9. TESTES DE INFERENCIA")
print("-"*60)

def predizer_risco(modelo, paciente):
    """Prediz o risco de obito do paciente"""
    df_paciente = pd.DataFrame([paciente])
    pred = modelo.predict(df_paciente)[0]
    prob = modelo.predict_proba(df_paciente)[0]
    
    print(f"\nIdade: {paciente['age']} | FE: {paciente['ejection_fraction']}% | Creatinina: {paciente['serum_creatinine']}")
    print(f"Predicao: {'OBITO' if pred == 1 else 'SOBREVIVENTE'}")
    print(f"P(obito): {prob[1]:.1%} | P(sobrevivencia): {prob[0]:.1%}")
    
    return pred, prob

# Teste 1: Paciente alto risco
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

print("="*80)
print("QUESTÃO 1 - HEART FAILURE CLINICAL RECORDS")
print("Analise de predicao de obito por insuficiencia cardiaca")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
base = ".venv/heart_failure_clinical_records_dataset.csv"
df_heart = pd.read_csv(base)

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

# ============================================================================
# 2. ANALISE DAS VARIAVEIS
# ============================================================================
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

continuous_vars = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']

for var in continuous_vars:
    print(f"\n{var}:")
    print(f"  Media: {df_heart[var].mean():.2f}")
    print(f"  Mediana: {df_heart[var].median():.2f}")
    print(f"  Desvio Padrao: {df_heart[var].std():.2f}")
    print(f"  Min: {df_heart[var].min():.2f}")
    print(f"  Max: {df_heart[var].max():.2f}")

# ============================================================================
# 3. JUSTIFICATIVA DO RANDOM FOREST
# ============================================================================
print("\n4. JUSTIFICATIVA DO RANDOM FOREST")
print("-"*60)
print("""
Random Forest foi escolhido porque:
- Nao requer normalizacao dos dados (variaveis em escalas diferentes)
- Lida bem com variaveis binarias e continuas misturadas
- Fornece importancia das features (interpretabilidade medica)
- Resistente a outliers (comum em dados clinicos como CPK)
- Boa performance em datasets pequenos (299 registros)
- Evita overfitting com ensemble de arvores
""")

# ============================================================================
# 4. PRE-PROCESSAMENTO
# ============================================================================
print("\n5. PRE-PROCESSAMENTO")
print("-"*60)

feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                'ejection_fraction', 'high_blood_pressure', 'platelets', 
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
target_col = 'DEATH_EVENT'

X = df_heart[feature_cols]
y = df_heart[target_col]

binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                    'platelets', 'serum_creatinine', 'serum_sodium', 'time']

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

# ============================================================================
# 5. TREINAMENTO
# ============================================================================
print("\n6. TREINAMENTO DO MODELO")
print("-"*60)

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

# Validacao cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 6. AVALIACAO
# ============================================================================
print("\n7. AVALIACAO DO MODELO")
print("-"*60)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusao:")
print(f"  Verdadeiros Negativos: {cm[0,0]}")
print(f"  Falsos Positivos:      {cm[0,1]}")
print(f"  Falsos Negativos:      {cm[1,0]}")
print(f"  Verdadeiros Positivos: {cm[1,1]}")

# ============================================================================
# 7. IMPORTANCIA DAS FEATURES
# ============================================================================
print("\n8. IMPORTANCIA DAS VARIAVEIS")
print("-"*60)

importances = rf_model.named_steps['classifier'].feature_importances_
all_features = numeric_features + binary_features

sorted_idx = np.argsort(importances)[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"{i+1}. {all_features[idx]:<30} {importances[idx]:.4f}")

# ============================================================================
# 8. FUNCAO DE INFERENCIA
# ============================================================================
print("\n9. TESTES DE INFERENCIA")
print("-"*60)

def predizer_risco(modelo, paciente):
    """Prediz o risco de obito do paciente"""
    df_paciente = pd.DataFrame([paciente])
    pred = modelo.predict(df_paciente)[0]
    prob = modelo.predict_proba(df_paciente)[0]
    
    print(f"\nIdade: {paciente['age']} | FE: {paciente['ejection_fraction']}% | Creatinina: {paciente['serum_creatinine']}")
    print(f"Predicao: {'OBITO' if pred == 1 else 'SOBREVIVENTE'}")
    print(f"P(obito): {prob[1]:.1%} | P(sobrevivencia): {prob[0]:.1%}")
    
    return pred, prob

# Teste 1: Paciente alto risco
paciente_teste1 = {
    'age': 75, 'anaemia': 1, 'creatinine_phosphokinase': 582, 'diabetes': 1,
    'ejection_fraction': 20, 'high_blood_pressure': 1, 'platelets': 265000,
    'serum_creatinine': 2.5, 'serum_sodium': 130, 'sex': 1, 'smoking': 1, 'time': 30
}

print("\nCASO 1 - ALTO RISCO")
predizer_risco(rf_model, paciente_teste1)

# Teste 2: Paciente baixo risco
paciente_teste2 = {
    'age': 52, 'anaemia': 0, 'creatinine_phosphokinase': 120, 'diabetes': 0,
    'ejection_fraction': 60, 'high_blood_pressure': 0, 'platelets': 350000,
    'serum_creatinine': 0.8, 'serum_sodium': 140, 'sex': 0, 'smoking': 0, 'time': 150
}

print("\nCASO 2 - BAIXO RISCO")
predizer_risco(rf_model, paciente_teste2)

# ============================================================================
# 9. SALVAR MODELO
# ============================================================================
import joblib
joblib.dump(rf_model, 'heart_failure_model.pkl')
print("\n10. MODELO SALVO")
print("-"*60)
print("Arquivo 'heart_failure_model.pkl' criado com sucesso!")

print("\n" + "="*80)
print("QUESTAO 1 CONCLUIDA")
print("="*80)

