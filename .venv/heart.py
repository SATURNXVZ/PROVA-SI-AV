# QUESTÃO 1 - Heart Failure Clinical Records

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUESTÃO 1 - HEART FAILURE CLINICAL RECORDS")
print("Análise de predição de óbito por insuficiência cardíaca")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
df_heart = pd.read_csv(url)

print("\n1. DOCUMENTAÇÃO DA BASE")
print("-"*60)
print("""
Variáveis e seus significados:
- age: Idade do paciente (anos)
- anaemia: Diminuição de glóbulos vermelhos ou hemoglobina (0=Não, 1=Sim)
- creatinine_phosphokinase: Nível da enzima CPK no sangue (mcg/L)
- diabetes: Se o paciente tem diabetes (0=Não, 1=Sim)
- ejection_fraction: Porcentagem de sangue que sai do coração a cada contração (%)
- high_blood_pressure: Se o paciente tem hipertensão (0=Não, 1=Sim)
- platelets: Plaquetas no sangue (kiloplatelets/mL)
- serum_creatinine: Nível de creatinina sérica no sangue (mg/dL)
- serum_sodium: Nível de sódio sérico no sangue (mEq/L)
- sex: Sexo (0=Mulher, 1=Homem)
- smoking: Se o paciente fuma (0=Não, 1=Sim)
- time: Período de acompanhamento (dias)
- DEATH_EVENT: Alvo - Se o paciente faleceu durante o acompanhamento (0=Não, 1=Sim)
""")

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA INICIAL
# ============================================================================
print("\n2. ANÁLISE EXPLORATÓRIA INICIAL")
print("-"*60)

print(f"Dimensões do dataset: {df_heart.shape}")
print(f"Total de pacientes: {df_heart.shape[0]}")
print(f"Total de variáveis: {df_heart.shape[1]}")
print(f"\nPrimeiros 5 registros:")
print(df_heart.head().to_string())

print(f"\nTipos de dados:")
print(df_heart.dtypes)

print(f"\nVerificação de valores ausentes:")
print(df_heart.isnull().sum())

# ============================================================================
# 3. ANÁLISE DAS VARIÁVEIS BINÁRIAS
# ============================================================================
print("\n3. ANÁLISE DAS VARIÁVEIS BINÁRIAS")
print("-"*60)
print("(Conforme documentação, estas são variáveis binárias que requerem atenção especial)")

binary_vars = {
    'anaemia': 'Diminuição de glóbulos vermelhos',
    'diabetes': 'Diabetes',
    'high_blood_pressure': 'Hipertensão',
    'sex': 'Sexo (0=Mulher, 1=Homem)',
    'smoking': 'Fumante',
    'DEATH_EVENT': 'Óbito (Alvo)'
}

for var, desc in binary_vars.items():
    print(f"\n{var} - {desc}:")
    counts = df_heart[var].value_counts()
    percentages = df_heart[var].value_counts(normalize=True) * 100
    for val in [0, 1]:
        label = {0: 'Não', 1: 'Sim'}[val]
        print(f"  {label}: {counts.get(val, 0)} pacientes ({percentages.get(val, 0):.1f}%)")

# ============================================================================
# 4. ANÁLISE DAS VARIÁVEIS CONTÍNUAS
# ============================================================================
print("\n4. ANÁLISE DAS VARIÁVEIS CONTÍNUAS E DISCRETAS")
print("-"*60)

continuous_vars = {
    'age': 'Idade (anos)',
    'creatinine_phosphokinase': 'CPK no sangue (mcg/L)',
    'ejection_fraction': 'Fração de ejeção (%)',
    'platelets': 'Plaquetas (kiloplatelets/mL)',
    'serum_creatinine': 'Creatinina sérica (mg/dL)',
    'serum_sodium': 'Sódio sérico (mEq/L)',
    'time': 'Período de acompanhamento (dias)'
}

for var, desc in continuous_vars.items():
    print(f"\n{var} - {desc}:")
    print(f"  Média: {df_heart[var].mean():.2f}")
    print(f"  Mediana: {df_heart[var].median():.2f}")
    print(f"  Desvio Padrão: {df_heart[var].std():.2f}")
    print(f"  Mínimo: {df_heart[var].min():.2f}")
    print(f"  Máximo: {df_heart[var].max():.2f}")

# ============================================================================
# 5. VISUALIZAÇÕES
# ============================================================================
print("\n5. GERANDO VISUALIZAÇÕES...")
print("-"*60)

# Configurar estilo
plt.style.use('default')
sns.set_palette("Set2")

# Figura 1: Distribuição das variáveis por classe
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Distribuição das Variáveis por Classe (Óbito vs Sobrevivência)', fontsize=16, fontweight='bold')

plot_vars = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 
             'creatinine_phosphokinase', 'platelets', 'time', 'anaemia', 'high_blood_pressure']

for i, var in enumerate(plot_vars):
    row, col = i // 3, i % 3
    ax = axes[row, col]
    
    if var in binary_vars:
        # Gráfico de barras para variáveis binárias
        counts = df_heart.groupby([var, 'DEATH_EVENT']).size().unstack()
        counts.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title(f'{var}', fontweight='bold')
        ax.set_xlabel('')
        ax.legend(['Sobrevivente', 'Óbito'])
    else:
        # Histograma para variáveis contínuas
        for target_val, color, label in [(0, 'green', 'Sobrevivente'), (1, 'red', 'Óbito')]:
            subset = df_heart[df_heart['DEATH_EVENT'] == target_val][var]
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=label)
        ax.set_title(f'{var}', fontweight='bold')
        ax.legend()
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heart_failure_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Gráfico de distribuições salvo")

# Figura 2: Matriz de correlação
plt.figure(figsize=(14, 12))
correlation_matrix = df_heart.corr()

# Criar máscara para triângulo superior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
            center=0, fmt='.2f', square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Heart Failure (Triângulo Inferior)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('heart_failure_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Matriz de correlação salva")

# ============================================================================
# 6. JUSTIFICATIVA DO METAESTIMADOR
# ============================================================================
print("\n6. JUSTIFICATIVA DA ESCOLHA DO METAESTIMADOR")
print("-"*60)

justificativa = """
METAESTIMADOR PRINCIPAL: RANDOM FOREST CLASSIFIER

Justificativas baseadas nas características da base:

1. NATUREZA DOS DADOS:
   - Base contém variáveis binárias (anaemia, diabetes, high_blood_pressure, sex, smoking)
   - Variáveis contínuas em diferentes escalas (idade em anos, CPK em mcg/L, plaquetas em kiloplatelets/mL)
   - Random Forest NÃO requer normalização dos dados, lidando naturalmente com diferentes escalas

2. TRATAMENTO DE VARIÁVEIS BINÁRIAS:
   - As variáveis binárias são tratadas adequadamente pelo Random Forest
   - O algoritmo consegue criar splits baseados em thresholds naturais (0/1)
   - Não há necessidade de codificação especial para estas variáveis

3. TAMANHO DA BASE:
   - Apenas 299 registros - base pequena
   - Random Forest com bootstrapping funciona bem em datasets pequenos
   - Técnica de ensemble reduz overfitting

4. INTERPRETABILIDADE (CRUCIAL EM CONTEXTO MÉDICO):
   - Random Forest fornece importância das features
   - Permite identificar quais fatores clínicos mais influenciam o risco de óbito
   - Fundamental para validação médica do modelo

5. ROBUSTEZ:
   - Resistente a outliers (comuns em dados clínicos como CPK)
   - Lida bem com features irrelevantes
   - Boa performance sem ajuste extensivo de hiperparâmetros

6. COMPARAÇÃO COM OUTROS ALGORITMOS:
   - SVM: Requer normalização cuidadosa, menos interpretável
   - Regressão Logística: Assume linearidade, pode não capturar interações complexas
   - Redes Neurais: Overfitting em bases pequenas, baixa interpretabilidade
   - Gradient Boosting: Similar performance, mas mais sensível a overfitting em bases pequenas

Portanto, Random Forest é o mais adequado para este problema de classificação binária
com dados clínicos mistos e necessidade de interpretabilidade médica.
"""

print(justificativa)

# ============================================================================
# 7. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n7. PROCEDIMENTOS DE PRÉ-PROCESSAMENTO")
print("-"*60)

print("""
Etapas do pré-processamento:

1. ANÁLISE DE VALORES AUSENTES:
   - Verificação: Não foram encontrados valores ausentes na base
   - Ação: Nenhuma imputação necessária

2. TRATAMENTO DE VARIÁVEIS BINÁRIAS:
   - Variáveis binárias (anaemia, diabetes, high_blood_pressure, sex, smoking)
     já estão codificadas como 0/1
   - Ação: Manter como estão (passthrough no preprocessador)
   - São variáveis categóricas binárias naturais, não requerem one-hot encoding

3. TRATAMENTO DE VARIÁVEIS CONTÍNUAS:
   - StandardScaler aplicado às variáveis contínuas para normalização
   - Importante para algoritmos sensíveis à escala (usado no SVM para comparação)
   - Variáveis: age, creatinine_phosphokinase, ejection_fraction, 
                 platelets, serum_creatinine, serum_sodium, time

4. DIVISÃO TREINO/TESTE:
   - Proporção: 80% treino, 20% teste
   - Estratificação: Manter proporção das classes (preserva distribuição de óbitos)
   - Random state fixado para reprodutibilidade

5. VALIDAÇÃO CRUZADA:
   - 5-fold cross-validation para avaliação robusta
   - Métrica principal: F1-Score (balanceia precisão e recall)
   - Também monitoramos: Accuracy, ROC-AUC, Precisão, Recall
""")

# Separar features e target
feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                'ejection_fraction', 'high_blood_pressure', 'platelets', 
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
target_col = 'DEATH_EVENT'

X = df_heart[feature_cols]
y = df_heart[target_col]

# Identificar tipos de colunas
binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                    'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# Criar preprocessador
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('binary', 'passthrough', binary_features)  # Binárias mantidas como estão
    ])

# Split com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamanho do conjunto de treino: {X_train.shape[0]} pacientes")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} pacientes")
print(f"Proporção de óbitos no treino: {y_train.mean():.1%}")
print(f"Proporção de óbitos no teste: {y_test.mean():.1%}")

# ============================================================================
# 8. TREINAMENTO E AVALIAÇÃO
# ============================================================================
print("\n8. TREINAMENTO E AVALIAÇÃO DO MODELO")
print("-"*60)

# Criar pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Importante para classes desbalanceadas
    ))
])

# Cross-validation
from sklearn.model_selection import cross_validate
cv_results = cross_validate(rf_pipeline, X_train, y_train, cv=5, 
                           scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

print("\nResultados da Validação Cruzada (5-fold):")
print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
print(f"  Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
print(f"  Recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std():.4f})")

# Treinar modelo final
rf_pipeline.fit(X_train, y_train)

# Predições
y_pred = rf_pipeline.predict(X_test)
y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]

# Métricas no teste
print("\nResultados no Conjunto de Teste:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusão:")
print(f"  Verdadeiro Negativo (Sobrevivente correto): {cm[0,0]}")
print(f"  Falso Positivo (Alarme falso):              {cm[0,1]}")
print(f"  Falso Negativo (Óbito não detectado):       {cm[1,0]}")
print(f"  Verdadeiro Positivo (Óbito correto):        {cm[1,1]}")

# Figura 3: Matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Sobrevivente', 'Óbito'],
            yticklabels=['Sobrevivente', 'Óbito'])
axes[0].set_title('Matriz de Confusão', fontweight='bold')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Taxa de Falso Positivo')
axes[1].set_ylabel('Taxa de Verdadeiro Positivo')
axes[1].set_title('Curva ROC', fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heart_failure_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# Importância das features
feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_
all_features = numeric_features + binary_features
importances_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(range(len(importances_df)), importances_df['Importance'])
plt.yticks(range(len(importances_df)), importances_df['Feature'])
plt.xlabel('Importância')
plt.title('Importância das Features - Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig('heart_failure_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nImportância das Features (ordenado):")
for idx, row in importances_df.iloc[::-1].iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

# ============================================================================
# 9. MÓDULO DE INFERÊNCIA
# ============================================================================
print("\n" + "="*80)
print("9. MÓDULO DE INFERÊNCIA EM FUNCIONAMENTO")
print("="*80)

def heart_failure_inference(model, patient_data, patient_name="Paciente"):
    """
    Sistema inteligente de inferência para predição de óbito por insuficiência cardíaca
    
    Parâmetros:
    - model: Modelo treinado (pipeline)
    - patient_data: Dicionário com dados do paciente
    - patient_name: Nome/identificador do paciente
    
    Retorna:
    - Dicionário com predição e probabilidades
    """
    
    # Criar DataFrame com os dados do paciente
    df_patient = pd.DataFrame([patient_data])
    df_patient = df_patient[feature_cols]  # Garantir ordem correta
    
    # Predição
    prediction = model.predict(df_patient)[0]
    probabilities = model.predict_proba(df_patient)[0]
    
    # Resultado
    result = {
        'prediction': prediction,
        'death_probability': probabilities[1],
        'survival_probability': probabilities[0],
        'risk_level': 'ALTO RISCO' if prediction == 1 else 'BAIXO RISCO',
        'confidence': probabilities[1] if prediction == 1 else probabilities[0]
    }
    
    # Exibir resultados formatados
    print(f"\n{'='*60}")
    print(f"RESULTADO DA AVALIAÇÃO - {patient_name}")
    print(f"{'='*60}")
    
    print(f"\nDADOS DO PACIENTE:")
    print(f"  Idade: {patient_data['age']} anos")
    print(f"  Sexo: {'Masculino' if patient_data['sex'] == 1 else 'Feminino'}")
    print(f"  Fração de Ejeção: {patient_data['ejection_fraction']}%")
    print(f"  Creatinina Sérica: {patient_data['serum_creatinine']} mg/dL")
    print(f"  Sódio Sérico: {patient_data['serum_sodium']} mEq/L")
    print(f"  CPK: {patient_data['creatinine_phosphokinase']} mcg/L")
    print(f"  Plaquetas: {patient_data['platelets']:.0f} kiloplatelets/mL")
    
    # Condições
    conditions = []
    if patient_data['anaemia'] == 1: conditions.append("Anemia")
    if patient_data['diabetes'] == 1: conditions.append("Diabetes")
    if patient_data['high_blood_pressure'] == 1: conditions.append("Hipertensão")
    if patient_data['smoking'] == 1: conditions.append("Tabagismo")
    print(f"  Condições: {', '.join(conditions) if conditions else 'Nenhuma condição reportada'}")
    print(f"  Tempo de acompanhamento: {patient_data['time']} dias")
    
    print(f"\nRESULTADO DA PREDIÇÃO:")
    print(f"  Classificação: {result['risk_level']}")
    print(f"  Probabilidade de Óbito: {result['death_probability']:.2%}")
    print(f"  Probabilidade de Sobrevivência: {result['survival_probability']:.2%}")
    print(f"  Confiança na Predição: {result['confidence']:.2%}")
    
    # Recomendações baseadas no risco
    print(f"\nRECOMENDAÇÕES CLÍNICAS:")
    if prediction == 1:
        print("  ⚠️  ALERTA: ALTO RISCO DE EVENTO CARDÍACO FATAL")
        print("  - Monitoramento intensivo imediato")
        print("  - Revisão urgente da medicação")
        print("  - Considerar internação para estabilização")
        print("  - Avaliar necessidade de dispositivos de assistência ventricular")
        if patient_data['ejection_fraction'] < 30:
            print("  - Fração de ejeção crítica - avaliar transplante cardíaco")
        if patient_data['serum_creatinine'] > 1.5:
            print("  - Função renal comprometida - ajustar medicações nefrotóxicas")
    else:
        print("  ✓ BAIXO RISCO DE EVENTO CARDÍACO FATAL")
        print("  - Manter acompanhamento ambulatorial regular")
        print("  - Continuar medicação conforme prescrito")
        print("  - Reavaliar em 3-6 meses")
        print("  - Manter hábitos saudáveis e controle de fatores de risco")
    
    return result

# Testar com pacientes exemplo
print("\n" + "="*60)
print("TESTES DE INFERÊNCIA")
print("="*60)

# Paciente 1: Perfil de alto risco
patient_1 = {
    'age': 75,
    'anaemia': 1,
    'creatinine_phosphokinase': 582,
    'diabetes': 1,
    'ejection_fraction': 20,
    'high_blood_pressure': 1,
    'platelets': 265000,
    'serum_creatinine': 2.5,
    'serum_sodium': 130,
    'sex': 1,
    'smoking': 1,
    'time': 30
}

result_1 = heart_failure_inference(rf_pipeline, patient_1, "Paciente Exemplo 1 - Perfil Alto Risco")

# Paciente 2: Perfil de baixo risco
patient_2 = {
    'age': 52,
    'anaemia': 0,
    'creatinine_phosphokinase': 120,
    'diabetes': 0,
    'ejection_fraction': 60,
    'high_blood_pressure': 0,
    'platelets': 350000,
    'serum_creatinine': 0.8,
    'serum_sodium': 140,
    'sex': 0,
    'smoking': 0,
    'time': 150
}

result_2 = heart_failure_inference(rf_pipeline, patient_2, "Paciente Exemplo 2 - Perfil Baixo Risco")

# Paciente 3: Caso limítrofe
patient_3 = {
    'age': 65,
    'anaemia': 0,
    'creatinine_phosphokinase': 250,
    'diabetes': 1,
    'ejection_fraction': 38,
    'high_blood_pressure': 1,
    'platelets': 280000,
    'serum_creatinine': 1.3,
    'serum_sodium': 136,
    'sex': 1,
    'smoking': 0,
    'time': 90
}

result_3 = heart_failure_inference(rf_pipeline, patient_3, "Paciente Exemplo 3 - Caso Limítrofe")

# Validar com paciente real do dataset
print(f"\n{'='*60}")
print("VALIDAÇÃO COM PACIENTES REAIS DO DATASET")
print("="*60)

for i in range(5):
    idx = np.random.randint(0, len(X_test))
    patient_real = X_test.iloc[idx].to_dict()
    true_label = y_test.iloc[idx]
    
    df_patient = pd.DataFrame([patient_real])[feature_cols]
    pred = rf_pipeline.predict(df_patient)[0]
    proba = rf_pipeline.predict_proba(df_patient)[0]
    
    print(f"\nPaciente Real #{i+1}:")
    print(f"  Idade: {patient_real['age']}, FE: {patient_real['ejection_fraction']}%")
    print(f"  Real: {'Óbito' if true_label == 1 else 'Sobrevivente'}")
    print(f"  Predito: {'Óbito' if pred == 1 else 'Sobrevivente'}")
    print(f"  Confiança: {max(proba):.2%}")
    print(f"  Correto: {'✓' if pred == true_label else '✗'}")

# Salvar modelo
import joblib
joblib.dump(rf_pipeline, 'heart_failure_model.pkl')
print(f"\n✓ Modelo salvo como 'heart_failure_model.pkl'")

print("\n" + "="*80)
print("QUESTÃO 1 CONCLUÍDA")
print("="*80)