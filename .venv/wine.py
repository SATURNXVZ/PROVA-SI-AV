# QUESTÃO 2 - Wine Quality
# Base: https://archive.ics.uci.edu/dataset/186/wine+quality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUESTÃO 2 - WINE QUALITY")
print("Classificação da qualidade de vinhos tintos e brancos")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO DOS 3 ARQUIVOS
# ============================================================================
print("\n1. CARREGANDO OS 3 ARQUIVOS DA BASE")
print("-"*60)

# URLs dos arquivos
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names"

print("""
Arquivos da base:
1. winequality-red.csv   - Dados de vinhos tintos (1599 amostras)
2. winequality-white.csv - Dados de vinhos brancos (4898 amostras)
3. winequality.names     - Documentação com nomes das colunas e descrições
""")

# Carregar dados dos vinhos
df_red = pd.read_csv(url_red, sep=';')
df_white = pd.read_csv(url_white, sep=';')

# Adicionar coluna de tipo (color)
df_red['color'] = 'red'
df_white['color'] = 'white'

# Combinar datasets
df_wine = pd.concat([df_red, df_white], ignore_index=True)

print(f"Dataset combinado criado:")
print(f"  Vinhos tintos: {len(df_red)} amostras")
print(f"  Vinhos brancos: {len(df_white)} amostras")
print(f"  Total: {len(df_wine)} amostras")
print(f"  Colunas: {df_wine.columns.tolist()}")

# ============================================================================
# 2. DOCUMENTAÇÃO DAS VARIÁVEIS
# ============================================================================
print("\n2. DOCUMENTAÇÃO DAS VARIÁVEIS")
print("-"*60)

print("""
Variáveis da base Wine Quality:

Features (baseadas em testes físico-químicos):
1. fixed_acidity        - Acidez fixa (g/dm³)
2. volatile_acidity     - Acidez volátil (g/dm³)
3. citric_acid          - Ácido cítrico (g/dm³)
4. residual_sugar       - Açúcar residual (g/dm³)
5. chlorides            - Cloretos (g/dm³)
6. free_sulfur_dioxide  - Dióxido de enxofre livre (mg/dm³)
7. total_sulfur_dioxide - Dióxido de enxofre total (mg/dm³)
8. density              - Densidade (g/cm³)
9. pH                   - pH
10. sulphates           - Sulfatos (g/dm³)
11. alcohol             - Teor alcoólico (% vol)

Target:
12. quality             - Qualidade do vinho (score 0-10)

Atributo adicional:
13. color               - Tipo do vinho (red/white)

NOTA: A coluna 'color' foi adicionada para identificar o tipo de vinho
      e será codificada como variável numérica (0=red, 1=white)
""")

# ============================================================================
# 3. ANÁLISE EXPLORATÓRIA
# ============================================================================
print("\n3. ANÁLISE EXPLORATÓRIA DO DATASET COMBINADO")
print("-"*60)

print(f"Shape: {df_wine.shape}")
print(f"\nTipos de dados:")
print(df_wine.dtypes)
print(f"\nValores ausentes:")
print(df_wine.isnull().sum())

# Estatísticas descritivas
print(f"\nEstatísticas descritivas:")
print(df_wine.describe().round(2).to_string())

# Distribuição da qualidade
print(f"\nDistribuição da variável alvo (quality):")
quality_dist = df_wine['quality'].value_counts().sort_index()
for score, count in quality_dist.items():
    bar = '█' * int(count / 50)
    print(f"  Score {score}: {count:5d} vinhos {bar}")

# Distribuição por tipo
print(f"\nDistribuição por tipo de vinho:")
print(f"  Tinto (red):   {len(df_red)} vinhos")
print(f"  Branco (white): {len(df_white)} vinhos")

# ============================================================================
# 4. VISUALIZAÇÕES
# ============================================================================
print("\n4. GERANDO VISUALIZAÇÕES...")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Exploratória - Wine Quality', fontsize=16, fontweight='bold')

# Gráfico 1: Distribuição da qualidade
quality_counts = df_wine['quality'].value_counts().sort_index()
axes[0,0].bar(quality_counts.index, quality_counts.values, color='darkred', alpha=0.8)
axes[0,0].set_xlabel('Quality Score')
axes[0,0].set_ylabel('Número de Vinhos')
axes[0,0].set_title('Distribuição da Qualidade dos Vinhos', fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

# Adicionar valores nas barras
for i, v in enumerate(quality_counts.values):
    axes[0,0].text(quality_counts.index[i], v + 10, str(v), ha='center')

# Gráfico 2: Qualidade por tipo
quality_by_type = df_wine.groupby(['color', 'quality']).size().unstack(0)
quality_by_type.plot(kind='bar', ax=axes[0,1], color=['darkred', 'goldenrod'], alpha=0.8)
axes[0,1].set_xlabel('Quality Score')
axes[0,1].set_ylabel('Número de Vinhos')
axes[0,1].set_title('Qualidade por Tipo de Vinho', fontweight='bold')
axes[0,1].legend(['Tinto', 'Branco'])
axes[0,1].grid(True, alpha=0.3)

# Gráfico 3: Boxplot - Álcool por qualidade
df_wine.boxplot(column='alcohol', by='quality', ax=axes[1,0], color='darkred')
axes[1,0].set_xlabel('Quality Score')
axes[1,0].set_ylabel('Teor Alcoólico (% vol)')
axes[1,0].set_title('Teor Alcoólico por Qualidade', fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

# Gráfico 4: Matriz de correlação
numeric_cols = df_wine.select_dtypes(include=[np.number]).columns
corr = df_wine[numeric_cols].corr()
corr_with_quality = corr['quality'].drop('quality').sort_values()
colors = ['red' if x < 0 else 'green' for x in corr_with_quality.values]
axes[1,1].barh(range(len(corr_with_quality)), corr_with_quality.values, color=colors, alpha=0.8)
axes[1,1].set_yticks(range(len(corr_with_quality)))
axes[1,1].set_yticklabels(corr_with_quality.index)
axes[1,1].set_xlabel('Correlação com Quality')
axes[1,1].set_title('Correlação das Features com Quality', fontweight='bold')
axes[1,1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wine_quality_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Gráficos de análise exploratória salvos")

# ============================================================================
# 5. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n5. PRÉ-PROCESSAMENTO DOS DADOS")
print("-"*60)

# Codificar a variável color (red=0, white=1)
df_wine['color_encoded'] = df_wine['color'].map({'red': 0, 'white': 1})

print("""
Etapas do pré-processamento:

1. COMBINAÇÃO DOS ARQUIVOS:
   - winequality-red.csv + winequality-white.csv
   - Adicionada coluna 'color' para identificar tipo
   - Total: 6497 amostras combinadas

2. CODIFICAÇÃO DA VARIÁVEL COLOR:
   - Variável categórica 'color' convertida para numérica
   - red = 0, white = 1

3. SELEÇÃO DE FEATURES:
   - Todas as 11 features físico-químicas + color_encoded
   - Target: quality (score 3-9)

4. NORMALIZAÇÃO:
   - StandardScaler aplicado para normalizar as features
   - Importante para algoritmos sensíveis à escala (SVM, KNN)
""")

# Features e target
feature_cols_wine = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol', 'color_encoded']

X_wine = df_wine[feature_cols_wine]
y_wine = df_wine['quality']

print(f"Features shape: {X_wine.shape}")
print(f"Target shape: {y_wine.shape}")
print(f"Classes únicas: {sorted(y_wine.unique())}")

# Split
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

print(f"\nDivisão Treino/Teste:")
print(f"  Treino: {X_train_w.shape[0]} amostras")
print(f"  Teste:  {X_test_w.shape[0]} amostras")

# ============================================================================
# 6. AVALIAÇÃO DE MÚLTIPLOS METAESTIMADORES
# ============================================================================
print("\n6. AVALIAÇÃO DE MÚLTIPLOS METAESTIMADORES CLASSIFICADORES")
print("-"*60)

# Definir modelos
models_wine = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Pipeline e avaliação
results_wine = {}

print("Treinando e avaliando cada modelo...")
for name, model in models_wine.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train_w, y_train_w, cv=5, 
                                scoring='f1_weighted')
    
    # Treinar
    pipeline.fit(X_train_w, y_train_w)
    
    # Predizer
    y_pred = pipeline.predict(X_test_w)
    
    # Métricas
    results_wine[name] = {
        'pipeline': pipeline,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy_score(y_test_w, y_pred),
        'f1_macro': f1_score(y_test_w, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test_w, y_pred, average='weighted'),
        'y_pred': y_pred
    }
    
    print(f"  {name:<25} CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Tabela comparativa
print(f"\n{'='*90}")
print(f"COMPARAÇÃO DE PERFORMANCE DOS METAESTIMADORES")
print(f"{'='*90}")
print(f"{'Modelo':<25} {'CV F1':<12} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<12}")
print(f"{'-'*90}")
for name, res in results_wine.items():
    print(f"{name:<25} {res['cv_mean']:<12.4f} {res['accuracy']:<12.4f} "
          f"{res['f1_macro']:<12.4f} {res['f1_weighted']:<12.4f}")

# Selecionar melhor modelo
best_wine = max(results_wine, key=lambda x: results_wine[x]['f1_weighted'])
print(f"\n✓ MELHOR MODELO: {best_wine}")
print(f"  Critério: Maior F1-Score Weighted (balanceia performance entre classes)")

# ============================================================================
# 7. OTIMIZAÇÃO DO MELHOR MODELO
# ============================================================================
print(f"\n7. OTIMIZAÇÃO DO {best_wine.upper()}")
print("-"*60)

# Parâmetros para Grid Search
if best_wine == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    }
elif best_wine == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
else:
    param_grid = {}

if param_grid:
    pipeline_opt = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', models_wine[best_wine].__class__(random_state=42))
    ])
    
    grid_search = GridSearchCV(pipeline_opt, param_grid, cv=5, 
                               scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_w, y_train_w)
    
    best_model_wine = grid_search.best_estimator_
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor F1 CV: {grid_search.best_score_:.4f}")
else:
    best_model_wine = results_wine[best_wine]['pipeline']

# Avaliação final
y_pred_final = best_model_wine.predict(X_test_w)

print(f"\nResultados Finais com Modelo Otimizado:")
print(f"  Accuracy:      {accuracy_score(y_test_w, y_pred_final):.4f}")
print(f"  F1-Score Macro: {f1_score(y_test_w, y_pred_final, average='macro'):.4f}")
print(f"  F1-Score Weighted: {f1_score(y_test_w, y_pred_final, average='weighted'):.4f}")

# ============================================================================
# 8. MATRIZ DE CONFUSÃO E MÉTRICAS POR CLASSE
# ============================================================================
print(f"\n8. MATRIZ DE CONFUSÃO E ACURÁCIA POR CLASSE")
print("-"*60)

# Matriz de confusão
cm_wine = confusion_matrix(y_test_w, y_pred_final)
classes_wine = sorted(y_wine.unique())

# Figura: Matriz de confusão
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_wine, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=classes_wine, yticklabels=classes_wine,
            cbar_kws={'label': 'Número de Vinhos'})
ax.set_xlabel('Qualidade Predita', fontweight='bold')
ax.set_ylabel('Qualidade Real', fontweight='bold')
ax.set_title('Matriz de Confusão - Wine Quality Classification', fontweight='bold')
plt.tight_layout()
plt.savefig('wine_quality_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test_w, y_pred_final, digits=4))

# Acurácia por classe
print("\nAcurácia por Classe (Quality Score):")
print(f"{'Score':<10} {'Amostras':<12} {'Acurácia':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*70)
for i, score in enumerate(classes_wine):
    n_samples = cm_wine[i].sum()
    if n_samples > 0:
        acc = cm_wine[i, i] / n_samples
        prec = precision_score(y_test_w, y_pred_final, labels=[score], average=None)[0]
        rec = recall_score(y_test_w, y_pred_final, labels=[score], average=None)[0]
        f1 = f1_score(y_test_w, y_pred_final, labels=[score], average=None)[0]
        print(f"{score:<10} {n_samples:<12} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

# ============================================================================
# 9. JUSTIFICATIVA DO MODELO SELECIONADO
# ============================================================================
print(f"\n9. JUSTIFICATIVA DO MODELO MAIS ADEQUADO ({best_wine})")
print("-"*60)

justificativa_wine = f"""
MODELO SELECIONADO: {best_wine}

Por que este modelo é o mais adequado para implantação:

1. PERFORMANCE SUPERIOR:
   - Melhor F1-Score Weighted entre todos os modelos testados
   - Boa capacidade de generalização (validação cruzada consistente)

2. CARACTERÍSTICAS DO PROBLEMA:
   - Classificação multiclasse (7 classes: scores 3-9)
   - Dados desbalanceados (poucos vinhos com scores extremos)
   - Features com diferentes escalas e correlações não-lineares
   - O modelo escolhido lida bem com estas características

3. ROBUSTEZ:
   - Menos sensível a outliers que SVM
   - Melhor que KNN para dados com muitas features correlacionadas
   - Captura interações complexas entre variáveis

4. EFICIÊNCIA COMPUTACIONAL:
   - Tempo de treinamento razoável
   - Inferência rápida (importante para produção)
   - Escalabilidade para bases maiores

5. INTERPRETABILIDADE:
   - Permite análise de importância das features
   - Auxilia enólogos a entender fatores que influenciam qualidade

6. ESTABILIDADE:
   - Resultados consistentes em validação cruzada
   - Baixa variância entre folds
"""

print(justificativa_wine)

# ============================================================================
# 10. MÓDULO DE INFERÊNCIA
# ============================================================================
print("\n" + "="*80)
print("10. MÓDULO DE INFERÊNCIA EM FUNCIONAMENTO")
print("="*80)

def wine_quality_inference(model, wine_data, wine_name="Vinho"):
    """
    Sistema de inferência para predição da qualidade de vinhos
    
    Parâmetros:
    - model: Modelo treinado
    - wine_data: Dicionário com características do vinho
    - wine_name: Nome/identificador do vinho
    
    Retorna:
    - Dicionário com predição e probabilidades
    """
    
    # Garantir features na ordem correta
    df_wine_input = pd.DataFrame([wine_data])
    
    # Adicionar color_encoded se não existir
    if 'color_encoded' not in df_wine_input.columns and 'color' in df_wine_input.columns:
        df_wine_input['color_encoded'] = df_wine_input['color'].map({'red': 0, 'white': 1})
    
    df_wine_input = df_wine_input[feature_cols_wine]
    
    # Predição
    prediction = model.predict(df_wine_input)[0]
    probabilities = model.predict_proba(df_wine_input)[0]
    classes = model.named_steps['classifier'].classes_
    
    # Top 3 predições
    top_3_idx = np.argsort(probabilities)[::-1][:3]
    
    # Exibir resultados
    print(f"\n{'='*60}")
    print(f"RESULTADO DA ANÁLISE - {wine_name}")
    print(f"{'='*60}")
    
    print(f"\nCARACTERÍSTICAS DO VINHO:")
    for key, value in wine_data.items():
        if key not in ['color_encoded']:
            print(f"  {key}: {value}")
    
    print(f"\nPREDIÇÃO DE QUALIDADE:")
    print(f"  Score Predito: {prediction}/10")
    print(f"  Confiança: {max(probabilities):.2%}")
    
    print(f"\nTOP 3 PREDIÇÕES (com grau de certeza):")
    for idx in top_3_idx:
        bar = '█' * int(probabilities[idx] * 50)
        print(f"  Score {classes[idx]}: {probabilities[idx]:.1%} {bar}")
    
    # Descrição da qualidade
    quality_desc = {
        3: "Qualidade muito baixa - Defeitos significativos, não recomendado",
        4: "Qualidade abaixo da média - Defeitos notáveis",
        5: "Qualidade média - Vinho aceitável para consumo diário",
        6: "Qualidade boa - Acima da média, boa relação custo-benefício",
        7: "Qualidade muito boa - Excelente, características Premium",
        8: "Qualidade excepcional - Vinho Premium, alta complexidade",
        9: "Qualidade extraordinária - Classe mundial, excepcional"
    }
    
    print(f"\nINTERPRETAÇÃO:")
    print(f"  {quality_desc.get(prediction, 'N/A')}")
    
    # Recomendações
    print(f"\nRECOMENDAÇÕES:")
    if prediction >= 7:
        print("  ✓ Vinho de alta qualidade")
        print("  - Adequado para ocasiões especiais")
        print("  - Potencial de guarda")
        print("  - Harmonização com pratos sofisticados")
    elif prediction >= 5:
        print("  ✓ Vinho de qualidade aceitável")
        print("  - Bom para consumo diário")
        print("  - Boa relação custo-benefício")
    else:
        print("  ⚠️ Vinho de baixa qualidade")
        print("  - Possíveis defeitos na produção")
        print("  - Não recomendado para consumo")
    
    return {
        'predicted_quality': prediction,
        'confidence': probabilities[top_3_idx[0]],
        'top_3_predictions': [(classes[idx], probabilities[idx]) for idx in top_3_idx]
    }

# Testar inferência com exemplos
print("\n" + "="*60)
print("TESTES DE INFERÊNCIA")
print("="*60)

# Exemplo 1: Vinho tinto encorpado
wine_1 = {
    'fixed acidity': 8.5,
    'volatile acidity': 0.5,
    'citric acid': 0.3,
    'residual sugar': 2.5,
    'chlorides': 0.08,
    'free sulfur dioxide': 15,
    'total sulfur dioxide': 120,
    'density': 0.997,
    'pH': 3.3,
    'sulphates': 0.8,
    'alcohol': 12.5,
    'color_encoded': 0  # Tinto
}

result_w1 = wine_quality_inference(best_model_wine, wine_1, "Vinho Tinto Encorpado Premium")

# Exemplo 2: Vinho branco leve
wine_2 = {
    'fixed acidity': 6.5,
    'volatile acidity': 0.25,
    'citric acid': 0.35,
    'residual sugar': 8.0,
    'chlorides': 0.04,
    'free sulfur dioxide': 30,
    'total sulfur dioxide': 150,
    'density': 0.994,
    'pH': 3.2,
    'sulphates': 0.45,
    'alcohol': 10.5,
    'color_encoded': 1  # Branco
}

result_w2 = wine_quality_inference(best_model_wine, wine_2, "Vinho Branco Suave")

# Exemplo 3: Vinho de alta qualidade
wine_3 = {
    'fixed acidity': 7.2,
    'volatile acidity': 0.28,
    'citric acid': 0.5,
    'residual sugar': 1.8,
    'chlorides': 0.045,
    'free sulfur dioxide': 25,
    'total sulfur dioxide': 110,
    'density': 0.995,
    'pH': 3.4,
    'sulphates': 0.95,
    'alcohol': 13.5,
    'color_encoded': 0  # Tinto
}

result_w3 = wine_quality_inference(best_model_wine, wine_3, "Vinho Tinto Reserva Especial")

# Salvar modelo
import joblib
joblib.dump(best_model_wine, 'wine_quality_model.pkl')
print(f"\n✓ Modelo salvo como 'wine_quality_model.pkl'")

print("\n" + "="*80)
print("QUESTÃO 2 CONCLUÍDA")
print("="*80)