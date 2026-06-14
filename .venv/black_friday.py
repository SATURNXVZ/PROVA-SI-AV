# QUESTÃO 3 - Black Friday Sales Dataset
print("\n" + "="*80)
print("QUESTÃO 3 - BLACK FRIDAY SALES DATASET")
print("="*80)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Carregar dados (assumindo que o arquivo está no diretório)
# Para este exemplo, vamos criar dados sintéticos baseados na estrutura do Black Friday
# Em produção, você carregaria: df = pd.read_csv('black_friday_sales.csv')

print("\nNOTA: Para demonstração, usaremos dados simulados baseados na estrutura real.")
print("Em produção, substitua pelo carregamento real do arquivo CSV.\n")

# Simular dados do Black Friday (estrutura real)
np.random.seed(42)
n_samples = 5000

# Criar dados simulados
data = {
    'User_ID': np.random.randint(1, 1001, n_samples),
    'Product_ID': [f'P{str(i).zfill(6)}' for i in np.random.randint(1, 500, n_samples)],
    'Gender': np.random.choice(['M', 'F'], n_samples, p=[0.65, 0.35]),
    'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], n_samples),
    'Occupation': np.random.randint(0, 21, n_samples),
    'City_Category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3]),
    'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], n_samples),
    'Marital_Status': np.random.choice([0, 1], n_samples),
    'Product_Category_1': np.random.randint(1, 21, n_samples),
    'Product_Category_2': np.random.choice([np.nan] + list(range(2, 19)), n_samples, p=[0.3] + [0.7/17]*17),
    'Product_Category_3': np.random.choice([np.nan] + list(range(3, 19)), n_samples, p=[0.5] + [0.5/16]*16),
    'Purchase': np.random.randint(5000, 25000, n_samples),
    'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'UPI', 'Net Banking'], n_samples),
    'Purchase_Time': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
}

df_bf = pd.DataFrame(data)

print(f"Shape do dataset: {df_bf.shape}")
print(f"\nColunas: {df_bf.columns.tolist()}")
print(f"\nPrimeiras linhas:")
print(df_bf.head())

# Criar faixas etárias para predição
def create_age_group(age):
    if age in ['0-17', '18-25']:
        return 'Jovem'
    elif age in ['26-35', '36-45']:
        return 'Adulto'
    else:
        return 'Senior'

df_bf['age_group'] = df_bf['Age'].apply(create_age_group)

print(f"\nDistribuição das faixas etárias:")
print(df_bf['age_group'].value_counts())

# 1. ANÁLISE EXPLORATÓRIA
print("\n1. ANÁLISE EXPLORATÓRIA")
print("-"*50)

print(f"Informações do dataset:")
print(df_bf.info())
print(f"\nValores nulos:")
print(df_bf.isnull().sum())

# Distribuições das variáveis alvo
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df_bf['Product_Category_1'].value_counts().head(10).plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Top 10 Categorias de Produto')
axes[0].set_xlabel('Categoria')
axes[0].set_ylabel('Contagem')

df_bf['Payment_Method'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=sns.color_palette('Set2'))
axes[1].set_title('Métodos de Pagamento')

df_bf['age_group'].value_counts().plot(kind='bar', ax=axes[2], color='coral')
axes[2].set_title('Distribuição de Faixas Etárias')
axes[2].set_xlabel('Faixa Etária')
axes[2].set_ylabel('Contagem')

plt.tight_layout()
plt.savefig('black_friday_targets.png', dpi=100)
plt.show()

# 2. PRÉ-PROCESSAMENTO
print("\n2. PRÉ-PROCESSAMENTO")
print("-"*50)

# Features para o modelo
feature_cols_bf = ['Gender', 'Age', 'Occupation', 'City_Category', 
                   'Stay_In_Current_City_Years', 'Marital_Status',
                   'Purchase', 'Purchase_Time']

# Tratar valores nulos em categorias de produto
df_bf['Product_Category_2'] = df_bf['Product_Category_2'].fillna(0)
df_bf['Product_Category_3'] = df_bf['Product_Category_3'].fillna(0)

# Criar preprocessador
categorical_features = ['Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Purchase_Time', 'Age']
numeric_features = ['Occupation', 'Marital_Status', 'Purchase']

preprocessor_bf = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# 3. SISTEMA 1: PREDIÇÃO DE CATEGORIA DE PRODUTO
print("\n3. SISTEMA 1: PREDIÇÃO DE CATEGORIA DE PRODUTO")
print("-"*50)

# Target 1: Product_Category_1
X1 = df_bf[feature_cols_bf]
y1 = df_bf['Product_Category_1']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)

# Modelo para categoria de produto
model_product = Pipeline([
    ('preprocessor', preprocessor_bf),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Treinar
print("Treinando modelo de categoria de produto...")
model_product.fit(X1_train, y1_train)

# Avaliar
y1_pred = model_product.predict(X1_test)
y1_proba = model_product.predict_proba(X1_test)

print(f"\nResultados - Categoria de Produto:")
print(f"Accuracy: {accuracy_score(y1_test, y1_pred):.4f}")
print(f"F1-Score (macro): {f1_score(y1_test, y1_pred, average='macro'):.4f}")
print(f"F1-Score (weighted): {f1_score(y1_test, y1_pred, average='weighted'):.4f}")

# Matriz de confusão para top categorias
top_categories = df_bf['Product_Category_1'].value_counts().head(5).index

plt.figure(figsize=(10, 8))
cm1 = confusion_matrix(y1_test, y1_pred)
# Filtrar para top categorias
mask = np.isin(model_product.named_steps['classifier'].classes_, top_categories)
sns.heatmap(cm1[:10, :10], annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Categoria de Produto (Top 10)')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('bf_product_confusion.png', dpi=100)
plt.show()

# 4. SISTEMA 2: PREDIÇÃO DE MÉTODO DE PAGAMENTO
print("\n4. SISTEMA 2: PREDIÇÃO DE MÉTODO DE PAGAMENTO")
print("-"*50)

# Target 2: Payment_Method
X2 = df_bf[feature_cols_bf]
y2 = df_bf['Payment_Method']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

# Modelo para método de pagamento
model_payment = Pipeline([
    ('preprocessor', preprocessor_bf),
    ('classifier', GradientBoostingClassifier(n_estimators=150, random_state=42))
])

# Treinar
print("Treinando modelo de método de pagamento...")
model_payment.fit(X2_train, y2_train)

# Avaliar
y2_pred = model_payment.predict(X2_test)
y2_proba = model_payment.predict_proba(X2_test)

print(f"\nResultados - Método de Pagamento:")
print(f"Accuracy: {accuracy_score(y2_test, y2_pred):.4f}")
print(f"F1-Score (macro): {f1_score(y2_test, y2_pred, average='macro'):.4f}")
print(f"F1-Score (weighted): {f1_score(y2_test, y2_pred, average='weighted'):.4f}")

# Matriz de confusão
plt.figure(figsize=(10, 8))
cm2 = confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=model_payment.named_steps['classifier'].classes_,
            yticklabels=model_payment.named_steps['classifier'].classes_)
plt.title('Matriz de Confusão - Método de Pagamento')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('bf_payment_confusion.png', dpi=100)
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação - Método de Pagamento:")
print(classification_report(y2_test, y2_pred))

# 5. SISTEMA 3: PREDIÇÃO DE FAIXA ETÁRIA
print("\n5. SISTEMA 3: PREDIÇÃO DE FAIXA ETÁRIA")
print("-"*50)

# Target 3: age_group
X3 = df_bf[feature_cols_bf]
y3 = df_bf['age_group']

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)

# Modelo para faixa etária
model_age = Pipeline([
    ('preprocessor', preprocessor_bf),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])

# Treinar
print("Treinando modelo de faixa etária...")
model_age.fit(X3_train, y3_train)

# Avaliar
y3_pred = model_age.predict(X3_test)
y3_proba = model_age.predict_proba(X3_test)

print(f"\nResultados - Faixa Etária:")
print(f"Accuracy: {accuracy_score(y3_test, y3_pred):.4f}")
print(f"F1-Score (macro): {f1_score(y3_test, y3_pred, average='macro'):.4f}")
print(f"F1-Score (weighted): {f1_score(y3_test, y3_pred, average='weighted'):.4f}")

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm3 = confusion_matrix(y3_test, y3_pred)
sns.heatmap(cm3, annot=True, fmt='d', cmap='Oranges',
            xticklabels=model_age.named_steps['classifier'].classes_,
            yticklabels=model_age.named_steps['classifier'].classes_)
plt.title('Matriz de Confusão - Faixa Etária')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('bf_age_confusion.png', dpi=100)
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação - Faixa Etária:")
print(classification_report(y3_test, y3_pred))

# 6. COMPARAÇÃO GERAL
print("\n" + "="*80)
print("COMPARAÇÃO GERAL DOS SISTEMAS")
print("-"*80)

systems = {
    'Categoria de Produto': {'model': model_product, 'y_test': y1_test, 'y_pred': y1_pred},
    'Método de Pagamento': {'model': model_payment, 'y_test': y2_test, 'y_pred': y2_pred},
    'Faixa Etária': {'model': model_age, 'y_test': y3_test, 'y_pred': y3_pred}
}

print(f"{'Sistema':<25} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
print("-"*70)

for name, sys in systems.items():
    acc = accuracy_score(sys['y_test'], sys['y_pred'])
    prec = precision_score(sys['y_test'], sys['y_pred'], average='weighted')
    rec = recall_score(sys['y_test'], sys['y_pred'], average='weighted')
    f1 = f1_score(sys['y_test'], sys['y_pred'], average='weighted')
    print(f"{name:<25} {acc:<10.4f} {prec:<12.4f} {rec:<10.4f} {f1:<10.4f}")

# 7. MÓDULO DE INFERÊNCIA INTEGRADO
print("\n" + "="*80)
print("MÓDULO DE INFERÊNCIA INTEGRADO - BLACK FRIDAY")
print("="*80)

def black_friday_inference(models, customer_data):
    """
    Sistema inteligente para predições do Black Friday
    
    Parameters:
    - models: dict with 'product', 'payment', 'age' models
    - customer_data: dict with customer/sale information
    """
    
    print("\n" + "="*60)
    print("SISTEMA INTELIGENTE DE ANÁLISE DE VENDAS - BLACK FRIDAY")
    print("="*60)
    
    # Criar DataFrame com dados do cliente
    df_input = pd.DataFrame([customer_data])
    
    # Garantir todas as features necessárias
    required_features = feature_cols_bf
    for feat in required_features:
        if feat not in df_input.columns:
            df_input[feat] = 'Unknown'
    
    df_input = df_input[required_features]
    
    print("\nDADOS DA CIRCUNSTÂNCIA DE VENDA:")
    print("-"*40)
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
    
    # 1. Predição de Categoria de Produto
    print("\n" + "="*40)
    print("1. PREDIÇÃO DE CATEGORIA DE PRODUTO")
    print("-"*40)
    
    product_pred = models['product'].predict(df_input)[0]
    product_proba = models['product'].predict_proba(df_input)[0]
    product_classes = models['product'].named_steps['classifier'].classes_
    
    # Top 3 categorias
    top3_product_idx = np.argsort(product_proba)[::-1][:3]
    
    print(f"Categoria Predita: {product_pred}")
    print(f"\nTop 3 Categorias com Grau de Certeza:")
    for idx in top3_product_idx:
        print(f"  Categoria {product_classes[idx]}: {product_proba[idx]:.2%}")
    
    # 2. Predição de Método de Pagamento
    print("\n" + "="*40)
    print("2. PREDIÇÃO DE MÉTODO DE PAGAMENTO")
    print("-"*40)
    
    payment_pred = models['payment'].predict(df_input)[0]
    payment_proba = models['payment'].predict_proba(df_input)[0]
    payment_classes = models['payment'].named_steps['classifier'].classes_
    
    top3_payment_idx = np.argsort(payment_proba)[::-1][:3]
    
    print(f"Método Predito: {payment_pred}")
    print(f"\nTop 3 Métodos com Grau de Certeza:")
    for idx in top3_payment_idx:
        print(f"  {payment_classes[idx]}: {payment_proba[idx]:.2%}")
    
    # 3. Predição de Faixa Etária
    print("\n" + "="*40)
    print("3. PREDIÇÃO DE FAIXA ETÁRIA")
    print("-"*40)
    
    age_pred = models['age'].predict(df_input)[0]
    age_proba = models['age'].predict_proba(df_input)[0]
    age_classes = models['age'].named_steps['classifier'].classes_
    
    top3_age_idx = np.argsort(age_proba)[::-1][:3]
    
    print(f"Faixa Etária Predita: {age_pred}")
    print(f"\nTop 3 Faixas com Grau de Certeza:")
    for idx in top3_age_idx:
        print(f"  {age_classes[idx]}: {age_proba[idx]:.2%}")
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DAS PREDIÇÕES")
    print("-"*60)
    print(f"  Categoria de Produto Recomendada: {product_pred}")
    print(f"  Método de Pagamento Mais Provável: {payment_pred}")
    print(f"  Faixa Etária Mais Provável: {age_pred}")
    
    return {
        'product_category': product_pred,
        'product_confidence': product_proba[top3_product_idx[0]],
        'payment_method': payment_pred,
        'payment_confidence': payment_proba[top3_payment_idx[0]],
        'age_group': age_pred,
        'age_confidence': age_proba[top3_age_idx[0]]
    }

# Testar o sistema com exemplos
test_scenarios = [
    {
        'Gender': 'M',
        'Age': '26-35',
        'Occupation': 5,
        'City_Category': 'A',
        'Stay_In_Current_City_Years': '2',
        'Marital_Status': 1,
        'Purchase': 15000,
        'Purchase_Time': 'Evening'
    },
    {
        'Gender': 'F',
        'Age': '18-25',
        'Occupation': 10,
        'City_Category': 'B',
        'Stay_In_Current_City_Years': '1',
        'Marital_Status': 0,
        'Purchase': 8000,
        'Purchase_Time': 'Afternoon'
    },
    {
        'Gender': 'M',
        'Age': '46-50',
        'Occupation': 15,
        'City_Category': 'C',
        'Stay_In_Current_City_Years': '4+',
        'Marital_Status': 1,
        'Purchase': 20000,
        'Purchase_Time': 'Morning'
    }
]

models_dict = {
    'product': model_product,
    'payment': model_payment,
    'age': model_age
}

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*60}")
    print(f"CENÁRIO {i}")
    print(f"{'='*60}")
    
    results = black_friday_inference(models_dict, scenario)

# SALVAR MODELOS
import joblib
joblib.dump(model_product, 'bf_product_model.pkl')
joblib.dump(model_payment, 'bf_payment_model.pkl')
joblib.dump(model_age, 'bf_age_model.pkl')

print("\n" + "="*80)
print("MODELOS SALVOS:")
print("-"*80)
print("  - bf_product_model.pkl (Categoria de Produto)")
print("  - bf_payment_model.pkl (Método de Pagamento)")
print("  - bf_age_model.pkl (Faixa Etária)")
print("\nSISTEMA COMPLETO!")