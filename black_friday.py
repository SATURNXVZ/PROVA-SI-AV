# QUESTAO 3 - Black Friday Sales - Sistema de Classificacao
# Dataset: https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARREGAMENTO
# ============================================================================
df = pd.read_csv(r"C:\Users\User\Documents\PROVA-SI-AV\retail_black_friday_sales_100k.csv")

print(f"Registros: {len(df)} | Colunas: {df.shape[1]}")
print(f"Colunas: {list(df.columns)}")
print(f"\nValores ausentes:\n{df.isnull().sum()}")

# ============================================================================
# 2. PRE-PROCESSAMENTO
# ============================================================================
# Remover colunas que nao sao features de contexto de venda
# Mantemos original_price, que ajuda a diferenciar categorias e valores
drop_cols = ['transaction_id', 'customer_id', 'product_id', 'purchase_date',
             'final_price', 'purchase_amount']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Codificar variaveis categoricas de entrada (incluindo as que serao usadas como feature)
cat_features = ['gender', 'city', 'customer_segment', 'product_category', 'payment_method']
le_map = {}
for col in cat_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

# Features base (presentes em todos os modelos)
feature_cols_base = ['gender', 'city', 'customer_segment', 'discount_pct',
                     'quantity', 'purchase_hour', 'is_weekend', 'is_black_friday',
                     'original_price']

# Conjuntos de features especificos para cada tarefa
X_cat = df[feature_cols_base]                                          # categoria de produto
X_pag = df[feature_cols_base + ['product_category']]                   # forma de pagamento
X_age = df[feature_cols_base + ['product_category', 'payment_method']] # faixa etaria

print(f"\nFeatures usadas para Categoria : {list(X_cat.columns)}")
print(f"Features usadas para Pagamento  : {list(X_pag.columns)}")
print(f"Features usadas para Idade      : {list(X_age.columns)}")

# ============================================================================
# 3. FUNCAO GENERICA DE TREINO E AVALIACAO
# ============================================================================
def treinar_avaliar(X, y, nome_alvo):
    """Treina Random Forest, avalia e retorna modelo e encoder do alvo."""

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42,
                                       class_weight='balanced'))
    ])

    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_weighted').mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    classes = le.classes_
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"ALVO: {nome_alvo.upper()}")
    print(f"{'='*60}")
    print(f"Acuracia Global : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score (wtd)  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score CV     : {cv_f1:.4f}")

    # Sensibilidade e Especificidade por classe
    print(f"\nMetricas por classe:")
    print(f"{'Classe':<22} {'Suporte':>8} {'Acuracia':>9} {'Sensib.':>9} {'Especif.':>9} {'F1':>8}")
    print("-" * 67)
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensib  = tp / (tp + fn) if (tp + fn) > 0 else 0
        especif = tn / (tn + fp) if (tn + fp) > 0 else 0
        suporte = cm[i, :].sum()
        acc_cls = tp / suporte if suporte > 0 else 0
        f1_cls  = f1_score(y_test, y_pred, labels=[i], average='macro', zero_division=0)

        print(f"{cls:<22} {suporte:>8} {acc_cls:>9.3f} {sensib:>9.3f} {especif:>9.3f} {f1_cls:>8.3f}")

    print(f"\nMatriz de Confusao (linhas=real, colunas=predito):")
    header = f"{'':22}" + "  ".join(f"{c[:7]:>7}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:<22}" + "  ".join(f"{v:>7}" for v in row))

    return pipe, le


# ============================================================================
# 4. TREINAR OS TRES CLASSIFICADORES (agora com X especifico)
# ============================================================================
modelo_cat, le_cat = treinar_avaliar(X_cat, df['product_category'], 'Categoria de Produto')
modelo_pag, le_pag = treinar_avaliar(X_pag, df['payment_method'],   'Forma de Pagamento')
modelo_age, le_age = treinar_avaliar(X_age, df['age_group'],        'Faixa Etaria')


# ============================================================================
# 5. MODULO DE INFERENCIA
# ============================================================================
def inferir(contexto: dict):
    """
    Recebe o contexto de uma venda e retorna as tres previsoes com grau de certeza.

    Parametros esperados:
      gender, city, customer_segment, discount_pct, quantity, purchase_hour,
      is_weekend, is_black_friday, original_price
      product_category (opcional, usado para pagamento e idade)
      payment_method   (opcional, usado para idade)
    """
    row = contexto.copy()
    for col, le in le_map.items():
        if col in row:
            try:
                row[col] = le.transform([str(row[col])])[0]
            except ValueError:
                row[col] = -1

    df_in = pd.DataFrame([row])

    def prever(modelo, le_alvo, nome, feature_list):
        df_modelo = df_in[feature_list]
        proba  = modelo.predict_proba(df_modelo)[0]
        idx    = np.argmax(proba)
        classe = le_alvo.classes_[idx]
        conf   = proba[idx]
        top3   = sorted(zip(le_alvo.classes_, proba), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {nome}: {classe}  (certeza: {conf:.1%})")
        for cls, p in top3:
            if p > 0.05:
                print(f"      - {cls}: {p:.1%}")

    print("\n" + "="*50)
    print("RESULTADO DA INFERENCIA")
    print("="*50)
    for k, v in contexto.items():
        print(f"  {k}: {v}")
    print()
    prever(modelo_cat, le_cat, "Categoria do Produto", feature_cols_base)
    prever(modelo_pag, le_pag, "Forma de Pagamento ", feature_cols_base + ['product_category'])
    prever(modelo_age, le_age, "Faixa Etaria       ", feature_cols_base + ['product_category', 'payment_method'])
    print()


# ============================================================================
# 6. DEMONSTRACAO
# ============================================================================
print("\n\n" + "="*60)
print("DEMONSTRACAO DO SISTEMA INTELIGENTE")
print("="*60)

# Caso 1: Cliente fiel, sem Black Friday
inferir({
    'gender': 'Male', 'city': 'San Francisco', 'customer_segment': 'Loyal',
    'discount_pct': 35, 'quantity': 1, 'original_price': 80,
    'purchase_hour': 0, 'is_weekend': 0, 'is_black_friday': 0,
    'product_category': 'Electronics', 'payment_method': 'Credit Card'
})

# Caso 2: Cliente novo, Black Friday, hora de pico
inferir({
    'gender': 'Female', 'city': 'Dallas', 'customer_segment': 'New',
    'discount_pct': 30, 'quantity': 2, 'original_price': 120,
    'purchase_hour': 16, 'is_weekend': 0, 'is_black_friday': 1,
    'product_category': 'Clothing', 'payment_method': 'Mobile Wallet'
})

# Caso 3: Fim de semana, grande desconto, multiplos itens
inferir({
    'gender': 'Female', 'city': 'Chicago', 'customer_segment': 'Regular',
    'discount_pct': 50, 'quantity': 3, 'original_price': 150,
    'purchase_hour': 20, 'is_weekend': 1, 'is_black_friday': 1,
    'product_category': 'Books', 'payment_method': 'Debit Card'
})

# ============================================================================
# 7. SALVAR MODELOS
# ============================================================================
import joblib
import os

joblib.dump(modelo_cat, 'bf_produto.pkl')
joblib.dump(modelo_pag, 'bf_pagamento.pkl')
joblib.dump(modelo_age, 'bf_idade.pkl')

print("\nModelos salvos:")
print(f"  bf_produto.pkl   ({os.path.getsize('bf_produto.pkl')} bytes)")
print(f"  bf_pagamento.pkl ({os.path.getsize('bf_pagamento.pkl')} bytes)")
print(f"  bf_idade.pkl     ({os.path.getsize('bf_idade.pkl')} bytes)")