# ============================================================================
# MÓDULO DE INFERÊNCIA - HEART FAILURE
# ============================================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Definir as colunas necessárias fora das funções (acessível globalmente)
COLUNAS_NECESSARIAS = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

print("="*80)
print("MODULO DE INFERENCIA - HEART FAILURE")
print("Sistema de Predicao de Risco Cardiaco")
print("="*80)

# ============================================================================
# 1. CARREGAR MODELO TREINADO
# ============================================================================
print("\n1. CARREGANDO MODELO TREINADO...")
print("-"*60)

# Carregar o modelo salvo
modelo = joblib.load('heart_failure_model.pkl')
print("Modelo carregado com sucesso!")
print(f"Tipo do modelo: {type(modelo.named_steps['classifier']).__name__}")

# ============================================================================
# 2. FUNCAO DE INFERENCIA
# ============================================================================

def inferencia_heart_failure(modelo, dados_paciente):
    """
    Funcao principal de inferencia
    """
    
    print("\n" + "="*70)
    print("SISTEMA DE PREDICAO DE RISCO CARDIACO")
    print("="*70)
    
    # ------------------------------------------------------------
    # PASSO 1: Preparar os dados do paciente
    # ------------------------------------------------------------
    print("\nPASSO 1: DADOS DO PACIENTE")
    print("-"*50)
    
    # Garantir que todas as colunas estao presentes
    for col in COLUNAS_NECESSARIAS:
        if col not in dados_paciente:
            raise ValueError(f"Coluna '{col}' faltando nos dados do paciente!")
    
    df_paciente = pd.DataFrame([dados_paciente])
    df_paciente = df_paciente[COLUNAS_NECESSARIAS]
    
    print(f"Idade: {dados_paciente['age']} anos")
    print(f"Sexo: {'Masculino' if dados_paciente['sex'] == 1 else 'Feminino'}")
    print(f"Fracao de Ejeção: {dados_paciente['ejection_fraction']}%")
    print(f"Creatinina Serica: {dados_paciente['serum_creatinine']} mg/dL")
    print(f" Sodio Serico: {dados_paciente['serum_sodium']} mEq/L")
    print(f"CPK: {dados_paciente['creatinine_phosphokinase']} mcg/L")
    print(f"Plaquetas: {dados_paciente['platelets']:.0f} kiloplatelets/mL")
    print(f"Tempo de Acompanhamento: {dados_paciente['time']} dias")
    
    # Condicoes medicas
    print("\nCONDICOES DO PACIENTE:")
    condicoes = []
    if dados_paciente['anaemia'] == 1:
        condicoes.append("Anemia")
    if dados_paciente['diabetes'] == 1:
        condicoes.append("Diabetes")
    if dados_paciente['high_blood_pressure'] == 1:
        condicoes.append("Hipertensao")
    if dados_paciente['smoking'] == 1:
        condicoes.append("Tabagismo")
    
    if condicoes:
        for c in condicoes:
            print(f"  - {c}")
    else:
        print("  Nenhuma condicao de risco")
    
    # ------------------------------------------------------------
    # PASSO 2: Predicao
    # ------------------------------------------------------------
    print("\nPASSO 2: PREDICAO DO MODELO")
    print("-"*50)
    
    # Fazer a predicao
    predicao = modelo.predict(df_paciente)[0]
    probabilidades = modelo.predict_proba(df_paciente)[0]
    
    grupo_predito = "OBITO" if predicao == 1 else "SOBREVIVENTE"
    prob_obito = probabilidades[1]
    prob_sobrevivencia = probabilidades[0]
    
    print(f"Grupo predito: {grupo_predito}")
    print(f"Probabilidade de Sobrevivencia: {prob_sobrevivencia:.2%}")
    print(f"Probabilidade de Obito: {prob_obito:.2%}")
    
    # ------------------------------------------------------------
    # PASSO 3: Grau de certeza
    # ------------------------------------------------------------
    print("\nPASSO 3: GRAU DE CERTEZA")
    print("-"*50)
    
    confianca = prob_obito if predicao == 1 else prob_sobrevivencia
    
    if confianca >= 0.90:
        nivel_confianca = "MUITO ALTO"
    elif confianca >= 0.75:
        nivel_confianca = "ALTO"
    elif confianca >= 0.60:
        nivel_confianca = "MODERADO"
    else:
        nivel_confianca = "BAIXO"
    
    print(f"Confianca: {confianca:.2%} - {nivel_confianca}")
    
    # ------------------------------------------------------------
    # PASSO 4: Fatores de risco
    # ------------------------------------------------------------
    print("\nPASSO 4: FATORES DE RISCO")
    print("-"*50)
    
    fatores_alerta = []
    
    if dados_paciente['ejection_fraction'] < 30:
        fatores_alerta.append("Fracao de ejeção CRITICA (<30%)")
    elif dados_paciente['ejection_fraction'] < 40:
        fatores_alerta.append("Fracao de ejeção REDUZIDA (<40%)")
    
    if dados_paciente['serum_creatinine'] > 1.5:
        fatores_alerta.append("Creatinina elevada (>1.5 mg/dL)")
    elif dados_paciente['serum_creatinine'] > 1.2:
        fatores_alerta.append("Creatinina limitrofe")
    
    if dados_paciente['serum_sodium'] < 135:
        fatores_alerta.append("Sodio baixo (<135 mEq/L)")
    
    if dados_paciente['age'] > 70:
        fatores_alerta.append("Idade avancada (>70 anos)")
    
    if dados_paciente['creatinine_phosphokinase'] > 300:
        fatores_alerta.append("CPK elevado (>300 mcg/L)")
    
    if fatores_alerta:
        print("FATORES DE ALERTA:")
        for f in fatores_alerta:
            print(f"  - {f}")
    else:
        print("Nenhum fator de alerta critico")
    
    # ------------------------------------------------------------
    # PASSO 5: Recomendacoes
    # ------------------------------------------------------------
    print("\nPASSO 5: RECOMENDACOES")
    print("-"*50)
    
    if predicao == 1:
        print("ALERTA: ALTO RISCO")
        print("- Monitoramento intensivo")
        print("- Avaliacao de internacao")
        print("- Revisao da medicacao")
    else:
        print("BAIXO RISCO")
        print("- Acompanhamento ambulatorial")
        print("- Manter habitos saudaveis")
        print("- Reavaliacao em 3-6 meses")
    
    # Resumo final
    print("\nRESUMO:")
    print(f"Grupo: {grupo_predito}")
    print(f"Confianca: {confianca:.1%}")
    print(f"P(Sobrevivencia): {prob_sobrevivencia:.1%}")
    print(f"P(Obito): {prob_obito:.1%}")
    
    return {
        'grupo': 'Obito' if predicao == 1 else 'Sobrevivente',
        'predicao_numerica': int(predicao),
        'probabilidade_sobrevivencia': prob_sobrevivencia,
        'probabilidade_obito': prob_obito,
        'confianca': confianca,
        'nivel_confianca': nivel_confianca,
        'fatores_alerta': fatores_alerta,
        'dados_paciente': dados_paciente
    }


# ============================================================================
# 3. FUNCAO PARA INFERENCIA EM LOTE
# ============================================================================

def inferencia_em_lote(modelo, lista_pacientes):
    """
    Processa multiplos pacientes de uma vez
    """
    resultados = []
    
    for i, paciente in enumerate(lista_pacientes, 1):
        df_pac = pd.DataFrame([paciente])[COLUNAS_NECESSARIAS]
        pred = modelo.predict(df_pac)[0]
        proba = modelo.predict_proba(df_pac)[0]
        
        resultados.append({
            'Paciente': f'Paciente {i}',
            'Idade': paciente['age'],
            'FE': paciente['ejection_fraction'],
            'Grupo': 'Obito' if pred == 1 else 'Sobrevivente',
            'P(Obito)': proba[1],
            'Confianca': max(proba)
        })
    
    return pd.DataFrame(resultados)


# ============================================================================
# 4. TESTES DE INFERENCIA
# ============================================================================

print("\n" + "="*80)
print("TESTES DO SISTEMA DE INFERENCIA")
print("="*80)

# Paciente 1: Alto risco
print("\n" + "#"*70)
print("CASO 1: PACIENTE ALTO RISCO")
print("#"*70)

paciente_alto_risco = {
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

resultado_1 = inferencia_heart_failure(modelo, paciente_alto_risco)

# Paciente 2: Baixo risco
print("\n" + "#"*70)
print("CASO 2: PACIENTE BAIXO RISCO")
print("#"*70)

paciente_baixo_risco = {
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

resultado_2 = inferencia_heart_failure(modelo, paciente_baixo_risco)

# Processamento em lote
print("\n" + "="*80)
print("INFERENCIA EM LOTE")
print("="*80)

lote_pacientes = [
    paciente_alto_risco,
    paciente_baixo_risco,
    {
        'age': 45,
        'anaemia': 0,
        'creatinine_phosphokinase': 100,
        'diabetes': 0,
        'ejection_fraction': 65,
        'high_blood_pressure': 0,
        'platelets': 400000,
        'serum_creatinine': 0.7,
        'serum_sodium': 142,
        'sex': 0,
        'smoking': 0,
        'time': 200
    }
]

df_lote = inferencia_em_lote(modelo, lote_pacientes)
print("\nResultado do processamento em lote:")
print(df_lote.to_string(index=False))

print("\n" + "="*80)
print("MODULO DE INFERENCIA CONCLUIDO")
print("="*80)