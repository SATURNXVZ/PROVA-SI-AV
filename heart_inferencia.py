#MODULO DE INFERENCIA - HEART FAILURE
import pandas as pd
import numpy as np
import joblib

COLUNAS_NECESSARIAS = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
]

print("MODULO DE INFERENCIA - HEART FAILURE")

#1- CARREGAR MODELO TREINADO

print("\n1. CARREGANDO MODELO TREINADO...")

modelo = joblib.load('heart_failure_model.pkl')
print("Modelo carregado com sucesso!")
print(f"Tipo do modelo: {type(modelo.named_steps['classifier']).__name__}")
print(f"Features esperadas: {len(COLUNAS_NECESSARIAS)}")

print("\nFeatures que o modelo utiliza:")
for i, f in enumerate(COLUNAS_NECESSARIAS, 1):
    print(f"  {i:2d}. {f}")


# 2. FUNCAO DE INFERENCIA

def inferencia_heart_failure(modelo, dados_paciente):
    print("SISTEMA DE PREDICAO DE RISCO CARDIACO")
    
    #Validar e preparar dados
    
    colunas_faltando = [c for c in COLUNAS_NECESSARIAS if c not in dados_paciente]
    if colunas_faltando:
        raise ValueError(f"Colunas faltando: {colunas_faltando}")
    
    
    #criar Dataframe
    df_paciente = pd.DataFrame([dados_paciente])
    df_paciente = df_paciente[COLUNAS_NECESSARIAS]
    
    #exibir dados do paciente
    print(f"Idade: {dados_paciente['age']} anos")
    print(f"Sexo: {'Masculino' if dados_paciente['sex'] == 1 else 'Feminino'}")
    print(f"Fracao de Ejecao: {dados_paciente['ejection_fraction']}%")
    print(f"Creatinina Serica: {dados_paciente['serum_creatinine']} mg/dL")
    print(f"Sodio Serico: {dados_paciente['serum_sodium']} mEq/L")
    print(f"CPK: {dados_paciente['creatinine_phosphokinase']} mcg/L")
    print(f"Plaquetas: {dados_paciente['platelets']:.0f} kiloplatelets/mL")
    
    #condicoes medicas
    print("\nCondicoes do paciente:")
    condicoes = []
    if dados_paciente.get('anaemia', 0) == 1:
        condicoes.append("Anemia (diminuicao de globulos vermelhos)")
    if dados_paciente.get('diabetes', 0) == 1:
        condicoes.append("Diabetes")
    if dados_paciente.get('high_blood_pressure', 0) == 1:
        condicoes.append("Hipertensao arterial")
    if dados_paciente.get('smoking', 0) == 1:
        condicoes.append("Tabagismo")
    
    if condicoes:
        for c in condicoes:
            print(f"  - {c}")
    else:
        print("Nenhuma condicao de risco reportada")
    
    #predicao
    predicao = modelo.predict(df_paciente)[0]
    probabilidades = modelo.predict_proba(df_paciente)[0]
    
    grupo_predito = "OBITO" if predicao == 1 else "SOBREVIVENTE"
    prob_obito = probabilidades[1]
    prob_sobrevivencia = probabilidades[0]
    
    print(f"Grupo predito: {grupo_predito}")
    print(f"Probabilidade de Sobrevivencia: {prob_sobrevivencia:.2%}")
    print(f"Probabilidade de Obito: {prob_obito:.2%}")
    
    # grau de certeza
    confianca = prob_obito if predicao == 1 else prob_sobrevivencia
    
    if confianca >= 0.90:
        nivel = "MUITO ALTO"
    elif confianca >= 0.75:
        nivel = "ALTO"
    elif confianca >= 0.60:
        nivel = "MODERADO"
    elif confianca >= 0.50:
        nivel = "BAIXO"
    else:
        nivel = "MUITO BAIXO"
    
    print(f"Confianca na predicao: {confianca:.2%} ({nivel})")
    
    #fatores de risco
    print("\nPASSO 4: ANALISE DE FATORES DE RISCO")
    print("-"*50)
    
    fatores_alerta = []
    
    #fracao de ejecao
    fe = dados_paciente['ejection_fraction']
    if fe < 25:
        fatores_alerta.append("Fracao de ejecao CRITICA (<25%) - Risco de choque cardiogenico")
    elif fe < 30:
        fatores_alerta.append("Fracao de ejecao MUITO BAIXA (<30%) - IC grave")
    elif fe < 40:
        fatores_alerta.append("Fracao de ejecao REDUZIDA (<40%) - Disfuncao ventricular")
    
    #creatinina
    creat = dados_paciente['serum_creatinine']
    if creat > 2.0:
        fatores_alerta.append("Creatinina GRAVE (>2.0 mg/dL) - Insuficiencia renal significativa")
    elif creat > 1.5:
        fatores_alerta.append("Creatinina ELEVADA (>1.5 mg/dL) - Disfuncao renal moderada")
    elif creat > 1.2:
        fatores_alerta.append("Creatinina LIMITROFE (>1.2 mg/dL) - Monitorar funcao renal")
    
    #sodio
    sodio = dados_paciente['serum_sodium']
    if sodio < 130:
        fatores_alerta.append("Hiponatremia GRAVE (<130 mEq/L) - Risco neurologico")
    elif sodio < 135:
        fatores_alerta.append("Hiponatremia MODERADA (<135 mEq/L) - Pior prognostico em IC")
    
    #idade
    idade = dados_paciente['age']
    if idade > 80:
        fatores_alerta.append("Idade AVANCADA (>80 anos) - Maior vulnerabilidade")
    elif idade > 70:
        fatores_alerta.append("Idade ELEVADA (>70 anos) - Fator de risco cardiovascular")
    
    #cpk
    cpk = dados_paciente['creatinine_phosphokinase']
    if cpk > 500:
        fatores_alerta.append("CPK MUITO ELEVADO (>500 mcg/L) - Investigar dano cardiaco agudo")
    elif cpk > 300:
        fatores_alerta.append("CPK ELEVADO (>300 mcg/L) - Possivel dano muscular/cardiaco")
    
    if fatores_alerta:
        print("FATORES DE ALERTA IDENTIFICADOS:")
        for f in fatores_alerta:
            print(f"  - {f}")
    else:
        print("Nenhum fator de alerta critico identificado.")
    
    # recomendacoes
    if predicao == 1:
        print("""
ALERTA: ALTO RISCO DE EVENTO CARDIACO FATAL

Recomendacoes urgentes:
  - Monitoramento intensivo imediato
  - Avaliar necessidade de internacao
  - Revisao urgente da medicacao
  - Considerar dispositivos de assistencia ventricular
  - Avaliar criterios para transplante cardiaco
""")
        
        if fe < 25:
            print("  URGENTE: FE < 25% - Avaliar transplante com prioridade.")
        if creat > 2.0:
            print("  URGENTE: Disfuncao renal grave - Ajustar medicacoes nefrotoxicas.")
            
    else:
        print("""
BAIXO RISCO DE EVENTO CARDIACO FATAL

Recomendacoes de acompanhamento:
  - Manter acompanhamento ambulatorial regular
  - Continuar medicacao conforme prescricao
  - Reavaliacao cardiologica em 3-6 meses
  - Manter habitos saudaveis
  - Controle de fatores de risco
""")
    
    # ------------------------------------------------------------
    # PASSO 6: Resumo final
    # ------------------------------------------------------------
    print("RESUMO FINAL DA CLASSIFICACAO")
    print("-"*50)
    print(f"Grupo: {grupo_predito}")
    print(f"Confianca: {confianca:.1%} ({nivel})")
    print(f"P(Sobrevivencia): {prob_sobrevivencia:.1%}")
    print(f"P(Obito): {prob_obito:.1%}")
    
    # Retornar resultados
    return {
        'grupo': 'Obito' if predicao == 1 else 'Sobrevivente',
        'predicao': int(predicao),
        'prob_sobrevivencia': prob_sobrevivencia,
        'prob_obito': prob_obito,
        'confianca': confianca,
        'nivel_confianca': nivel,
        'fatores_alerta': fatores_alerta
    }


# ============================================================================
# 3. FUNCAO PARA INFERENCIA EM LOTE
# ============================================================================

def inferencia_em_lote(modelo, lista_pacientes):
    #processa multiplos pacientes de uma vez.
    
    resultados = []
    
    for i, paciente in enumerate(lista_pacientes, 1):
        # Remover 'time' se presente
        if 'time' in paciente:
            paciente = {k: v for k, v in paciente.items() if k != 'time'}
        
        df_pac = pd.DataFrame([paciente])[COLUNAS_NECESSARIAS]
        pred = modelo.predict(df_pac)[0]
        proba = modelo.predict_proba(df_pac)[0]
        
        resultados.append({
            'Paciente': f'Paciente {i}',
            'Idade': paciente['age'],
            'FE (%)': paciente['ejection_fraction'],
            'Grupo': 'OBITO' if pred == 1 else 'SOBREVIVENTE',
            'P(Obito)': f"{proba[1]:.1%}",
            'Confianca': f"{max(proba):.1%}"
        })
    
    return pd.DataFrame(resultados)


# 
#4- DEMONSTRACAO DO SISTEMA DE INFERENCIA
#caso 1: paciente alto risco

print("# CASO 1: PACIENTE COM PERFIL DE ALTO RISCO")
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
    'smoking': 1
}

resultado_1 = inferencia_heart_failure(modelo, paciente_alto_risco)

# caso 2: Paciente baixo risco
print("# CASO 2: PACIENTE COM PERFIL DE BAIXO RISCO")

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
    'smoking': 0
}

resultado_2 = inferencia_heart_failure(modelo, paciente_baixo_risco)

#caso 3: paciente limitofre
print("# CASO 3: PACIENTE COM PERFIL LIMITROFE")

paciente_limitrofe = {
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
    'smoking': 0
}

resultado_3 = inferencia_heart_failure(modelo, paciente_limitrofe)

#5- COMPARACAO DOS RESULTADOS

print("COMPARACAO DOS RESULTADOS DAS INFERENCIAS")

print(f"\n{'Caso':<35} {'Grupo':<15} {'Confianca':<12} {'P(Obito)':<12}")
print("-"*75)

casos = [
    ("Caso 1 - Alto Risco", resultado_1),
    ("Caso 2 - Baixo Risco", resultado_2),
    ("Caso 3 - Limitrofe", resultado_3),
]

for nome, res in casos:
    print(f"{nome:<35} {res['grupo']:<15} {res['confianca']:<12.1%} {res['prob_obito']:<12.1%}")


# 6- INFERENCIA EM LOTE
print("PROCESSAMENTO EM LOTE")
lote = [paciente_alto_risco, paciente_baixo_risco, paciente_limitrofe]
df_resultado = inferencia_em_lote(modelo, lote)

print("\nResultados do processamento em lote:")
print(df_resultado.to_string(index=False))

print("MODULO DE INFERENCIA CONCLUIDO COM SUCESSO")