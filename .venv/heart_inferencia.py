# ============================================================================
# MÓDULO DE INFERÊNCIA - HEART FAILURE
# ============================================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

print("="*80)
print("MÓDULO DE INFERÊNCIA - HEART FAILURE")
print("Sistema Inteligente de Predição de Risco Cardíaco")
print("="*80)

# ============================================================================
# 1. CARREGAR MODELO TREINADO
# ============================================================================
print("\n1. CARREGANDO MODELO TREINADO...")
print("-"*60)

# Carregar o modelo salvo
modelo = joblib.load('heart_failure_model.pkl')
print("✓ Modelo carregado com sucesso!")
print(f"  Tipo do modelo: {type(modelo.named_steps['classifier']).__name__}")

# ============================================================================
# 2. FUNÇÃO DE INFERÊNCIA
# ============================================================================

def inferencia_heart_failure(modelo, dados_paciente):
    """
    FUNÇÃO PRINCIPAL DE INFERÊNCIA
    
    Esta função recebe um paciente desconhecido e:
    1. Processa os dados do paciente
    2. Faz a predição usando o modelo treinado
    3. Indica a qual grupo o paciente pertence (Sobrevivente ou Óbito)
    4. Mostra o grau de certeza da predição
    5. Fornece recomendações baseadas no resultado
    
    Parâmetros:
    -----------
    modelo : Pipeline
        Modelo treinado (pré-processador + Random Forest)
    dados_paciente : dict
        Dicionário com os 12 atributos do paciente
    
    Retorna:
    --------
    dict : Resultado completo da inferência
    """
    
    print("\n" + "="*70)
    print("🏥 SISTEMA INTELIGENTE DE PREDIÇÃO DE RISCO CARDÍACO")
    print("="*70)
    
    # ------------------------------------------------------------
    # PASSO 1: Preparar os dados do paciente
    # ------------------------------------------------------------
    print("\n📋 PASSO 1: DADOS DO PACIENTE RECEBIDOS")
    print("-"*50)
    
    # Criar DataFrame com os dados (mantendo a ordem das features)
    colunas_necessarias = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]
    
    # Garantir que todas as colunas estão presentes
    for col in colunas_necessarias:
        if col not in dados_paciente:
            raise ValueError(f"Coluna '{col}' faltando nos dados do paciente!")
    
    df_paciente = pd.DataFrame([dados_paciente])
    df_paciente = df_paciente[colunas_necessarias]
    
    # Exibir dados do paciente de forma organizada
    print(f"  Idade: {dados_paciente['age']} anos")
    print(f"  Sexo: {'Masculino ♂️' if dados_paciente['sex'] == 1 else 'Feminino ♀️'}")
    print(f"  Fração de Ejeção: {dados_paciente['ejection_fraction']}%")
    print(f"  Creatinina Sérica: {dados_paciente['serum_creatinine']} mg/dL")
    print(f"  Sódio Sérico: {dados_paciente['serum_sodium']} mEq/L")
    print(f"  CPK: {dados_paciente['creatinine_phosphokinase']} mcg/L")
    print(f"  Plaquetas: {dados_paciente['platelets']:.0f} kiloplatelets/mL")
    print(f"  Tempo de Acompanhamento: {dados_paciente['time']} dias")
    
    # Condições médicas (variáveis binárias)
    print(f"\n  CONDIÇÕES DO PACIENTE:")
    condicoes = []
    if dados_paciente['anaemia'] == 1:
        condicoes.append("🔴 Anemia (diminuição de glóbulos vermelhos)")
    if dados_paciente['diabetes'] == 1:
        condicoes.append("🔴 Diabetes")
    if dados_paciente['high_blood_pressure'] == 1:
        condicoes.append("🔴 Hipertensão arterial")
    if dados_paciente['smoking'] == 1:
        condicoes.append("🔴 Tabagismo")
    
    if condicoes:
        for c in condicoes:
            print(f"    {c}")
    else:
        print("    ✅ Nenhuma condição de risco reportada")
    
    # ------------------------------------------------------------
    # PASSO 2: Processamento dos dados (automático pelo pipeline)
    # ------------------------------------------------------------
    print(f"\n⚙️  PASSO 2: PRÉ-PROCESSAMENTO AUTOMÁTICO")
    print("-"*50)
    print("  • Variáveis contínuas: Normalizadas (StandardScaler)")
    print("  • Variáveis binárias: Mantidas como 0/1")
    print("  • Features processadas e enviadas ao classificador")
    
    # ------------------------------------------------------------
    # PASSO 3: Predição
    # ------------------------------------------------------------
    print(f"\n🤖 PASSO 3: PREDIÇÃO DO MODELO")
    print("-"*50)
    
    # Fazer a predição
    predicao = modelo.predict(df_paciente)[0]  # 0 = Sobrevivente, 1 = Óbito
    probabilidades = modelo.predict_proba(df_paciente)[0]  # [prob_sobrevivencia, prob_obito]
    
    # Determinar o grupo
    grupo_predito = "ÓBITO ⚠️" if predicao == 1 else "SOBREVIVENTE ✅"
    prob_obito = probabilidades[1]
    prob_sobrevivencia = probabilidades[0]
    
    print(f"  Classe predita: {grupo_predito}")
    print(f"  Probabilidade de Sobrevivência: {prob_sobrevivencia:.2%}")
    print(f"  Probabilidade de Óbito: {prob_obito:.2%}")
    
    # ------------------------------------------------------------
    # PASSO 4: Grau de certeza
    # ------------------------------------------------------------
    print(f"\n📊 PASSO 4: GRAU DE CERTEZA DA PREDIÇÃO")
    print("-"*50)
    
    # O grau de certeza é a probabilidade da classe predita
    confianca = prob_obito if predicao == 1 else prob_sobrevivencia
    
    # Classificar o nível de confiança
    if confianca >= 0.90:
        nivel_confianca = "MUITO ALTO 🟢"
        barra = "██████████"
    elif confianca >= 0.75:
        nivel_confianca = "ALTO 🟢"
        barra = "████████░░"
    elif confianca >= 0.60:
        nivel_confianca = "MODERADO 🟡"
        barra = "██████░░░░"
    elif confianca >= 0.50:
        nivel_confianca = "BAIXO 🟠"
        barra = "████░░░░░░"
    else:
        nivel_confianca = "MUITO BAIXO 🔴"
        barra = "██░░░░░░░░"
    
    print(f"  Grau de Certeza: {confianca:.2%} - {nivel_confianca}")
    print(f"  Visualização: [{barra}] {confianca:.0%}")
    
    # ------------------------------------------------------------
    # PASSO 5: Análise dos fatores de risco
    # ------------------------------------------------------------
    print(f"\n🔍 PASSO 5: ANÁLISE DOS FATORES DE RISCO")
    print("-"*50)
    
    fatores_alerta = []
    
    # Verificar cada fator de risco conhecido
    if dados_paciente['ejection_fraction'] < 30:
        fatores_alerta.append("⚠️  Fração de ejeção CRÍTICA (<30%) - Risco elevado de insuficiência cardíaca grave")
    elif dados_paciente['ejection_fraction'] < 40:
        fatores_alerta.append("⚠️  Fração de ejeção REDUZIDA (<40%) - Disfunção ventricular significativa")
    
    if dados_paciente['serum_creatinine'] > 1.5:
        fatores_alerta.append("⚠️  Creatinina sérica ELEVADA (>1.5 mg/dL) - Possível disfunção renal")
    elif dados_paciente['serum_creatinine'] > 1.2:
        fatores_alerta.append("⚠️  Creatinina sérica LIMÍTROFE - Monitorar função renal")
    
    if dados_paciente['serum_sodium'] < 135:
        fatores_alerta.append("⚠️  Sódio sérico BAIXO (<135 mEq/L) - Hiponatremia, risco de pior prognóstico")
    
    if dados_paciente['age'] > 70:
        fatores_alerta.append("⚠️  Idade AVANÇADA (>70 anos) - Maior vulnerabilidade cardiovascular")
    
    if dados_paciente['creatinine_phosphokinase'] > 300:
        fatores_alerta.append("⚠️  CPK ELEVADO (>300 mcg/L) - Possível dano muscular/cardíaco")
    
    if fatores_alerta:
        print("  FATORES DE ALERTA IDENTIFICADOS:")
        for f in fatores_alerta:
            print(f"    {f}")
    else:
        print("  ✅ Nenhum fator de alerta crítico identificado")
    
    # ------------------------------------------------------------
    # PASSO 6: Recomendações
    # ------------------------------------------------------------
    print(f"\n💊 PASSO 6: RECOMENDAÇÕES BASEADAS NO RESULTADO")
    print("-"*50)
    
    if predicao == 1:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  🚨 ALERTA: ALTO RISCO DE EVENTO CARDÍACO FATAL    ║
  ╚══════════════════════════════════════════════════════╝
  
  RECOMENDAÇÕES URGENTES:
  ┌──────────────────────────────────────────────────────┐
  │ ✓ Monitoramento intensivo imediato                   │
  │ ✓ Internação para estabilização hemodinâmica         │
  │ ✓ Revisão urgente da medicação                       │
  │ ✓ Considerar dispositivos de assistência ventricular │
  │ ✓ Avaliar critérios para transplante cardíaco        │
  │ ✓ Acompanhamento com equipe multidisciplinar         │
  └──────────────────────────────────────────────────────┘
        """)
        
        # Recomendações específicas
        if dados_paciente['ejection_fraction'] < 25:
            print("  ⚡ AÇÃO IMEDIATA: Fração de ejeção < 25%")
            print("     → Avaliar transplante cardíaco com urgência")
            print("     → Considerar balão intra-aórtico")
        
        if dados_paciente['serum_creatinine'] > 2.0:
            print("  ⚡ AÇÃO IMEDIATA: Disfunção renal significativa")
            print("     → Ajustar medicações nefrotóxicas")
            print("     → Considerar terapia de substituição renal")
            
    else:
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║  ✅ BAIXO RISCO DE EVENTO CARDÍACO FATAL           ║
  ╚══════════════════════════════════════════════════════╝
  
  RECOMENDAÇÕES DE ACOMPANHAMENTO:
  ┌──────────────────────────────────────────────────────┐
  │ ✓ Manter acompanhamento ambulatorial regular         │
  │ ✓ Continuar medicação conforme prescrição            │
  │ ✓ Reavaliação cardiológica em 3-6 meses              │
  │ ✓ Manter hábitos saudáveis                           │
  │ ✓ Controle rigoroso de fatores de risco              │
  │ ✓ Atividade física conforme orientação médica        │
  └──────────────────────────────────────────────────────┘
        """)
    
    # ------------------------------------------------------------
    # PASSO 7: Resumo final
    # ------------------------------------------------------------
    print(f"\n📝 RESUMO FINAL DA INFERÊNCIA")
    print("-"*50)
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │ RESULTADO DA CLASSIFICAÇÃO                          │
  ├─────────────────────────────────────────────────────┤
  │ Grupo: {grupo_predito:<43} │
  │ Confiança: {confianca:.1%} ({nivel_confianca}){'':>22} │
  │ Prob. Sobrevivência: {prob_sobrevivencia:.1%}{'':>24} │
  │ Prob. Óbito: {prob_obito:.1%}{'':>31} │
  └─────────────────────────────────────────────────────┘
    """)
    
    # Retornar resultados como dicionário
    return {
        'grupo': 'Óbito' if predicao == 1 else 'Sobrevivente',
        'predicao_numerica': int(predicao),
        'probabilidade_sobrevivencia': prob_sobrevivencia,
        'probabilidade_obito': prob_obito,
        'confianca': confianca,
        'nivel_confianca': nivel_confianca,
        'fatores_alerta': fatores_alerta,
        'dados_paciente': dados_paciente
    }


# ============================================================================
# 3. TESTES DE INFERÊNCIA COM DIFERENTES PACIENTES
# ============================================================================

print("\n" + "="*80)
print("TESTES DO SISTEMA DE INFERÊNCIA")
print("="*80)

# ==============================================================
# PACIENTE 1: Perfil de ALTO RISCO
# ==============================================================
print("\n" + "#"*70)
print("# CASO 1: PACIENTE COM PERFIL DE ALTO RISCO")
print("#"*70)

paciente_alto_risco = {
    'age': 75,                          # Idoso
    'anaemia': 1,                       # Tem anemia
    'creatinine_phosphokinase': 582,    # CPK muito elevado
    'diabetes': 1,                      # Diabético
    'ejection_fraction': 20,           # Fração de ejeção muito baixa
    'high_blood_pressure': 1,          # Hipertenso
    'platelets': 265000,               # Plaquetas normais
    'serum_creatinine': 2.5,           # Creatinina elevada (disfunção renal)
    'serum_sodium': 130,               # Sódio baixo (hiponatremia)
    'sex': 1,                          # Masculino
    'smoking': 1,                      # Fumante
    'time': 30                         # Pouco tempo de acompanhamento
}

resultado_1 = inferencia_heart_failure(modelo, paciente_alto_risco)


# ==============================================================
# PACIENTE 2: Perfil de BAIXO RISCO
# ==============================================================
print("\n" + "#"*70)
print("# CASO 2: PACIENTE COM PERFIL DE BAIXO RISCO")
print("#"*70)

paciente_baixo_risco = {
    'age': 52,                          # Meia-idade
    'anaemia': 0,                       # Sem anemia
    'creatinine_phosphokinase': 120,    # CPK normal
    'diabetes': 0,                      # Não diabético
    'ejection_fraction': 60,           # Fração de ejeção normal
    'high_blood_pressure': 0,          # Normotenso
    'platelets': 350000,               # Plaquetas normais
    'serum_creatinine': 0.8,           # Creatinina normal
    'serum_sodium': 140,               # Sódio normal
    'sex': 0,                          # Feminino
    'smoking': 0,                      # Não fumante
    'time': 150                        # Longo acompanhamento
}

resultado_2 = inferencia_heart_failure(modelo, paciente_baixo_risco)


# ==============================================================
# PACIENTE 3: Caso LIMÍTROFE/DUVIDOSO
# ==============================================================
print("\n" + "#"*70)
print("# CASO 3: PACIENTE COM PERFIL LIMÍTROFE")
print("#"*70)

paciente_limitrofe = {
    'age': 65,                          # Idoso jovem
    'anaemia': 0,                       # Sem anemia
    'creatinine_phosphokinase': 250,    # CPK levemente elevado
    'diabetes': 1,                      # Diabético
    'ejection_fraction': 38,           # Fração de ejeção limítrofe
    'high_blood_pressure': 1,          # Hipertenso
    'platelets': 280000,               # Plaquetas normais
    'serum_creatinine': 1.3,           # Creatinina limítrofe
    'serum_sodium': 136,               # Sódio levemente baixo
    'sex': 1,                          # Masculino
    'smoking': 0,                      # Não fumante
    'time': 90                         # Acompanhamento médio
}

resultado_3 = inferencia_heart_failure(modelo, paciente_limitrofe)


# ==============================================================
# PACIENTE 4: Jovem com condição cardíaca grave
# ==============================================================
print("\n" + "#"*70)
print("# CASO 4: PACIENTE JOVEM COM CONDIÇÃO CARDÍACA GRAVE")
print("#"*70)

paciente_jovem_grave = {
    'age': 35,                          # Jovem
    'anaemia': 0,                       # Sem anemia
    'creatinine_phosphokinase': 780,    # CPK muito elevado
    'diabetes': 0,                      # Não diabético
    'ejection_fraction': 25,           # Fração de ejeção muito baixa
    'high_blood_pressure': 0,          # Normotenso
    'platelets': 300000,               # Plaquetas normais
    'serum_creatinine': 1.1,           # Creatinina normal
    'serum_sodium': 137,               # Sódio normal
    'sex': 1,                          # Masculino
    'smoking': 0,                      # Não fumante
    'time': 60                         # Acompanhamento curto
}

resultado_4 = inferencia_heart_failure(modelo, paciente_jovem_grave)


# ============================================================================
# 4. COMPARAÇÃO DOS RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("COMPARAÇÃO DOS RESULTADOS DAS INFERÊNCIAS")
print("="*80)

print(f"\n{'Caso':<30} {'Grupo':<20} {'Confiança':<15} {'P(Óbito)':<15}")
print("-"*80)

casos = [
    ("Paciente 1 - Alto Risco", resultado_1),
    ("Paciente 2 - Baixo Risco", resultado_2),
    ("Paciente 3 - Limítrofe", resultado_3),
    ("Paciente 4 - Jovem Grave", resultado_4)
]

for nome, res in casos:
    print(f"{nome:<30} {res['grupo']:<20} {res['confianca']:<15.1%} {res['probabilidade_obito']:<15.1%}")

# ============================================================================
# 5. FUNÇÃO PARA INFERÊNCIA EM LOTE (MÚLTIPLOS PACIENTES)
# ============================================================================
print(f"\n{'='*80}")
print("INFERÊNCIA EM LOTE - PROCESSANDO MÚLTIPLOS PACIENTES")
print("="*80)

def inferencia_em_lote(modelo, lista_pacientes):
    """
    Processa múltiplos pacientes de uma vez
    
    Parâmetros:
    -----------
    modelo : Pipeline
        Modelo treinado
    lista_pacientes : list of dict
        Lista com dados de vários pacientes
    
    Retorna:
    --------
    DataFrame com resultados consolidados
    """
    
    resultados = []
    
    for i, paciente in enumerate(lista_pacientes, 1):
        df_pac = pd.DataFrame([paciente])[colunas_necessarias]
        pred = modelo.predict(df_pac)[0]
        proba = modelo.predict_proba(df_pac)[0]
        
        resultados.append({
            'Paciente': f'Paciente {i}',
            'Idade': paciente['age'],
            'FE': paciente['ejection_fraction'],
            'Grupo Predito': 'Óbito' if pred == 1 else 'Sobrevivente',
            'P(Óbito)': proba[1],
            'Confiança': max(proba),
            'Fatores Risco': sum([paciente['anaemia'], paciente['diabetes'], 
                                  paciente['high_blood_pressure'], paciente['smoking']])
        })
    
    return pd.DataFrame(resultados)

# Criar lote de pacientes variados
lote_pacientes = [
    paciente_alto_risco,
    paciente_baixo_risco,
    paciente_limitrofe,
    paciente_jovem_grave,
    {
        'age': 45, 'anaemia': 0, 'creatinine_phosphokinase': 100,
        'diabetes': 0, 'ejection_fraction': 65, 'high_blood_pressure': 0,
        'platelets': 400000, 'serum_creatinine': 0.7, 'serum_sodium': 142,
        'sex': 0, 'smoking': 0, 'time': 200
    }
]

df_lote = inferencia_em_lote(modelo, lote_pacientes)
print("\nResultado do processamento em lote:")
print(df_lote.to_string(index=False))

print("\n" + "="*80)
print("✅ MÓDULO DE INFERÊNCIA CONCLUÍDO COM SUCESSO")
print("="*80)