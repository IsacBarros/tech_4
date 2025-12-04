# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# --- CONFIGURAÇÃO ---
MODEL_FILE = 'obesity_prediction_model_pipeline.pkl'
ENCODER_FILE = 'label_encoder.pkl'
# Usando o nome do arquivo mais recente fornecido
DATA_FILE = 'Obesity_normalizado_ok.csv' 

# --- 1. Carregamento de Recursos ---
@st.cache_data
def load_resources():
    """Carrega o modelo, encoder e dados brutos de forma eficiente."""
    # Garante que os arquivos do modelo existem antes de tentar carregar
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        return None, None, None

    model_pipeline = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    
    # Lógica de carregamento e limpeza idêntica ao script de treinamento para o Dashboard
    try:
        df = pd.read_csv(DATA_FILE, sep=',')
        if df.shape[1] == 1:
            df = pd.read_csv(DATA_FILE, sep=';')
            
        # Seleciona as colunas originais e a coluna alvo ('Obesity')
        original_cols = [col for col in df.columns if not col.startswith('N_') and col != 'Obesity']
        df = df[original_cols + ['Obesity']]
        
        # Trata o separador decimal e converte para numérico
        cols_to_convert = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Feature Engineering: Cálculo do IMC
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        # Mapeamento de classes para o Dashboard
        class_mapping = {
            'Insufficient_Weight': 'Baixo Peso',
            'Normal_Weight': 'Peso Normal',
            'Overweight_Level_I': 'Sobrepeso I',
            'Overweight_Level_II': 'Sobrepeso II',
            'Obesity_Type_I': 'Obesidade I',
            'Obesity_Type_II': 'Obesidade II',
            'Obesity_Type_III': 'Obesidade III'
        }
        df['Obesity_Class'] = df['Obesity'].replace(class_mapping)
        # Remove NAs para garantir que os gráficos funcionem
        df.dropna(subset=['Age', 'BMI', 'Obesity_Class', 'Gender'], inplace=True)
        
        return model_pipeline, le, df
    except Exception as e:
        st.error(f"Erro ao carregar ou limpar os dados para o dashboard. Verifique o arquivo '{DATA_FILE}'. Erro: {e}")
        return model_pipeline, le, pd.DataFrame()

model_pipeline, le, df_raw = load_resources()

# Verifica se o modelo e encoder foram carregados
if model_pipeline is None or le is None:
    st.error("Falha ao carregar recursos. O sistema preditivo não está disponível. Execute 'pipeline_treinamento.py' primeiro para gerar os arquivos .pkl.")
    st.stop()
    
# --- 2. Função de Previsão ---
def predict_obesity(input_data):
    """Realiza a previsão do nível de obesidade."""
    input_df = pd.DataFrame([input_data])
    prediction_encoded = model_pipeline.predict(input_df)
    prediction_label = le.inverse_transform(prediction_encoded)
    return prediction_label[0]

# --- 3. Configuração do Streamlit ---
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS (Foco Clínico e Clean)
st.markdown("""
<style>
.stApp {background-color: #f0f2f6;}
.header-title {font-size: 2.5em; font-weight: bold; color: #1e88e5; padding-bottom: 10px; border-bottom: 3px solid #1e88e5;}
.prediction-box {border: 2px solid #1e88e5; padding: 20px; border-radius: 10px; background-color: #e3f2fd; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="header-title">Sistema Preditivo de Obesidade Clínica</p>', unsafe_allow_html=True)
st.markdown("Ferramenta de Machine Learning para auxiliar o diagnóstico e a intervenção de risco em pacientes.")

tab1, tab2 = st.tabs(["🩺 Sistema Preditivo (Diagnóstico)", "📊 Dashboard Analítico (Insights)"])

# --- TAB 1: SISTEMA PREDITIVO (Etapa 6) ---
with tab1:
    st.header("Entrada de Dados do Paciente")
    
    # Inputs do Paciente na Sidebar
    with st.sidebar:
        st.header("Dados do Paciente")
        
        # Variáveis Físicas e Demográficas
        gender = st.selectbox('Gênero', ('Female', 'Male'))
        age = st.slider('Idade', 10, 80, 25)
        height = st.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        
        # Variáveis Comportamentais
        st.subheader("Hábitos Alimentares e Estilo de Vida")
        family_history = st.selectbox('Histórico Familiar de Excesso de Peso', ('yes', 'no'))
        favc = st.selectbox('Consumo Frequente de Alimentos Calóricos?', ('yes', 'no'))
        fcvc = st.slider('Consumo de Vegetais (1=Raramente, 3=Sempre)', 1.0, 3.0, 2.0, 0.1)
        ncp = st.slider('Refeições Principais Diárias', 1.0, 4.0, 3.0, 0.1)
        caec = st.selectbox('Come entre Refeições', ('no', 'Sometimes', 'Frequently', 'Always'))
        
        st.subheader("Outros Fatores")
        smoke = st.selectbox('Fumante', ('no', 'yes'))
        ch2o = st.slider('Ingestão Diária de Água (L)', 1.0, 3.0, 2.0, 0.1)
        scc = st.selectbox('Monitora Calorias', ('no', 'yes'))
        faf = st.slider('Pratica Atividade Física (0=Nunca, 3=Diariamente)', 0.0, 3.0, 1.0, 0.1)
        tue = st.slider('Tempo de Tela (h/dia)', 0.0, 3.0, 1.0, 0.1)
        calc = st.selectbox('Consumo de Álcool', ('no', 'Sometimes', 'Frequently', 'Always'))
        mtrans = st.selectbox('Meio de Transporte', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

    input_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight, 'family_history': family_history,
        'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o,
        'SCC': scc, 'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
    }
    
    bmi_calc = weight / (height ** 2)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="IMC (Índice de Massa Corporal) Calculado", value=f"{bmi_calc:.2f}")

    with col2:
        if st.button('Fazer Previsão de Risco', type="primary"):
            predicted_level = predict_obesity(input_data)
            
            # Mapeamento de Risco (Visão de Negócio/Clínica)
            if 'Obesity_Type_III' in predicted_level:
                color, icon, recommendation = '#d32f2f', '🚨', "Risco Extremo. Requer intervenção multidisciplinar urgente (nutrição, endocrinologia, cirurgia bariátrica)."
            elif 'Obesity_Type_II' in predicted_level:
                color, icon, recommendation = '#ff7043', '⚠️', "Risco Alto. Programa de perda de peso supervisionado e rastreamento de comorbidades."
            elif 'Obesity_Type_I' in predicted_level:
                color, icon, recommendation = '#ffb300', '🔶', "Risco Moderado. Foco em educação nutricional e aumento gradual da atividade física."
            elif 'Overweight' in predicted_level:
                color, icon, recommendation = '#43a047', '⬆️', "Risco Leve (Sobrepeso). Aconselhamento para prevenção e manutenção de hábitos saudáveis."
            else:
                color, icon, recommendation = '#00c853', '✅', "Risco Baixo. Reforçar manutenção de hábitos saudáveis e monitoramento periódico."
            
            st.markdown(f"""
            <div class="prediction-box" style="border-color: {color}; background-color: {color}1a;">
                <h3>{icon} Previsão do Nível de Obesidade:</h3>
                <h1 style="color: {color};">{predicted_level.replace('_', ' ')}</h1>
                <p><strong>RECOMENDAÇÃO CLÍNICA:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: DASHBOARD ANALÍTICO (Etapa 7) ---
with tab2:
    if df_raw.empty:
        st.warning("Não foi possível carregar o dataset para a análise. O dashboard não está disponível.")
    else:
        st.header("Painel de Insights: Fatores-Chave para a Obesidade")
        st.markdown("Análise de padrões comportamentais e físicos para direcionar programas de saúde e alocação de recursos hospitalares.")
        
        colA, colB = st.columns(2)

        # 1. Distribuição dos Níveis de Obesidade
        with colA:
            st.subheader("1. Distribuição de Casos por Nível de Obesidade")
            fig_dist = px.bar(
                df_raw['Obesity_Class'].value_counts().reset_index(name='count'), # Adicionado nome explícito para o index resetado
                x='count', y='Obesity_Class', orientation='h',
                title='Prevalência na Base de Dados',
                color='Obesity_Class',
                color_discrete_sequence=px.colors.qualitative.D3
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("**Insight para Gestão:** Identificar as classes mais prevalentes (ex: Obesidade III) direciona a alocação de recursos (nutricionistas, endocrinologistas, cirurgia) para o grupo de maior demanda e risco de comorbidades.")

        # 2. Relação IMC e Idade
        with colB:
            st.subheader("2. IMC e Idade por Gênero")
            fig_bmi_age = px.scatter(
                df_raw, x='Age', y='BMI', color='Gender',
                title='IMC em Relação à Idade',
                hover_data=['Weight', 'Height'],
                color_discrete_map={'Female': 'blue', 'Male': 'red'}
            )
            st.plotly_chart(fig_bmi_age, use_container_width=True)
            st.markdown("**Insight Clínico:** A variação de IMC é alta em todas as idades, mas a tendência de peso excessivo pode começar precocemente. Sugere a necessidade de programas de prevenção desde a adolescência/início da vida adulta.")
            
        st.markdown("---")

        colC, colD = st.columns(2)

        # 3. Consumo de Álcool (CALC) vs. Obesidade
        with colC:
            st.subheader("3. Consumo de Álcool (CALC) por Nível de Obesidade")
            # Correção na agregação para compatibilidade com Plotly
            df_calc = df_raw.groupby('Obesity_Class')['CALC'].value_counts(normalize=True).mul(100).rename('Percentual').reset_index()
            
            fig_calc = px.bar(
                df_calc, x='Obesity_Class', y='Percentual', color='CALC',
                title='Frequência de Consumo de Álcool por Nível',
                color_discrete_sequence=px.colors.qualitative.Antique
            )
            fig_calc.update_layout(barmode='stack', xaxis_tickangle=-45)
            st.plotly_chart(fig_calc, use_container_width=True)
            st.markdown("**Insight Comportamental:** Avaliar a frequência de ingestão de álcool, que adiciona calorias vazias à dieta, é vital nas consultas de acompanhamento para pacientes com alto risco, especialmente nas classes de Sobrepeso/Obesidade I.")

        # 4. Boxplots de Tempo de Tela (TUE) e Nível de Obesidade
        with colD:
            st.subheader("4. Tempo de Tela (TUE) vs. Obesidade")
            fig_tue = px.box(
                df_raw, x='Obesity_Class', y='TUE', 
                title='Distribuição do Tempo de Tela (TUE) por Nível',
                color='Obesity_Class',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_tue.update_layout(xaxis_title="Nível de Obesidade", yaxis_title="TUE (0-3)")
            st.plotly_chart(fig_tue, use_container_width=True)
            st.markdown("**Insight de Risco:** O aumento do tempo de tela (TUE) está associado à inatividade (sedentarismo), um fator de risco clássico. Pacientes com Obesidade II e III tendem a ter um tempo de tela mais elevado, exigindo foco na redução do comportamento sedentário.")