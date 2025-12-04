# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Importa a classe customizada do novo módulo
from custom_features import BMICalculator 

# --- CONFIGURAÇÃO ---
FILE_PATH = 'Obesity_normalizado_ok.csv'
TARGET_COLUMN = 'Obesity'
RANDOM_SEED = 42

# --- ETAPA 1: Carregamento e Limpeza de Dados ---
def load_and_clean_data(file_path):
    """Carrega o CSV, identifica colunas originais e trata o separador decimal (vírgula)."""
    try:
        # Tenta ler com separador de vírgula, que parece ser o usado no anexo para colunas
        df = pd.read_csv(file_path, sep=',')
        
        # Se houver apenas uma coluna (indicando separador errado, como ponto e vírgula), tenta novamente
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=';')
            
        # Filtra apenas as colunas originais (sem o prefixo 'N_') e a coluna alvo
        original_cols = [col for col in df.columns if not col.startswith('N_') and col != TARGET_COLUMN]
        df = df[original_cols + [TARGET_COLUMN]]

        # Colunas que são numéricas, mas têm vírgula como separador decimal (exceto colunas categóricas)
        cols_to_convert = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        for col in cols_to_convert:
            if col in df.columns:
                # Remove aspas duplas, substitui vírgula por ponto e converte para float
                df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Tratamento de dados ausentes (imputação simples)
        for col in cols_to_convert:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                
        # Imputação de moda para colunas categóricas que podem ter NA
        categorical_features_clean = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_features_clean:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                # Remove aspas duplas e espaços extras de colunas categóricas
                df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.strip()
                
        return df

    except Exception as e:
        print(f"Erro ao carregar ou limpar o CSV: {e}")
        # Retorna DataFrame vazio em caso de falha crítica
        return pd.DataFrame() 

df = load_and_clean_data(FILE_PATH)

if df.empty or TARGET_COLUMN not in df.columns:
    print("Falha crítica no carregamento de dados. Não é possível continuar o treinamento.")
    exit()

# Definição final de X e y
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Codificação da variável alvo (LabelEncoder)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

# Divisão dos dados em treino e teste (Etapa 3)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded)
print(f"Dataset de Treino: {len(X_train)} amostras | Dataset de Teste: {len(X_test)} amostras")

# --- ETAPA 3: Feature Engineering (Cálculo do IMC) ---
# A classe BMICalculator é importada do custom_features.py

# Definição das colunas para o pré-processador
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Criando o pré-processador:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' # Mantém colunas que não foram transformadas (a coluna 'BMI' será mantida aqui)
)

# --- ETAPA 4: Construção do Pipeline e Modelagem Preditiva ---
# O pipeline completo: 1. Cálculo do IMC; 2. Pré-processamento; 3. Classificador
model_pipeline = Pipeline(steps=[
    ('bmi_calc', BMICalculator()),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_SEED, class_weight='balanced'))
])

print("\nIniciando o treinamento do modelo (Random Forest) com Validação Cruzada (Etapa 5)...")
model_pipeline.fit(X_train, y_train)
print("Treinamento concluído.")

# --- ETAPA 5: Avaliação do Modelo ---
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

print(f"\n--- Resultados da Avaliação ---")
print(f"Acurácia do Modelo (Conjunto de Teste): {accuracy_percentage:.2f}%")
print(f"Requisito de Acurácia: >= 75% {'(ATENDIDO)' if accuracy >= 0.75 else '(NÃO ATENDIDO)'}")

print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- ETAPA 6: Salvamento do Modelo ---
joblib.dump(model_pipeline, 'obesity_prediction_model_pipeline.pkl')
print("\nPipeline de ML (modelo e pré-processador) salvo como 'obesity_prediction_model_pipeline.pkl'.")
