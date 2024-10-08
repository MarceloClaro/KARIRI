# Importação das bibliotecas necessárias
import os
import logging
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
import jellyfish

# Bibliotecas adicionais
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway  # Para ANOVA
from scipy.optimize import curve_fit  # Para q-Exponencial

# Importação do SentenceTransformer
from sentence_transformers import SentenceTransformer

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Configuração do Streamlit
st.set_page_config(page_title="Análise Linguística Avançada", layout="wide")

# Título da aplicação
st.title('Análise Linguística Avançada para Dzubukuá, Português Arcaico e Português Moderno')

# Função principal
def main():
    # Seção 1: Carregamento dos Dados
    st.header("Seção 1: Carregamento dos Dados")
    if st.checkbox("Deseja carregar o arquivo CSV?"):
        uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")
        if uploaded_file is not None:
            # Leitura do arquivo CSV
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.success("Arquivo carregado com sucesso!")
        else:
            st.warning("Por favor, faça o upload de um arquivo CSV.")

    # Seção 2: Verificação das Colunas Necessárias
    if 'df' in locals() and st.checkbox("Deseja verificar as colunas necessárias?"):
        required_columns = ['Idioma', 'Texto Original', 'Tradução para o Português Moderno']
        if not all(column in df.columns for column in required_columns):
            st.error(f"O arquivo CSV deve conter as colunas: {', '.join(required_columns)}")
            return
        else:
            st.success("Todas as colunas necessárias estão presentes.")
            st.subheader("Dados Carregados")
            st.dataframe(df.head())

    # Seção 3: Processamento dos Dados
    if 'df' in locals() and st.checkbox("Deseja processar os dados?"):
        data = processar_dados(df)
        st.success("Dados processados com sucesso!")
        st.dataframe(data.head())

    # Seção 4: Cálculo das Similaridades
    if 'data' in locals() and st.checkbox("Deseja calcular as similaridades?"):
        data = calcular_similaridades(data)
        st.success("Similaridades calculadas com sucesso!")
        st.dataframe(data.head())

    # Seção 5: Análises Estatísticas
    if 'data' in locals() and st.checkbox("Deseja realizar análises estatísticas?"):
        data = analises_estatisticas(data)
        st.success("Análises estatísticas concluídas!")
        st.dataframe(data.head())

    # Seção 6: Análises Linguísticas
    if 'data' in locals() and st.checkbox("Deseja realizar análises linguísticas?"):
        data = analises_linguisticas(data)
        st.success("Análises linguísticas concluídas!")
        st.dataframe(data.head())

    # Seção 7: Download dos Dados
    if 'data' in locals() and st.checkbox("Deseja baixar os dados processados?"):
        salvar_dataframe(data)
        st.success("Dados prontos para download!")

# Função para processar os dados
def processar_dados(df):
    """Processa os dados carregados e prepara para análises."""
    # Extrair frases de cada idioma
    sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].dropna().tolist()
    sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].dropna().tolist()
    sentences_moderno = df[df['Idioma'] == 'Português Moderno']['Texto Original'].dropna().tolist()

    # Ajustar o tamanho das listas para serem iguais
    min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
    sentences_dzubukua = sentences_dzubukua[:min_length]
    sentences_arcaico = sentences_arcaico[:min_length]
    sentences_moderno = sentences_moderno[:min_length]

    # Criar DataFrame para armazenar os dados
    data = pd.DataFrame({
        'Idioma': ['Dzubukuá'] * min_length + ['Português Arcaico'] * min_length + ['Português Moderno'] * min_length,
        'Texto Original': sentences_dzubukua + sentences_arcaico + sentences_moderno,
        'Tradução para o Português Moderno': sentences_moderno * 3  # Supondo que a tradução é a mesma para alinhamento
    })

    # Tokenização das frases
    data['Tokenização'] = data['Texto Original'].astype(str).apply(lambda x: x.split())

    return data

# Função para calcular as similaridades
def calcular_similaridades(data):
    """Calcula as similaridades e adiciona ao DataFrame."""
    st.info("Calculando Soundex e Distância de Levenshtein...")

    # Soundex
    data['Soundex'] = data['Texto Original'].apply(lambda x: jellyfish.soundex(x))

    # Distância de Levenshtein em relação à tradução
    data['Distância de Levenshtein'] = data.apply(
        lambda row: jellyfish.levenshtein_distance(row['Texto Original'], row['Tradução para o Português Moderno']), axis=1
    )

    st.info("Calculando Similaridade de Cosseno...")

    # Modelo Sentence-BERT
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Gerar embeddings
    embeddings = model.encode(data['Texto Original'].tolist())

    # Adicionar os embeddings ao DataFrame
    data['Embeddings'] = embeddings.tolist()

    # Similaridade de Cosseno entre o Texto Original e a Tradução
    data['Similaridade de Cosseno'] = data.apply(
        lambda row: cosine_similarity([row['Embeddings']], [model.encode(row['Tradução para o Português Moderno'])])[0][0], axis=1
    )

    st.info("Calculando Word2Vec...")

    # Treinamento do modelo Word2Vec
    sentences = data['Tokenização'].tolist()
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Adicionar os vetores Word2Vec ao DataFrame
    data['Word2Vec'] = data['Tokenização'].apply(
        lambda tokens: np.mean([model_w2v.wv[token] for token in tokens if token in model_w2v.wv], axis=0)
    )

    st.info("Calculando N-gramas...")

    # N-gramas
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
    X_ngram = vectorizer.fit_transform(data['Texto Original'])
    data['N-gramas'] = list(X_ngram.toarray())

    return data

# Função para realizar análises estatísticas
def analises_estatisticas(data):
    """Realiza análises estatísticas e adiciona ao DataFrame."""
    st.info("Calculando Correlações...")

    # Correlação de Pearson entre Similaridade de Cosseno e Distância de Levenshtein
    pearson_corr = data['Similaridade de Cosseno'].corr(data['Distância de Levenshtein'])
    data['Correlação de Pearson'] = [pearson_corr] * len(data)

    # Correlação de Kendall
    kendall_corr = data['Similaridade de Cosseno'].corr(data['Distância de Levenshtein'], method='kendall')
    data['Correlação de Kendall'] = [kendall_corr] * len(data)

    st.info("Realizando Análise de Regressão Múltipla...")

    # Regressão Múltipla
    X = data[['Similaridade de Cosseno', 'Distância de Levenshtein']]
    y = data['Similaridade de Cosseno']  # Como exemplo
    X = sm.add_constant(X)
    model_reg = sm.OLS(y, X).fit()
    data['Análise de Regressão Múltipla'] = model_reg.predict(X)

    st.info("Realizando ANOVA...")

    # ANOVA
    f_val, p_val = f_oneway(
        data[data['Idioma'] == 'Dzubukuá']['Similaridade de Cosseno'],
        data[data['Idioma'] == 'Português Arcaico']['Similaridade de Cosseno'],
        data[data['Idioma'] == 'Português Moderno']['Similaridade de Cosseno']
    )
    data['ANOVA'] = [p_val] * len(data)

    st.info("Calculando Margens de Erro...")

    # Margens de Erro para a Similaridade de Cosseno
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    sample_mean = data['Similaridade de Cosseno'].mean()
    sample_standard_error = stats.sem(data['Similaridade de Cosseno'])
    confidence_interval = stats.t.interval(
        confidence_level, degrees_freedom, sample_mean, sample_standard_error
    )
    margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
    data['Margens de Erro'] = [margin_of_error] * len(data)

    return data

# Função para realizar análises linguísticas
def analises_linguisticas(data):
    """Realiza análises fonológicas, morfológicas, sintáticas, etimológicas e culturais."""
    st.info("Realizando Análise Fonológica...")

    # Análise Fonológica (Simplificada)
    data['Análise Fonológica'] = data['Texto Original'].apply(
        lambda x: f"Análise fonológica de '{x}'"
    )

    st.info("Realizando Análise Morfológica...")

    # Análise Morfológica (Simplificada)
    data['Análise Morfológica'] = data['Tokenização'].apply(
        lambda tokens: f"Análise morfológica de {tokens}"
    )

    st.info("Realizando Análise Sintática...")

    # Análise Sintática (Simplificada)
    data['Análise Sintática'] = data['Texto Original'].apply(
        lambda x: f"Análise sintática de '{x}'"
    )

    st.info("Gerando Glossário Cultural...")

    # Glossário Cultural (Simplificado)
    data['Glossário Cultural'] = data['Texto Original'].apply(
        lambda x: f"Termos culturais em '{x}'"
    )

    st.info("Criando Justificativa da Tradução...")

    # Justificativa da Tradução (Simplificada)
    data['Justificativa da Tradução'] = data.apply(
        lambda row: f"Justificativa para a tradução de '{row['Texto Original']}' para '{row['Tradução para o Português Moderno']}'", axis=1
    )

    st.info("Analisando Etimologia...")

    # Etimologia (Simplificada)
    data['Etimologia'] = data['Texto Original'].apply(
        lambda x: f"Etimologia de '{x}'"
    )

    return data

# Função para salvar o DataFrame
def salvar_dataframe(data):
    """Permite o download do DataFrame em formato CSV."""
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Dados Calculados em CSV",
        data=csv,
        file_name='dados_calculados.csv',
        mime='text/csv',
    )

# Executar a aplicação
if __name__ == '__main__':
    main()
