# Importar as bibliotecas necessárias
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
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
import jellyfish

# Bibliotecas adicionais para as novas implementações
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway  # Para ANOVA
from scipy.optimize import curve_fit  # Para q-Exponencial

import os
import logging
import base64

# Definir o caminho do ícone
icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

# Verificar se o arquivo de ícone existe antes de configurá-lo
if os.path.exists(icon_path):
    st.set_page_config(page_title="Geomaker +IA", page_icon=icon_path, layout="wide")
    logging.info(f"Ícone {icon_path} carregado com sucesso.")
else:
    # Se o ícone não for encontrado, carrega sem favicon
    st.set_page_config(page_title="Geomaker +IA", layout="wide")
    logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")

# Layout da página
if os.path.exists('capa.png'):
    st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
else:
    st.warning("Imagem 'capa.png' não encontrada.")

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=200)
else:
    st.sidebar.text("Imagem do logotipo não encontrada.")

st.sidebar.title("Geomaker +IA")

# Expander de Insights
with st.sidebar.expander("Pesquisa compreenda:"):
    st.markdown("""
    # **Análise Comparativa de Idiomas: Dzubukuá, Português Arcaico e Português Moderno**

    Este estudo apresenta uma análise comparativa entre três idiomas: **Dzubukuá** (uma língua extinta), **Português Arcaico** e **Português Moderno**. O objetivo principal é investigar as similaridades e diferenças entre esses idiomas em termos de semântica, léxico e fonologia, utilizando técnicas avançadas de Processamento de Linguagem Natural (PLN) e métodos estatísticos.
    """)

# Função principal para rodar a aplicação no Streamlit
def main():
    st.title('Análises Avançadas de Similaridade Linguística para Línguas Kariri-Dzubukuá, Português Arcaico e Português Moderno')

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Exibir a tabela completa do dataset
        st.subheader("Tabela Completa do Dataset")
        st.dataframe(df)

        # Verificar se as colunas necessárias existem
        required_columns = ['Idioma', 'Texto Original', 'Tradução para o Português Moderno']
        if not all(column in df.columns for column in required_columns):
            st.error(f"O arquivo CSV deve conter as colunas: {', '.join(required_columns)}")
            return

        # Dados Processados
        st.subheader("Dados Processados")
        st.markdown("""
        **Entrada:**
        - Frases em Dzubukuá, Português Arcaico e Português Moderno, extraídas de um arquivo CSV.
        - Texto original e traduções organizados em colunas.

        **Pré-Processamento:**
        - Tokenização de frases.
        - Geração de vetores semânticos e lexicais.
        """)

        # Extrair frases de cada idioma
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

        # Certificar-se de que há dados suficientes para análise
        if not sentences_dzubukua or not sentences_arcaico or not sentences_moderno:
            st.error("Dados insuficientes em uma ou mais categorias linguísticas.")
            return

        # Tokenização das frases
        df['Tokenização'] = df['Texto Original'].apply(lambda x: x.split())

        # Metodologias Utilizadas e Resultados Calculados
        st.subheader("Metodologias Utilizadas e Resultados Calculados")

        # Similaridade Semântica (Sentence-BERT)
        st.info("Calculando similaridade semântica...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_arcaico_dzubukua_sem, similarity_moderno_dzubukua_sem, similarity_arcaico_moderno_sem = calcular_similaridade_semantica(
            model, sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.markdown("""
        **Similaridade Semântica:**
        - Usando o Sentence-BERT para gerar embeddings de frases e calcular a similaridade de cosseno.
        - Resultados fornecem uma medida da semelhança semântica entre frases em diferentes idiomas.
        """)

        # Similaridade Lexical (N-gramas)
        st.info("Calculando similaridade lexical (N-gramas)...")
        similarity_arcaico_dzubukua_ng, similarity_moderno_dzubukua_ng, similarity_arcaico_moderno_ng = calcular_similaridade_ngramas(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.markdown("""
        **Similaridade Lexical:**
        - N-gramas e Coeficiente de Sorensen-Dice para analisar a semelhança estrutural das palavras.
        """)

        # Similaridade Lexical (Word2Vec)
        st.info("Calculando similaridade lexical (Word2Vec)...")
        similarity_arcaico_dzubukua_w2v, similarity_moderno_dzubukua_w2v, similarity_arcaico_moderno_w2v = calcular_similaridade_word2vec(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.markdown("""
        - Word2Vec para calcular a similaridade lexical baseada no contexto.
        """)

        # Similaridade Fonológica
        st.info("Calculando similaridade fonológica...")
        similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon = calcular_similaridade_fonologica(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.markdown("""
        **Similaridade Fonológica:**
        - Utilizou-se Soundex e Distância de Levenshtein para medir a semelhança dos sons das palavras.
        """)

        # Adicionando resultados ao DataFrame
        df['Similaridade de Cosseno'] = similarity_arcaico_dzubukua_sem + list(similarity_moderno_dzubukua_sem) + list(similarity_arcaico_moderno_sem)
        df['Word2Vec'] = list(similarity_arcaico_dzubukua_w2v) + list(similarity_moderno_dzubukua_w2v) + list(similarity_arcaico_moderno_w2v)
        df['N-gramas'] = list(similarity_arcaico_dzubukua_ng) + list(similarity_moderno_dzubukua_ng) + list(similarity_arcaico_moderno_ng)
        df['Soundex'] = list(similarity_arcaico_dzubukua_phon)
        df['Distância de Levenshtein'] = list(similarity_moderno_dzubukua_phon)

        # Correlação
        st.subheader("Correlação entre Medidas de Similaridade")
        correlation_pearson = df.corr(method='pearson')
        st.markdown("**Correlação de Pearson**")
        st.dataframe(correlation_pearson)

        # Mapa de calor das correlações
        st.subheader("Mapa de Calor das Correlações")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_pearson, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[['Similaridade de Cosseno', 'Word2Vec', 'N-gramas']])
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Idioma', title='Visualização PCA')
        st.plotly_chart(fig_pca)

        # Regressão Linear e Múltipla
        st.subheader("Análise de Regressão Múltipla")
        X = df[['Similaridade de Cosseno', 'Word2Vec', 'N-gramas']]
        y = df['Distância de Levenshtein']
        model = LinearRegression()
        model.fit(X, y)
        df['Análise de Regressão Múltipla'] = model.predict(X)
        st.dataframe(df[['Similaridade de Cosseno', 'Word2Vec', 'N-gramas', 'Distância de Levenshtein', 'Análise de Regressão Múltipla']])

        # Gráfico de Regressão Linear
        st.subheader("Gráfico de Regressão Linear")
        fig_reg = px.scatter(df, x='Similaridade de Cosseno', y='Distância de Levenshtein', trendline='ols', title='Regressão Linear: Similaridade de Cosseno vs Distância de Levenshtein')
        st.plotly_chart(fig_reg)

        # ANOVA (Análise de Variância)
        st.subheader("ANOVA (Análise de Variância)")
        f_val, p_val = f_oneway(df['Similaridade de Cosseno'], df['Word2Vec'], df['N-gramas'])
        st.markdown(f"F-Valor: {f_val}, P-Valor: {p_val}")

        # Clusterização (K-Means)
        st.subheader("Clusterização com K-Means")
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(df[['Similaridade de Cosseno', 'Word2Vec', 'N-gramas']])
        fig_kmeans = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', title='Clusterização com K-Means')
        st.plotly_chart(fig_kmeans)

        # Dendrograma para Análise Hierárquica
        st.subheader("Dendrograma para Análise Hierárquica")
        linked = linkage(df[['Similaridade de Cosseno', 'Word2Vec', 'N-gramas']], method='ward')
        fig_dend, ax = plt.subplots(figsize=(10, 8))
        dendrogram(linked, ax=ax)
        st.pyplot(fig_dend)

        # Outros Recursos e Download
        st.subheader("Baixar Dados Calculados")
        salvar_dataframe(df)

# Função para calcular similaridade semântica usando Sentence-BERT
def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    embeddings = model.encode(all_sentences, batch_size=32, normalize_embeddings=True)

    embeddings_dzubukua = embeddings[:len(sentences_dzubukua)]
    embeddings_arcaico = embeddings[len(sentences_dzubukua):len(sentences_dzubukua) + len(sentences_arcaico)]
    embeddings_moderno = embeddings[len(sentences_dzubukua) + len(sentences_arcaico):]

    similarity_arcaico_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_arcaico).diagonal()
    similarity_moderno_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_moderno).diagonal()
    similarity_arcaico_moderno = cosine_similarity(embeddings_arcaico, embeddings_moderno).diagonal()

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade de N-gramas
def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=2):
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), binary=True, analyzer='char_wb').fit(sentences)
        ngram_matrix = vectorizer.transform(sentences).toarray()
        return ngram_matrix

    ngramas_dzubukua = ngramas(sentences_dzubukua, n)
    ngramas_arcaico = ngramas(sentences_arcaico, n)
    ngramas_moderno = ngramas(sentences_moderno, n)

    num_frases = min(len(ngramas_dzubukua), len(ngramas_arcaico), len(ngramas_moderno))

    ngramas_dzubukua = ngramas_dzubukua[:num_frases]
    ngramas_arcaico = ngramas_arcaico[:num_frases]
    ngramas_moderno = ngramas_moderno[:num_frases]

    min_dim = min(ngramas_dzubukua.shape[1], ngramas_arcaico.shape[1], ngramas_moderno.shape[1])
    ngramas_dzubukua = ngramas_dzubukua[:, :min_dim]
    ngramas_arcaico = ngramas_arcaico[:, :min_dim]
    ngramas_moderno = ngramas_moderno[:, :min_dim]

    def sorensen_dice(a, b):
        intersection = np.sum(np.minimum(a, b))
        total = np.sum(a) + np.sum(b)
        return 2 * intersection / total if total > 0 else 0

    similarity_arcaico_dzubukua = [sorensen_dice(ngramas_dzubukua[i], ngramas_arcaico[i]) for i in range(num_frases)]
    similarity_moderno_dzubukua = [sorensen_dice(ngramas_dzubukua[i], ngramas_moderno[i]) for i in range(num_frases)]
    similarity_arcaico_moderno = [sorensen_dice(ngramas_arcaico[i], ngramas_moderno[i]) for i in range(num_frases)]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade usando Word2Vec
def calcular_similaridade_word2vec(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    tokenized_sentences = [sentence.split() for sentence in (sentences_dzubukua + sentences_arcaico + sentences_moderno)]
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

    def sentence_vector(sentence, model):
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    vectors_dzubukua = [sentence_vector(sentence, model) for sentence in sentences_dzubukua]
    vectors_arcaico = [sentence_vector(sentence, model) for sentence in sentences_arcaico]
    vectors_moderno = [sentence_vector(sentence, model) for sentence in sentences_moderno]

    similarity_arcaico_dzubukua = cosine_similarity(vectors_dzubukua, vectors_arcaico).diagonal()
    similarity_moderno_dzubukua = cosine_similarity(vectors_dzubukua, vectors_moderno).diagonal()
    similarity_arcaico_moderno = cosine_similarity(vectors_arcaico, vectors_moderno).diagonal()

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade fonológica
def calcular_similaridade_fonologica(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    def average_levenshtein_similarity(s1_list, s2_list):
        similarities = []
        for s1, s2 in zip(s1_list, s2_list):
            # Codificação fonética usando Soundex
            s1_phonetic = jellyfish.soundex(s1)
            s2_phonetic = jellyfish.soundex(s2)
            # Distância de Levenshtein
            dist = jellyfish.levenshtein_distance(s1_phonetic, s2_phonetic)
            # Normalizar a distância para obter a similaridade
            max_len = max(len(s1_phonetic), len(s2_phonetic))
            similarity = 1 - (dist / max_len) if max_len > 0 else 1
            similarities.append(similarity)
        return similarities

    # Garantir que as listas tenham o mesmo comprimento
    min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
    sentences_dzubukua = sentences_dzubukua[:min_length]
    sentences_arcaico = sentences_arcaico[:min_length]
    sentences_moderno = sentences_moderno[:min_length]

    similarity_arcaico_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_arcaico)
    similarity_moderno_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_moderno)
    similarity_arcaico_moderno_phon = average_levenshtein_similarity(sentences_arcaico, sentences_moderno)

    return similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon

# Função para salvar o DataFrame
def salvar_dataframe(similarity_df):
    csv = similarity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Similaridades em CSV",
        data=csv,
        file_name='similaridades_linguisticas.csv',
        mime='text/csv',
    )

if __name__ == '__main__':
    main()
