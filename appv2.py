# Importar as bibliotecas necessárias
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

# Importação do SentenceTransformer
from sentence_transformers import SentenceTransformer

# Configuração do logging
logging.basicConfig(level=logging.INFO)

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
    st.image('capa.png', caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
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
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin-1')

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
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].dropna().tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].dropna().tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].dropna().tolist()

        # Certificar-se de que há dados suficientes para análise
        if not sentences_dzubukua or not sentences_arcaico or not sentences_moderno:
            st.error("Dados insuficientes em uma ou mais categorias linguísticas.")
            return

        # Ajustar o tamanho das listas para serem iguais
        min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
        sentences_dzubukua = sentences_dzubukua[:min_length]
        sentences_arcaico = sentences_arcaico[:min_length]
        sentences_moderno = sentences_moderno[:min_length]

        # Tokenização das frases
        df['Tokenização'] = df['Texto Original'].astype(str).apply(lambda x: x.split())

        # Metodologias Utilizadas e Resultados Calculados
        st.subheader("Metodologias Utilizadas e Resultados Calculados")

        # Similaridade Semântica (Sentence-BERT)
        st.info("Calculando similaridade semântica...")
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Erro ao carregar o modelo SentenceTransformer: {e}")
            return

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

        # Criar DataFrame de similaridades
        similarity_df = pd.DataFrame({
            'Idioma Dzubukuá': sentences_dzubukua,
            'Idioma Português Arcaico': sentences_arcaico,
            'Idioma Português Moderno': sentences_moderno,
            'Similaridade Semântica (Arcaico-Dzubukuá)': similarity_arcaico_dzubukua_sem,
            'Similaridade Semântica (Moderno-Dzubukuá)': similarity_moderno_dzubukua_sem,
            'Similaridade Semântica (Arcaico-Moderno)': similarity_arcaico_moderno_sem,
            'Similaridade N-gramas (Arcaico-Dzubukuá)': similarity_arcaico_dzubukua_ng,
            'Similaridade N-gramas (Moderno-Dzubukuá)': similarity_moderno_dzubukua_ng,
            'Similaridade N-gramas (Arcaico-Moderno)': similarity_arcaico_moderno_ng,
            'Similaridade Word2Vec (Arcaico-Dzubukuá)': similarity_arcaico_dzubukua_w2v,
            'Similaridade Word2Vec (Moderno-Dzubukuá)': similarity_moderno_dzubukua_w2v,
            'Similaridade Word2Vec (Arcaico-Moderno)': similarity_arcaico_moderno_w2v,
            'Similaridade Fonológica (Arcaico-Dzubukuá)': similarity_arcaico_dzubukua_phon,
            'Similaridade Fonológica (Moderno-Dzubukuá)': similarity_moderno_dzubukua_phon,
            'Similaridade Fonológica (Arcaico-Moderno)': similarity_arcaico_moderno_phon,
        })

        st.subheader("Tabela de Similaridades")
        st.dataframe(similarity_df)

        # Correlação
        st.subheader("Correlação entre Medidas de Similaridade")
        correlation_pearson = similarity_df.select_dtypes(include=[np.number]).corr(method='pearson')
        st.markdown("**Correlação de Pearson**")
        st.dataframe(correlation_pearson)

        # Mapa de calor das correlações
        st.subheader("Mapa de Calor das Correlações")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_pearson, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca = PCA(n_components=2)
        features = similarity_df.select_dtypes(include=[np.number]).columns
        pca_result = pca.fit_transform(similarity_df[features])
        similarity_df['PCA1'] = pca_result[:, 0]
        similarity_df['PCA2'] = pca_result[:, 1]
        fig_pca = px.scatter(similarity_df, x='PCA1', y='PCA2', color='Idioma Dzubukuá', title='Visualização PCA')
        st.plotly_chart(fig_pca)

        # Regressão Linear e Múltipla
        st.subheader("Análise de Regressão Múltipla")
        X = similarity_df[['Similaridade Semântica (Arcaico-Dzubukuá)', 'Similaridade N-gramas (Arcaico-Dzubukuá)', 'Similaridade Word2Vec (Arcaico-Dzubukuá)']]
        y = similarity_df['Similaridade Fonológica (Arcaico-Dzubukuá)']
        model = LinearRegression()
        model.fit(X, y)
        similarity_df['Predição Regressão'] = model.predict(X)
        st.dataframe(similarity_df[['Similaridade Semântica (Arcaico-Dzubukuá)', 'Similaridade N-gramas (Arcaico-Dzubukuá)', 'Similaridade Word2Vec (Arcaico-Dzubukuá)', 'Similaridade Fonológica (Arcaico-Dzubukuá)', 'Predição Regressão']])

        # Gráfico de Regressão Linear
        st.subheader("Gráfico de Regressão Linear")
        fig_reg = px.scatter(similarity_df, x='Similaridade Semântica (Arcaico-Dzubukuá)', y='Similaridade Fonológica (Arcaico-Dzubukuá)', trendline='ols', title='Regressão Linear: Similaridade Semântica vs Similaridade Fonológica')
        st.plotly_chart(fig_reg)

        # ANOVA (Análise de Variância)
        st.subheader("ANOVA (Análise de Variância)")
        f_val, p_val = f_oneway(similarity_df['Similaridade Semântica (Arcaico-Dzubukuá)'], similarity_df['Similaridade N-gramas (Arcaico-Dzubukuá)'], similarity_df['Similaridade Word2Vec (Arcaico-Dzubukuá)'])
        st.markdown(f"F-Valor: {f_val:.4f}, P-Valor: {p_val:.4f}")

        # Clusterização (K-Means)
        st.subheader("Clusterização com K-Means")
        kmeans = KMeans(n_clusters=3, random_state=42)
        similarity_df['Cluster'] = kmeans.fit_predict(similarity_df[features])
        fig_kmeans = px.scatter(similarity_df, x='PCA1', y='PCA2', color='Cluster', title='Clusterização com K-Means')
        st.plotly_chart(fig_kmeans)

        # Dendrograma para Análise Hierárquica
        st.subheader("Dendrograma para Análise Hierárquica")
        linked = linkage(similarity_df[features], method='ward')
        fig_dend, ax = plt.subplots(figsize=(12, 8))
        dendrogram(linked, ax=ax)
        st.pyplot(fig_dend)

        # Outros Recursos e Download
        st.subheader("Baixar Dados Calculados")
        salvar_dataframe(similarity_df)

def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    embeddings = model.encode(all_sentences, batch_size=32, normalize_embeddings=True)

    len_dzubukua = len(sentences_dzubukua)
    len_arcaico = len(sentences_arcaico)
    len_moderno = len(sentences_moderno)

    embeddings_dzubukua = embeddings[:len_dzubukua]
    embeddings_arcaico = embeddings[len_dzubukua:len_dzubukua + len_arcaico]
    embeddings_moderno = embeddings[len_dzubukua + len_arcaico:]

    similarity_arcaico_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_arcaico[i]])[0][0] for i in range(len(embeddings_dzubukua))]
    similarity_moderno_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_moderno[i]])[0][0] for i in range(len(embeddings_dzubukua))]
    similarity_arcaico_moderno = [cosine_similarity([embeddings_arcaico[i]], [embeddings_moderno[i]])[0][0] for i in range(len(embeddings_dzubukua))]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=2):
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), binary=True, analyzer='char_wb')
        ngram_matrix = vectorizer.fit_transform(sentences).toarray()
        return ngram_matrix

    ngramas_dzubukua = ngramas(sentences_dzubukua, n)
    ngramas_arcaico = ngramas(sentences_arcaico, n)
    ngramas_moderno = ngramas(sentences_moderno, n)

    min_samples = min(len(ngramas_dzubukua), len(ngramas_arcaico), len(ngramas_moderno))

    ngramas_dzubukua = ngramas_dzubukua[:min_samples]
    ngramas_arcaico = ngramas_arcaico[:min_samples]
    ngramas_moderno = ngramas_moderno[:min_samples]

    def sorensen_dice(a, b):
        intersection = np.sum(np.minimum(a, b))
        total = np.sum(a) + np.sum(b)
        return (2 * intersection) / total if total > 0 else 0

    similarity_arcaico_dzubukua = [sorensen_dice(ngramas_dzubukua[i], ngramas_arcaico[i]) for i in range(min_samples)]
    similarity_moderno_dzubukua = [sorensen_dice(ngramas_dzubukua[i], ngramas_moderno[i]) for i in range(min_samples)]
    similarity_arcaico_moderno = [sorensen_dice(ngramas_arcaico[i], ngramas_moderno[i]) for i in range(min_samples)]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

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

    similarity_arcaico_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_arcaico[i]])[0][0] for i in range(len(vectors_dzubukua))]
    similarity_moderno_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_moderno[i]])[0][0] for i in range(len(vectors_dzubukua))]
    similarity_arcaico_moderno = [cosine_similarity([vectors_arcaico[i]], [vectors_moderno[i]])[0][0] for i in range(len(vectors_dzubukua))]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_fonologica(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    def average_levenshtein_similarity(s1_list, s2_list):
        similarities = []
        for s1, s2 in zip(s1_list, s2_list):
            try:
                s1_phonetic = jellyfish.soundex(s1)
                s2_phonetic = jellyfish.soundex(s2)
                dist = jellyfish.levenshtein_distance(s1_phonetic, s2_phonetic)
                max_len = max(len(s1_phonetic), len(s2_phonetic))
                similarity = 1 - (dist / max_len) if max_len > 0 else 1
                similarities.append(similarity)
            except Exception as e:
                similarities.append(0)
        return similarities

    similarity_arcaico_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_arcaico)
    similarity_moderno_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_moderno)
    similarity_arcaico_moderno_phon = average_levenshtein_similarity(sentences_arcaico, sentences_moderno)

    return similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon

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
