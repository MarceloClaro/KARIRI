# Importar as bibliotecas necessárias

# Bibliotecas padrão
import os
import logging
import base64

# Bibliotecas de terceiros
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import jellyfish
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression  # Importação adicional

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Configuração do Streamlit
icon_path = "logo.png"
if os.path.exists(icon_path):
    st.set_page_config(page_title="Geomaker +IA", page_icon=icon_path, layout="wide")
    logging.info(f"Ícone {icon_path} carregado com sucesso.")
else:
    st.set_page_config(page_title="Geomaker +IA", layout="wide")
    logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")

# Layout da página
if os.path.exists('capa.png'):
    st.image('capa.png', caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
else:
    st.warning("Imagem 'capa.png' não encontrada.")

# Barra lateral
st.sidebar.title("Geomaker +IA")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=200)
else:
    st.sidebar.text("Imagem do logotipo não encontrada.")

# Expanders com informações adicionais
with st.sidebar.expander("Pesquisa compreenda:"):
    st.markdown("""
    # **Análise Comparativa de Idiomas: Dzubukuá, Português Arcaico e Português Moderno**
    [Conteúdo adicional...]
    """)

with st.sidebar.expander("Insights metodológicos"):
    st.markdown("""
    ### **Introdução**
    [Conteúdo adicional...]
    """)

# Contato
if os.path.exists("eu.ico"):
    st.sidebar.image("eu.ico", width=80)
else:
    st.sidebar.text("Imagem do contato não encontrada.")

st.sidebar.write("""
Projeto Geomaker + IA

https://doi.org/10.5281/zenodo.13856575
- Professor: Marcelo Claro.
Contatos: marceloclaro@gmail.com
Whatsapp: (88)981587145
Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
""")

# Controle de Áudio
st.sidebar.title("Controle de Áudio")
mp3_files = {
    "Áudio explicativo 1": "kariri.mp3",
}

selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))
loop = st.sidebar.checkbox("Repetir áudio")
play_button = st.sidebar.button("Play")
audio_placeholder = st.sidebar.empty()

def check_file_exists(mp3_path):
    if not os.path.exists(mp3_path):
        st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
        return False
    return True

if play_button and selected_mp3:
    mp3_path = mp3_files[selected_mp3]
    if check_file_exists(mp3_path):
        try:
            with open(mp3_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                loop_attr = "loop" if loop else ""
                audio_html = f"""
                <audio id="audio-player" controls autoplay {loop_attr}>
                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                  Seu navegador não suporta o elemento de áudio.
                </audio>
                """
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")

# Funções auxiliares

def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade semântica usando o modelo Sentence-BERT."""
    min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
    sentences_dzubukua = sentences_dzubukua[:min_length]
    sentences_arcaico = sentences_arcaico[:min_length]
    sentences_moderno = sentences_moderno[:min_length]

    embeddings_dzubukua = model.encode(sentences_dzubukua, batch_size=32, normalize_embeddings=True)
    embeddings_arcaico = model.encode(sentences_arcaico, batch_size=32, normalize_embeddings=True)
    embeddings_moderno = model.encode(sentences_moderno, batch_size=32, normalize_embeddings=True)

    similarity_arcaico_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_arcaico[i]])[0][0] for i in range(min_length)]
    similarity_moderno_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_moderno[i]])[0][0] for i in range(min_length)]
    similarity_arcaico_moderno = [cosine_similarity([embeddings_arcaico[i]], [embeddings_moderno[i]])[0][0] for i in range(min_length)]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade de N-gramas
def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=3):
    """Calcula a similaridade lexical usando N-gramas e Coeficiente de Sorensen-Dice."""
    from sklearn.feature_extraction.text import CountVectorizer

    # Combinar todas as frases para ajustar o vectorizer
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='char_wb', binary=True)
    vectorizer.fit(all_sentences)

    # Gerar N-gramas para cada conjunto de frases usando o mesmo vectorizer
    ngramas_dzubukua = vectorizer.transform(sentences_dzubukua).toarray()
    ngramas_arcaico = vectorizer.transform(sentences_arcaico).toarray()
    ngramas_moderno = vectorizer.transform(sentences_moderno).toarray()

    # Garantir que o número de frases seja o mesmo entre todos os conjuntos
    min_length = min(len(ngramas_dzubukua), len(ngramas_arcaico), len(ngramas_moderno))
    ngramas_dzubukua = ngramas_dzubukua[:min_length]
    ngramas_arcaico = ngramas_arcaico[:min_length]
    ngramas_moderno = ngramas_moderno[:min_length]

    # Agora, os vetores de N-gramas têm a mesma dimensão e podem ser comparados
    def sorensen_dice(a, b):
        intersection = np.sum(np.minimum(a, b))
        total = np.sum(a) + np.sum(b)
        return 2 * intersection / total if total > 0 else 0

    similarity_arcaico_dzubukua = [
        sorensen_dice(ngramas_dzubukua[i], ngramas_arcaico[i]) 
        for i in range(min_length)
    ]
    similarity_moderno_dzubukua = [
        sorensen_dice(ngramas_dzubukua[i], ngramas_moderno[i]) 
        for i in range(min_length)
    ]
    similarity_arcaico_moderno = [
        sorensen_dice(ngramas_arcaico[i], ngramas_moderno[i]) 
        for i in range(min_length)
    ]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno


def calcular_similaridade_word2vec(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade lexical usando Word2Vec."""
    tokenized_sentences = [sentence.split() for sentence in sentences_dzubukua + sentences_arcaico + sentences_moderno]

    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

    def sentence_vector(sentence, model):
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
    sentences_dzubukua = sentences_dzubukua[:min_length]
    sentences_arcaico = sentences_arcaico[:min_length]
    sentences_moderno = sentences_moderno[:min_length]

    vectors_dzubukua = [sentence_vector(sentence, model) for sentence in sentences_dzubukua]
    vectors_arcaico = [sentence_vector(sentence, model) for sentence in sentences_arcaico]
    vectors_moderno = [sentence_vector(sentence, model) for sentence in sentences_moderno]

    similarity_arcaico_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_arcaico[i]])[0][0] for i in range(min_length)]
    similarity_moderno_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_moderno[i]])[0][0] for i in range(min_length)]
    similarity_arcaico_moderno = [cosine_similarity([vectors_arcaico[i]], [vectors_moderno[i]])[0][0] for i in range(min_length)]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_fonologica(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade fonológica usando codificação fonética e distância de Levenshtein."""
    def average_levenshtein_similarity(s1_list, s2_list):
        similarities = []
        for s1, s2 in zip(s1_list, s2_list):
            s1_phonetic = jellyfish.soundex(s1)
            s2_phonetic = jellyfish.soundex(s2)
            dist = jellyfish.levenshtein_distance(s1_phonetic, s2_phonetic)
            max_len = max(len(s1_phonetic), len(s2_phonetic))
            similarity = 1 - (dist / max_len) if max_len > 0 else 1
            similarities.append(similarity)
        return similarities

    min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
    sentences_dzubukua = sentences_dzubukua[:min_length]
    sentences_arcaico = sentences_arcaico[:min_length]
    sentences_moderno = sentences_moderno[:min_length]

    similarity_arcaico_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_arcaico)
    similarity_moderno_dzubukua_phon = average_levenshtein_similarity(sentences_dzubukua, sentences_moderno)
    similarity_arcaico_moderno_phon = average_levenshtein_similarity(sentences_arcaico, sentences_moderno)

    return similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon

# Outras funções auxiliares (ex: regressão, PCA, clustering, etc.) devem ser definidas aqui

# Função principal
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

        # Extrair frases de cada idioma
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].dropna().tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].dropna().tolist()
        sentences_moderno = df[df['Idioma'] == 'Português Moderno']['Texto Original'].dropna().tolist()

        # Caso não haja frases em Português Moderno na coluna 'Texto Original', usar a coluna de 'Tradução'
        if not sentences_moderno:
            sentences_moderno = df['Tradução para o Português Moderno'].dropna().tolist()

        # Certificar-se de que há dados suficientes para análise
        if not sentences_dzubukua or not sentences_arcaico or not sentences_moderno:
            st.error("Dados insuficientes em uma ou mais categorias linguísticas.")
            return

        # Similaridade Semântica (Sentence-BERT)
        st.info("Calculando similaridade semântica...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_arcaico_dzubukua_sem, similarity_moderno_dzubukua_sem, similarity_arcaico_moderno_sem = calcular_similaridade_semantica(
            model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Lexical (N-gramas)
        st.info("Calculando similaridade lexical (N-gramas)...")
        similarity_arcaico_dzubukua_ng, similarity_moderno_dzubukua_ng, similarity_arcaico_moderno_ng = calcular_similaridade_ngramas(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Lexical (Word2Vec)
        st.info("Calculando similaridade lexical (Word2Vec)...")
        similarity_arcaico_dzubukua_w2v, similarity_moderno_dzubukua_w2v, similarity_arcaico_moderno_w2v = calcular_similaridade_word2vec(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Fonológica
        st.info("Calculando similaridade fonológica...")
        similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon = calcular_similaridade_fonologica(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Criando DataFrame com as similaridades calculadas
        similarity_df = pd.DataFrame({
            'Dzubukuá - Arcaico (Semântica)': similarity_arcaico_dzubukua_sem,
            'Dzubukuá - Moderno (Semântica)': similarity_moderno_dzubukua_sem,
            'Arcaico - Moderno (Semântica)': similarity_arcaico_moderno_sem,
            'Dzubukuá - Arcaico (N-gramas)': similarity_arcaico_dzubukua_ng,
            'Dzubukuá - Moderno (N-gramas)': similarity_moderno_dzubukua_ng,
            'Arcaico - Moderno (N-gramas)': similarity_arcaico_moderno_ng,
            'Dzubukuá - Arcaico (Word2Vec)': similarity_arcaico_dzubukua_w2v,
            'Dzubukuá - Moderno (Word2Vec)': similarity_moderno_dzubukua_w2v,
            'Arcaico - Moderno (Word2Vec)': similarity_arcaico_moderno_w2v,
            'Dzubukuá - Arcaico (Fonológica)': similarity_arcaico_dzubukua_phon,
            'Dzubukuá - Moderno (Fonológica)': similarity_moderno_dzubukua_phon,
            'Arcaico - Moderno (Fonológica)': similarity_arcaico_moderno_phon
        })

        # Exibir o DataFrame das similaridades
        st.subheader("Similaridade Calculada entre as Três Línguas")
        st.dataframe(similarity_df)

        # [Chamar as demais funções de análise e visualização aqui...]

        # Perguntar se o usuário deseja baixar os resultados como CSV
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            csv = similarity_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Similaridades em CSV",
                data=csv,
                file_name='similaridades_linguisticas.csv',
                mime='text/csv',
            )

if __name__ == '__main__':
    main()
