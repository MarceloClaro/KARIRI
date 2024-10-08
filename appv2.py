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
from scipy.cluster.hierarchy import dendrogram
import jellyfish

# Bibliotecas adicionais
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway, t  # Para ANOVA e cálculo de margens de erro
from scipy.optimize import curve_fit  # Para q-Exponencial

# Importação do SentenceTransformer
from sentence_transformers import SentenceTransformer

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Definir o caminho do ícone
icon_path = "logo.png"

# Configuração do Streamlit
if os.path.exists(icon_path):
    st.set_page_config(page_title="Geomaker +IA", page_icon=icon_path, layout="wide")
else:
    st.set_page_config(page_title="Geomaker +IA", layout="wide")

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

        # Verificar se as colunas necessárias existem
        required_columns = ['Idioma', 'Texto Original', 'Tradução para o Português Moderno']
        if not all(column in df.columns for column in required_columns):
            st.error(f"O arquivo CSV deve conter as colunas: {', '.join(required_columns)}")
            return

        # Exibir a tabela completa do dataset
        st.subheader("Tabela Completa do Dataset")
        st.dataframe(df)

        # Dados Processados
        st.subheader("Dados Processados")

        # Perguntar ao usuário se deseja prosseguir com a extração de frases
        if st.checkbox("Deseja prosseguir com a extração de frases?"):
            # Extrair frases de cada idioma
            sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].dropna().tolist()
            sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].dropna().tolist()
            sentences_moderno = df[df['Idioma'] == 'Português Moderno']['Texto Original'].dropna().tolist()

            # Certificar-se de que há dados suficientes para análise
            if not sentences_dzubukua or not sentences_arcaico or not sentences_moderno:
                st.error("Dados insuficientes em uma ou mais categorias linguísticas.")
                return

            # Ajustar o tamanho das listas para serem iguais
            min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
            sentences_dzubukua = sentences_dzubukua[:min_length]
            sentences_arcaico = sentences_arcaico[:min_length]
            sentences_moderno = sentences_moderno[:min_length]

            # Criar DataFrame para armazenar os dados
            data = pd.DataFrame({
                'Dzubukuá': sentences_dzubukua,
                'Português Arcaico': sentences_arcaico,
                'Português Moderno': sentences_moderno
            })

            # Tokenização das frases
            data['Tokenização'] = data['Dzubukuá'].astype(str).apply(lambda x: x.split())

            # Metodologias Utilizadas e Resultados Calculados
            st.subheader("Metodologias Utilizadas e Resultados Calculados")

            # Perguntar ao usuário se deseja prosseguir com o cálculo de similaridade semântica
            if st.checkbox("Deseja calcular a similaridade semântica?"):
                # Similaridade Semântica (Sentence-BERT)
                st.info("Calculando similaridade semântica...")
                try:
                    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo SentenceTransformer: {e}")
                    return

                similarity_arcaico_dzubukua_sem, similarity_moderno_dzubukua_sem, similarity_arcaico_moderno_sem = calcular_similaridade_semantica(
                    model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

                # Adicionar as similaridades ao DataFrame
                data['Similaridade Semântica (Dzubukuá-Arcaico)'] = similarity_arcaico_dzubukua_sem
                data['Similaridade Semântica (Dzubukuá-Moderno)'] = similarity_moderno_dzubukua_sem
                data['Similaridade Semântica (Arcaico-Moderno)'] = similarity_arcaico_moderno_sem

            # Perguntar ao usuário se deseja prosseguir com o cálculo de similaridade lexical (N-gramas)
            if st.checkbox("Deseja calcular a similaridade lexical (N-gramas)?"):
                # Similaridade Lexical (N-gramas)
                st.info("Calculando similaridade lexical (N-gramas)...")
                similarity_arcaico_dzubukua_ng, similarity_moderno_dzubukua_ng, similarity_arcaico_moderno_ng = calcular_similaridade_ngramas(
                    sentences_dzubukua, sentences_arcaico, sentences_moderno)

                # Adicionar as similaridades ao DataFrame
                data['Similaridade N-gramas (Dzubukuá-Arcaico)'] = similarity_arcaico_dzubukua_ng
                data['Similaridade N-gramas (Dzubukuá-Moderno)'] = similarity_moderno_dzubukua_ng
                data['Similaridade N-gramas (Arcaico-Moderno)'] = similarity_arcaico_moderno_ng

            # Perguntar ao usuário se deseja prosseguir com o cálculo de similaridade lexical (Word2Vec)
            if st.checkbox("Deseja calcular a similaridade lexical (Word2Vec)?"):
                # Similaridade Lexical (Word2Vec)
                st.info("Calculando similaridade lexical (Word2Vec)...")
                similarity_arcaico_dzubukua_w2v, similarity_moderno_dzubukua_w2v, similarity_arcaico_moderno_w2v = calcular_similaridade_word2vec(
                    sentences_dzubukua, sentences_arcaico, sentences_moderno)

                # Adicionar as similaridades ao DataFrame
                data['Similaridade Word2Vec (Dzubukuá-Arcaico)'] = similarity_arcaico_dzubukua_w2v
                data['Similaridade Word2Vec (Dzubukuá-Moderno)'] = similarity_moderno_dzubukua_w2v
                data['Similaridade Word2Vec (Arcaico-Moderno)'] = similarity_arcaico_moderno_w2v

            # Perguntar ao usuário se deseja prosseguir com o cálculo de similaridade fonológica
            if st.checkbox("Deseja calcular a similaridade fonológica?"):
                # Similaridade Fonológica
                st.info("Calculando similaridade fonológica...")
                similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon = calcular_similaridade_fonologica(
                    sentences_dzubukua, sentences_arcaico, sentences_moderno)

                # Adicionar as similaridades ao DataFrame
                data['Similaridade Fonológica (Dzubukuá-Arcaico)'] = similarity_arcaico_dzubukua_phon
                data['Similaridade Fonológica (Dzubukuá-Moderno)'] = similarity_moderno_dzubukua_phon
                data['Similaridade Fonológica (Arcaico-Moderno)'] = similarity_arcaico_moderno_phon

            # Perguntar ao usuário se deseja prosseguir com o cálculo das estatísticas descritivas
            if st.checkbox("Deseja calcular as estatísticas descritivas?"):
                # Cálculo das Estatísticas Descritivas
                st.subheader("Estatísticas Descritivas")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                stats_desc = data[numeric_cols].describe().transpose()
                st.dataframe(stats_desc)

            # Perguntar ao usuário se deseja prosseguir com o cálculo das margens de erro
            if st.checkbox("Deseja calcular as margens de erro?"):
                # Cálculo das Margens de Erro
                st.subheader("Cálculo das Margens de Erro")
                confidence_level = 0.95
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                data_margins = {}
                for col in numeric_cols:
                    std_error = data[col].std() / np.sqrt(len(data[col]))
                    margin_error = z_score * std_error
                    data_margins[col] = margin_error
                margins_df = pd.DataFrame.from_dict(data_margins, orient='index', columns=['Margem de Erro'])
                st.dataframe(margins_df)

                # Adicionar as margens de erro ao DataFrame principal
                for col in numeric_cols:
                    data[f'Margem de Erro ({col})'] = data_margins[col]

            # Exibir o DataFrame final
            st.subheader("Dados Finais Processados")
            st.dataframe(data)

            # Perguntar ao usuário se deseja prosseguir com a análise de correlação
            if st.checkbox("Deseja realizar a análise de correlação?"):
                # Análise de Correlação
                st.subheader("Análise de Correlação")
                correlation_matrix = data[numeric_cols].corr()
                st.dataframe(correlation_matrix)
                # Mapa de calor das correlações
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            # Perguntar ao usuário se deseja prosseguir com a análise de regressão
            if st.checkbox("Deseja realizar a análise de regressão?"):
                # Análise de Regressão Múltipla
                st.subheader("Análise de Regressão Múltipla")
                X = data[['Similaridade Semântica (Dzubukuá-Arcaico)', 'Similaridade N-gramas (Dzubukuá-Arcaico)', 'Similaridade Word2Vec (Dzubukuá-Arcaico)']]
                y = data['Similaridade Fonológica (Dzubukuá-Arcaico)']
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                st.write(model.summary())

            # Perguntar ao usuário se deseja prosseguir com a análise de clusterização
            if st.checkbox("Deseja realizar a análise de clusterização?"):
                # Análise de Clusterização
                st.subheader("Análise de Clusterização")
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(data[numeric_cols].dropna(axis=1))
                kmeans = KMeans(n_clusters=3, random_state=42)
                data['Cluster'] = kmeans.fit_predict(scaled_features)
                fig_cluster = px.scatter(data, x='Similaridade Semântica (Dzubukuá-Arcaico)', y='Similaridade Fonológica (Dzubukuá-Arcaico)', color='Cluster', title='Clusterização com K-Means')
                st.plotly_chart(fig_cluster)

            # Perguntar ao usuário se deseja baixar os dados processados
            if st.checkbox("Deseja baixar os dados processados?"):
                # Baixar Dados Calculados
                st.subheader("Baixar Dados Calculados")
                salvar_dataframe(data)

# Funções para calcular as similaridades
def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade semântica usando o modelo Sentence-BERT."""
    # Combinar todas as frases
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    # Gerar embeddings para todas as frases
    embeddings = model.encode(all_sentences, batch_size=32, normalize_embeddings=True)

    # Separar embeddings de cada conjunto de frases
    embeddings_dzubukua = embeddings[:len(sentences_dzubukua)]
    embeddings_arcaico = embeddings[len(sentences_dzubukua):len(sentences_dzubukua) + len(sentences_arcaico)]
    embeddings_moderno = embeddings[len(sentences_dzubukua) + len(sentences_arcaico):]

    # Calculando a similaridade de cosseno entre os embeddings correspondentes
    similarity_arcaico_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_arcaico[i]])[0][0] for i in range(len(embeddings_dzubukua))]
    similarity_moderno_dzubukua = [cosine_similarity([embeddings_dzubukua[i]], [embeddings_moderno[i]])[0][0] for i in range(len(embeddings_dzubukua))]
    similarity_arcaico_moderno = [cosine_similarity([embeddings_arcaico[i]], [embeddings_moderno[i]])[0][0] for i in range(len(embeddings_arcaico))]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=2):
    """Calcula a similaridade lexical usando N-gramas e Coeficiente de Sorensen-Dice."""
    # Função para gerar N-gramas binários
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), binary=True, analyzer='char_wb').fit(sentences)
        ngram_matrix = vectorizer.transform(sentences).toarray()
        return ngram_matrix

    # Gerar N-gramas para cada conjunto de frases
    ngramas_dzubukua = ngramas(sentences_dzubukua, n)
    ngramas_arcaico = ngramas(sentences_arcaico, n)
    ngramas_moderno = ngramas(sentences_moderno, n)

    # Garantir que o número de frases seja o mesmo entre todos os conjuntos
    num_frases = min(len(ngramas_dzubukua), len(ngramas_arcaico), len(ngramas_moderno))

    # Ajustar os vetores de N-gramas para ter o mesmo número de frases
    ngramas_dzubukua = ngramas_dzubukua[:num_frases]
    ngramas_arcaico = ngramas_arcaico[:num_frases]
    ngramas_moderno = ngramas_moderno[:num_frases]

    # Certifique-se de que os vetores de N-gramas tenham o mesmo número de colunas (dimensão)
    min_dim = min(ngramas_dzubukua.shape[1], ngramas_arcaico.shape[1], ngramas_moderno.shape[1])
    ngramas_dzubukua = ngramas_dzubukua[:, :min_dim]
    ngramas_arcaico = ngramas_arcaico[:, :min_dim]
    ngramas_moderno = ngramas_moderno[:, :min_dim]

    # Calculando o Coeficiente de Sorensen-Dice entre os N-gramas
    def sorensen_dice(a, b):
        intersection = np.sum(np.minimum(a, b))
        total = np.sum(a) + np.sum(b)
        return 2 * intersection / total if total > 0 else 0

    similarity_arcaico_dzubukua = [
        sorensen_dice(ngramas_dzubukua[i], ngramas_arcaico[i]) 
        for i in range(num_frases)
    ]
    similarity_moderno_dzubukua = [
        sorensen_dice(ngramas_dzubukua[i], ngramas_moderno[i]) 
        for i in range(num_frases)
    ]
    similarity_arcaico_moderno = [
        sorensen_dice(ngramas_arcaico[i], ngramas_moderno[i]) 
        for i in range(num_frases)
    ]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_word2vec(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade lexical usando Word2Vec."""
    # Tokenização das frases
    tokenized_sentences = [sentence.split() for sentence in (sentences_dzubukua + sentences_arcaico + sentences_moderno)]

    # Treinamento do modelo Word2Vec
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Função para obter o vetor médio de uma frase
    def sentence_vector(sentence, model):
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Obter vetores para cada frase
    vectors_dzubukua = [sentence_vector(sentence, model) for sentence in sentences_dzubukua]
    vectors_arcaico = [sentence_vector(sentence, model) for sentence in sentences_arcaico]
    vectors_moderno = [sentence_vector(sentence, model) for sentence in sentences_moderno]

    # Calculando a similaridade de cosseno entre os vetores correspondentes
    similarity_arcaico_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_arcaico[i]])[0][0] for i in range(len(vectors_dzubukua))]
    similarity_moderno_dzubukua = [cosine_similarity([vectors_dzubukua[i]], [vectors_moderno[i]])[0][0] for i in range(len(vectors_dzubukua))]
    similarity_arcaico_moderno = [cosine_similarity([vectors_arcaico[i]], [vectors_moderno[i]])[0][0] for i in range(len(vectors_arcaico))]

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

def calcular_similaridade_fonologica(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade fonológica usando codificação fonética e distância de Levenshtein."""
    # Função para calcular a similaridade fonológica média entre duas listas de frases
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
