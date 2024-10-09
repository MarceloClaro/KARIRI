# Importar as bibliotecas necessárias
import os
import logging
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import jellyfish
import scipy.stats as stats

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import f_oneway  # Para ANOVA
from scipy.optimize import curve_fit  # Para q-Exponencial
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

import streamlit as st

# Configuração da página do Streamlit
icon_path = "logo.png"

if os.path.exists(icon_path):
    st.set_page_config(page_title="Geomaker +IA", page_icon=icon_path, layout="wide")
    logging.info(f"Ícone {icon_path} carregado com sucesso.")
else:
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

# Você pode adicionar aqui os expanders e o controle de áudio, conforme necessário.

# Funções de cálculo de similaridades e análises estatísticas

# Função para calcular similaridade semântica usando Sentence-BERT
def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade semântica usando o modelo Sentence-BERT."""
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    embeddings = model.encode(all_sentences, batch_size=32, normalize_embeddings=True)

    # Separar embeddings de cada conjunto de frases
    embeddings_dzubukua = embeddings[:len(sentences_dzubukua)]
    embeddings_arcaico = embeddings[len(sentences_dzubukua):len(sentences_dzubukua) + len(sentences_arcaico)]
    embeddings_moderno = embeddings[len(sentences_dzubukua) + len(sentences_arcaico):]

    # Calculando a similaridade de cosseno entre os embeddings
    similarity_arcaico_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_arcaico).diagonal()
    similarity_moderno_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_moderno).diagonal()
    similarity_arcaico_moderno = cosine_similarity(embeddings_arcaico, embeddings_moderno).diagonal()

    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade de N-gramas
def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=2):
    """Calcula a similaridade lexical usando N-gramas e Coeficiente de Sorensen-Dice."""
    from sklearn.feature_extraction.text import CountVectorizer

    # Função para gerar N-gramas binários
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='char_wb', binary=True).fit(sentences)
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

# Função para calcular a similaridade usando Word2Vec
def calcular_similaridade_word2vec(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade lexical usando Word2Vec."""
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
    """Calcula a similaridade fonológica usando codificação fonética e distância de Levenshtein."""
    import jellyfish

    def average_levenshtein_similarity(s1_list, s2_list):
        similarities = []
        for s1, s2 in zip(s1_list, s2_list):
            # Codificação fonética usando Soundex
            s1_phonetic = ''.join([jellyfish.soundex(word) for word in s1.split()])
            s2_phonetic = ''.join([jellyfish.soundex(word) for word in s2.split()])
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

# Função para realizar regressão linear
def regressao_linear(similarity_df):
    """Aplica regressão linear entre duas variáveis de similaridade e realiza testes de significância estatística."""
    X = similarity_df['Dzubukuá - Arcaico (Semântica)']
    y = similarity_df['Dzubukuá - Moderno (Semântica)']
    X_const = add_constant(X)
    model = OLS(y, X_const).fit()
    y_pred = model.predict(X_const)

    # Obter intervalos de confiança
    intervalo_confianca = model.conf_int(alpha=0.05)

    # Diagnósticos de resíduos
    residuos = model.resid
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuos, kde=True, ax=axs[0])
    axs[0].set_title('Distribuição dos Resíduos')
    stats.probplot(residuos, dist="norm", plot=axs[1])
    axs[1].set_title('QQ-Plot dos Resíduos')
    st.pyplot(fig)

    return model, y_pred

# Função para realizar regressão múltipla
def regressao_multipla(similarity_df):
    """Aplica regressão múltipla entre as variáveis de similaridade e realiza testes de significância estatística."""
    # Definir variáveis independentes (X) e variável dependente (y)
    X = similarity_df.drop(columns=['Dzubukuá - Moderno (Semântica)', 'Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore')
    y = similarity_df['Dzubukuá - Moderno (Semântica)']

    # Verificar multicolinearidade usando VIF
    X_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1) for i in range(len(X.columns))]
    st.write("Fatores de Inflação da Variância (VIF):")
    st.dataframe(vif_data)

    # Ajustar o modelo
    model = OLS(y, X_const).fit()

    # Diagnósticos de resíduos
    residuos = model.resid
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuos, kde=True, ax=axs[0])
    axs[0].set_title('Distribuição dos Resíduos da Regressão Múltipla')
    stats.probplot(residuos, dist="norm", plot=axs[1])
    axs[1].set_title('QQ-Plot dos Resíduos da Regressão Múltipla')
    st.pyplot(fig)

    return model

# Função para realizar análise de variância (ANOVA)
def analise_anova(similarity_df):
    """Realiza ANOVA para comparar as médias das similaridades entre grupos."""
    # Exemplo usando as similaridades semânticas
    f_stat, p_value = f_oneway(
        similarity_df['Dzubukuá - Arcaico (Semântica)'],
        similarity_df['Dzubukuá - Moderno (Semântica)'],
        similarity_df['Arcaico - Moderno (Semântica)']
    )
    st.write(f"Resultado da ANOVA:")
    st.write(f"Estatística F: {f_stat:.4f}")
    st.write(f"Valor-p: {p_value:.4e}")
    if p_value < 0.05:
        st.write("Conclusão: As médias das similaridades semânticas diferem significativamente entre os grupos.")
    else:
        st.write("Conclusão: Não há evidência suficiente para afirmar que as médias diferem entre os grupos.")

# Função para realizar testes de hipóteses
def testes_hipotese(similarity_df):
    """Realiza testes de hipóteses estatísticas nas similaridades."""
    # Teste t para duas amostras independentes
    t_stat, p_value = stats.ttest_ind(
        similarity_df['Dzubukuá - Arcaico (Semântica)'],
        similarity_df['Dzubukuá - Moderno (Semântica)']
    )
    st.write(f"Teste t para duas amostras independentes (Semântica):")
    st.write(f"Estatística t: {t_stat:.4f}")
    st.write(f"Valor-p: {p_value:.4e}")
    if p_value < 0.05:
        st.write("Conclusão: Há uma diferença significativa entre as médias das similaridades semânticas.")
    else:
        st.write("Conclusão: Não há evidência suficiente para afirmar que as médias diferem.")

# Função para ajustar uma distribuição q-exponencial
def ajuste_q_exponencial(data):
    """Ajusta uma distribuição q-exponencial aos dados e retorna os parâmetros."""
    def q_exponential(x, a, b, q):
        return a * (1 - (1 - q) * b * x) ** (1 / (1 - q))

    hist, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Ajuste dos parâmetros
    params, _ = curve_fit(q_exponential, bin_centers, hist, maxfev=10000)
    a, b, q = params
    return a, b, q

# Função para aplicar PCA e plotar
def aplicar_pca_and_plot(similarity_df):
    """Reduz a dimensionalidade usando PCA e plota os componentes principais."""
    pca = PCA(n_components=2)
    features = similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore')
    pca_result = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=100)
    ax.set_title('Análise de Componentes Principais (PCA)', fontsize=16, pad=20)
    ax.set_xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.set_ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para realizar análise de clustering e visualizar
def analise_clustering_and_visualize(similarity_df):
    """Realiza análise de clustering e visualiza os clusters."""
    # Padronizar os dados
    scaler = StandardScaler()
    features = similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore')
    data_scaled = scaler.fit_transform(features)

    # K-Means clustering
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(data_scaled)
        distortions.append(kmeanModel.inertia_)

    # Plotar o gráfico do método Elbow
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('Número de Clusters')
    ax.set_ylabel('Distortion')
    ax.set_title('Método Elbow para Determinação do Número Ótimo de Clusters')
    st.pyplot(fig)

    # Escolher k com base no método Elbow ou outras considerações
    k = 3  # Ajuste conforme necessário
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_scaled)
    similarity_df['Cluster_KMeans'] = kmeans_labels

    # Avaliar o modelo K-Means usando o coeficiente de silhueta
    silhouette_avg = silhouette_score(data_scaled, kmeans_labels)
    st.write(f"Coeficiente de Silhueta para K-Means: {silhouette_avg:.4f}")

    # DBSCAN clustering
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data_scaled)
    similarity_df['Cluster_DBSCAN'] = dbscan_labels

    # Avaliar o modelo DBSCAN (ignorar ruído, label = -1)
    labels_unique = np.unique(dbscan_labels)
    labels_unique = labels_unique[labels_unique != -1]
    if len(labels_unique) > 1:
        silhouette_avg_dbscan = silhouette_score(data_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        st.write(f"Coeficiente de Silhueta para DBSCAN: {silhouette_avg_dbscan:.4f}")
    else:
        st.write("DBSCAN não encontrou clusters significativos.")

    # Visualizar clusters
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(features)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para K-Means
    scatter = axs[0].scatter(data_pca[:, 0], data_pca[:, 1], c=similarity_df['Cluster_KMeans'], cmap='Set1', s=50)
    axs[0].set_title('Clusters K-Means', fontsize=16)
    axs[0].set_xlabel('Componente Principal 1', fontsize=14)
    axs[0].set_ylabel('Componente Principal 2', fontsize=14)
    axs[0].grid(True, linestyle='--', linewidth=0.5)

    # Gráfico para DBSCAN
    scatter = axs[1].scatter(data_pca[:, 0], data_pca[:, 1], c=similarity_df['Cluster_DBSCAN'], cmap='Set1', s=50)
    axs[1].set_title('Clusters DBSCAN', fontsize=16)
    axs[1].set_xlabel('Componente Principal 1', fontsize=14)
    axs[1].set_ylabel('Componente Principal 2', fontsize=14)
    axs[1].grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar gráfico interativo de correlação
def grafico_interativo_plotly(similarity_df):
    """Gera um mapa de calor interativo da matriz de correlação com Plotly."""
    # Calcula a matriz de correlação
    corr = similarity_df.corr()

    # Cria um mapa de calor da matriz de correlação
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        colorbar=dict(title='Coeficiente de Correlação')
    ))

    # Ajusta o layout para melhor legibilidade
    fig.update_layout(
        title="Mapa de Correlação entre Similaridades",
        xaxis_tickangle=-45,
        xaxis={'side': 'bottom'},
        width=800,
        height=800,
        margin=dict(l=200, r=200, b=200, t=100),
        font=dict(size=10),
    )

    st.plotly_chart(fig)

# Função para gerar gráficos de correlação
def mapas_de_correlacoes(similarity_df):
    """Gera mapas de correlações para áreas semântica, lexical e fonológica."""
    # Correlações Semânticas
    st.markdown("### Correlações Semânticas")
    semantic_columns = ['Dzubukuá - Arcaico (Semântica)', 'Dzubukuá - Moderno (Semântica)', 'Arcaico - Moderno (Semântica)']
    semantic_df = similarity_df[semantic_columns]
    pearson_corr_sem = semantic_df.corr(method='pearson')
    sns.heatmap(pearson_corr_sem, annot=True, cmap='coolwarm')
    st.pyplot()

    # Correlações Lexicais
    st.markdown("### Correlações Lexicais")
    lexical_columns = ['Dzubukuá - Arcaico (N-gramas)', 'Dzubukuá - Moderno (N-gramas)', 'Arcaico - Moderno (N-gramas)',
                       'Dzubukuá - Arcaico (Word2Vec)', 'Dzubukuá - Moderno (Word2Vec)', 'Arcaico - Moderno (Word2Vec)']
    lexical_df = similarity_df[lexical_columns]
    pearson_corr_lex = lexical_df.corr(method='pearson')
    sns.heatmap(pearson_corr_lex, annot=True, cmap='coolwarm')
    st.pyplot()

    # Correlações Fonológicas
    st.markdown("### Correlações Fonológicas")
    phonological_columns = ['Dzubukuá - Arcaico (Fonológica)', 'Dzubukuá - Moderno (Fonológica)', 'Arcaico - Moderno (Fonológica)']
    phonological_df = similarity_df[phonological_columns]
    pearson_corr_phon = phonological_df.corr(method='pearson')
    sns.heatmap(pearson_corr_phon, annot=True, cmap='coolwarm')
    st.pyplot()

# Função para gerar dendrograma
def grafico_dendrograma(similarity_df):
    """Gera um dendrograma para visualizar relações hierárquicas entre as variáveis."""
    from scipy.cluster.hierarchy import linkage

    linked = linkage(similarity_df.T, 'ward', metric='euclidean')
    labelList = similarity_df.columns
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linked, labels=labelList, ax=ax, orientation='top')
    ax.set_title('Dendrograma das Similaridades', fontsize=16, pad=20)
    ax.set_xlabel('Variáveis', fontsize=14, labelpad=15)
    ax.set_ylabel('Distância Euclidiana', fontsize=14, labelpad=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para salvar o DataFrame
def salvar_dataframe(similarity_df):
    """Permite o download do DataFrame em formato CSV."""
    csv = similarity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Similaridades em CSV",
        data=csv,
        file_name='similaridades_linguisticas.csv',
        mime='text/csv',
    )

# Função para gerar DataFrame de n-gramas
def gerar_dataframe_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Gera um DataFrame contendo os n-gramas (unigrama, bigrama, trigrama) para cada frase."""
    from sklearn.feature_extraction.text import CountVectorizer

    # Função auxiliar para gerar n-gramas de um conjunto de frases
    def extrair_ngramas(sentences, n, idioma):
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')
        X = vectorizer.fit_transform(sentences)
        ngram_names = vectorizer.get_feature_names_out()
        df_ngrams = pd.DataFrame(X.toarray(), columns=ngram_names)
        df_ngrams['Idioma'] = idioma
        df_ngrams['N-grama'] = f'{n}-grama'
        return df_ngrams

    # Gerar DataFrames de n-gramas para cada idioma e para cada n
    n_values = [1, 2, 3]  # unigrama, bigrama, trigrama
    dataframes = []

    for n in n_values:
        df_dzubukua = extrair_ngramas(sentences_dzubukua, n, 'Dzubukuá')
        df_arcaico = extrair_ngramas(sentences_arcaico, n, 'Português Arcaico')
        df_moderno = extrair_ngramas(sentences_moderno, n, 'Português Moderno')

        dataframes.extend([df_dzubukua, df_arcaico, df_moderno])

    # Concatenar todos os DataFrames
    df_ngramas = pd.concat(dataframes, ignore_index=True)

    return df_ngramas

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

        # Gerar o DataFrame de n-gramas
        st.subheader("DataFrame de N-gramas (Unigrama, Bigrama, Trigrama)")
        df_ngramas = gerar_dataframe_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.dataframe(df_ngramas)

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

        # Inicializar o estado da sessão
        if 'analysis_step' not in st.session_state:
            st.session_state['analysis_step'] = 0

        # Função para avançar para a próxima análise
        def next_analysis():
            st.session_state['analysis_step'] += 1

        # Lista de análises
        analyses = [
            ("Mapa de Correlação entre Similaridades", grafico_interativo_plotly),
            ("Análise de Regressão Linear entre Dzubukuá e Português Moderno (Semântica)", regressao_linear),
            ("Análise de Regressão Múltipla", regressao_multipla),
            ("Testes de Hipóteses Estatísticas", testes_hipotese),
            ("Análise de Variância (ANOVA)", analise_anova),
            ("Ajuste q-Exponencial", ajuste_q_exponencial),
            ("Análise de Componentes Principais (PCA)", aplicar_pca_and_plot),
            ("Análise de Agrupamentos (Clustering)", analise_clustering_and_visualize),
            ("Mapas de Correlações nas Áreas Lexical, Semântica e Fonológica", mapas_de_correlacoes),
            ("Análise de Agrupamento Hierárquico (Dendrograma)", grafico_dendrograma)
        ]

        # Executar a análise atual
        if st.session_state['analysis_step'] < len(analyses):
            analysis_name, analysis_function = analyses[st.session_state['analysis_step']]
            st.write(f"Deseja realizar a seguinte análise? **{analysis_name}**")
            if st.button("Sim", key=f"button_{st.session_state['analysis_step']}"):
                # Chamar a função correspondente
                if analysis_name == "Análise de Regressão Linear entre Dzubukuá e Português Moderno (Semântica)":
                    model_linear, y_pred_linear = regressao_linear(similarity_df)
                    st.write(model_linear.summary())
                elif analysis_name == "Análise de Regressão Múltipla":
                    model_multipla = regressao_multipla(similarity_df)
                    st.write(model_multipla.summary())
                elif analysis_name == "Ajuste q-Exponencial":
                    a, b, q = ajuste_q_exponencial(similarity_df['Dzubukuá - Moderno (Semântica)'])
                    st.write(f"Parâmetros ajustados:")
                    st.write(f"a = {a:.4f}")
                    st.write(f"b = {b:.4f}")
                    st.write(f"q = {q:.4f}")
                    st.write("O parâmetro q indica o grau de não-extensividade da distribuição, relevante em sistemas complexos.")
                else:
                    # Chamar a função normalmente
                    analysis_function(similarity_df)
                next_analysis()
        else:
            st.write("Todas as análises foram concluídas.")

        # Opção para baixar os resultados
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

if __name__ == '__main__':
    main()
