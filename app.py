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
from sklearn.metrics import jaccard_score
import scipy.stats as stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import jellyfish

# Função para calcular correlações de Pearson, Spearman e Kendall
def calcular_correlacoes_avancadas(similarity_df):
    """Calcula as correlações de Pearson, Spearman e Kendall entre as variáveis de similaridade."""
    pearson_corr = similarity_df.corr(method='pearson')
    spearman_corr = similarity_df.corr(method='spearman')
    kendall_corr = similarity_df.corr(method='kendall')
    return pearson_corr, spearman_corr, kendall_corr

# Função para realizar regressão linear com teste de significância
def regressao_linear(similarity_df):
    """Aplica regressão linear entre as variáveis de similaridade e realiza testes de significância estatística."""
    X = similarity_df['Dzubukuá - Arcaico (Semântica)'].values.reshape(-1, 1)
    y = similarity_df['Dzubukuá - Moderno (Semântica)'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)

    # Teste de significância estatística
    n = len(X)
    p = 1  # Número de preditores
    dof = n - p - 1  # Graus de liberdade
    t_stat = reg.coef_ / (np.sqrt((1 - r2) / dof))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    return y_pred, r2, p_value[0]

# Função para aplicar PCA (Análise de Componentes Principais)
def aplicar_pca(similarity_df):
    """Reduz a dimensionalidade usando PCA para entender os padrões nas similaridades."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(similarity_df)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# Função para gerar o gráfico de PCA
def grafico_pca(similarity_df, pca_result, explained_variance):
    """Plota os resultados da Análise de Componentes Principais (PCA)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=100)
    ax.set_title('Análise de Componentes Principais (PCA)', fontsize=16, pad=20)
    ax.set_xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.set_ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar dendrograma (Análise de Agrupamento Hierárquico)
def grafico_dendrograma(similarity_df):
    """Gera um dendrograma para visualizar relações hierárquicas entre as variáveis."""
    linked = linkage(similarity_df.T, 'single', metric='euclidean')
    labelList = similarity_df.columns
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked, labels=labelList, ax=ax, orientation='top')
    ax.set_title('Dendrograma das Similaridades', fontsize=16, pad=20)
    ax.set_xlabel('Variáveis', fontsize=14, labelpad=15)
    ax.set_ylabel('Distância Euclidiana', fontsize=14, labelpad=15)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar gráficos de correlações
def grafico_matriz_correlacao(pearson_corr, spearman_corr, kendall_corr):
    """Gera gráficos para as correlações Pearson, Spearman e Kendall."""
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Correlação de Pearson
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Correlação de Pearson', pad=20)
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].tick_params(axis='y', rotation=0)
    axs[0].set_xlabel('Variáveis', labelpad=15)
    axs[0].set_ylabel('Variáveis', labelpad=15)

    # Correlação de Spearman
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Correlação de Spearman', pad=20)
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].tick_params(axis='y', rotation=0)
    axs[1].set_xlabel('Variáveis', labelpad=15)
    axs[1].set_ylabel('Variáveis', labelpad=15)

    # Correlação de Kendall
    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', ax=axs[2])
    axs[2].set_title('Correlação de Kendall', pad=20)
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].tick_params(axis='y', rotation=0)
    axs[2].set_xlabel('Variáveis', labelpad=15)
    axs[2].set_ylabel('Variáveis', labelpad=15)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar gráficos interativos com Plotly
def grafico_interativo_plotly(similarity_df):
    """Gera gráficos interativos com Plotly."""
    fig = px.scatter_matrix(similarity_df, dimensions=similarity_df.columns, title="Correlação entre Similaridades")
    fig.update_layout(
        height=800,
        width=800,
        xaxis_tickangle=-45,
        yaxis_tickangle=45,
        margin=dict(l=50, r=50, b=100, t=100, pad=10),
        font=dict(size=12),
    )
    st.plotly_chart(fig)

# Função para gerar gráficos de dispersão interativos com Plotly
def grafico_regressao_plotly(similarity_df, y_pred, r2, p_value):
    """Gera gráfico interativo com a linha de regressão e informações estatísticas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico (Semântica)'], 
                             y=similarity_df['Dzubukuá - Moderno (Semântica)'], 
                             mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico (Semântica)'], 
                             y=y_pred, mode='lines', name='Regressão Linear'))
    fig.update_layout(
        title=f"Regressão Linear - R²: {r2:.2f}, p-value: {p_value:.4f}",
        xaxis_title="Similaridade Dzubukuá - Arcaico (Semântica)",
        yaxis_title="Similaridade Dzubukuá - Moderno (Semântica)"
    )
    st.plotly_chart(fig)

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
        return 2 * np.sum(a & b) / (np.sum(a) + np.sum(b))

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
            # Codificação fonética usando Double Metaphone
            s1_phonetic = ''.join(jellyfish.metaphone(s1))
            s2_phonetic = ''.join(jellyfish.metaphone(s2))
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

# Função principal para rodar a aplicação no Streamlit
def main():
    st.title('Análises Avançadas de Similaridade Linguística para Línguas Mortas')

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Exibir a tabela completa do dataset
        st.subheader("Tabela Completa do Dataset")
        st.dataframe(df)

        # Extrair frases de cada idioma
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

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

        # Gráfico interativo de correlações usando Plotly
        st.subheader("Gráfico Interativo de Correlações entre Similaridades")
        grafico_interativo_plotly(similarity_df)

        # Regressão Linear entre Dzubukuá e Moderno
        st.subheader("Análise de Regressão Linear entre Dzubukuá e Português Moderno (Semântica)")
        y_pred, r2, p_value = regressao_linear(similarity_df)
        st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
        st.write(f"Valor-p da Regressão: {p_value:.4f}")
        grafico_regressao_plotly(similarity_df, y_pred, r2, p_value)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_result, explained_variance = aplicar_pca(similarity_df)
        grafico_pca(similarity_df, pca_result, explained_variance)

        # Mapas de Correlações nas Áreas Lexical, Semântica e Fonológica
        st.subheader("Mapas de Correlações nas Áreas Lexical, Semântica e Fonológica")

        # Correlações Semânticas
        st.markdown("### Correlações Semânticas")
        semantic_columns = ['Dzubukuá - Arcaico (Semântica)', 'Dzubukuá - Moderno (Semântica)', 'Arcaico - Moderno (Semântica)']
        semantic_df = similarity_df[semantic_columns]
        pearson_corr_sem, spearman_corr_sem, kendall_corr_sem = calcular_correlacoes_avancadas(semantic_df)
        grafico_matriz_correlacao(pearson_corr_sem, spearman_corr_sem, kendall_corr_sem)

        # Correlações Lexicais
        st.markdown("### Correlações Lexicais")
        lexical_columns = ['Dzubukuá - Arcaico (N-gramas)', 'Dzubukuá - Moderno (N-gramas)', 'Arcaico - Moderno (N-gramas)',
                           'Dzubukuá - Arcaico (Word2Vec)', 'Dzubukuá - Moderno (Word2Vec)', 'Arcaico - Moderno (Word2Vec)']
        lexical_df = similarity_df[lexical_columns]
        pearson_corr_lex, spearman_corr_lex, kendall_corr_lex = calcular_correlacoes_avancadas(lexical_df)
        grafico_matriz_correlacao(pearson_corr_lex, spearman_corr_lex, kendall_corr_lex)

        # Correlações Fonológicas
        st.markdown("### Correlações Fonológicas")
        phonological_columns = ['Dzubukuá - Arcaico (Fonológica)', 'Dzubukuá - Moderno (Fonológica)', 'Arcaico - Moderno (Fonológica)']
        phonological_df = similarity_df[phonological_columns]
        pearson_corr_phon, spearman_corr_phon, kendall_corr_phon = calcular_correlacoes_avancadas(phonological_df)
        grafico_matriz_correlacao(pearson_corr_phon, spearman_corr_phon, kendall_corr_phon)

        # Dendrograma (Análise de Agrupamento Hierárquico)
        st.subheader("Análise de Agrupamento Hierárquico")
        grafico_dendrograma(similarity_df)

        # Perguntar se o usuário deseja baixar os resultados como CSV
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

# Função para salvar o dataframe como CSV para download
def salvar_dataframe(similarity_df):
    """Permite o download do DataFrame em formato CSV."""
    csv = similarity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Similaridades em CSV",
        data=csv,
        file_name='similaridades_linguisticas.csv',
        mime='text/csv',
    )

# Importações adicionais necessárias
from scipy.cluster.hierarchy import linkage

# Função principal para rodar a aplicação no Streamlit
if __name__ == '__main__':
    main()
