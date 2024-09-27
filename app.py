# Importar as bibliotecas necessárias
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import stats
from sklearn.decomposition import PCA
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import jaccard_score

# Função para calcular correlações de Pearson, Spearman e Kendall
def calcular_correlacoes_avancadas(similarity_df):
    """Calcula as correlações de Pearson, Spearman e Kendall."""
    pearson_corr = similarity_df.corr(method='pearson')
    spearman_corr = similarity_df.corr(method='spearman')
    kendall_corr = similarity_df.corr(method='kendall')
    return pearson_corr, spearman_corr, kendall_corr

# Função para realizar regressão linear
def regressao_linear(similarity_df):
    """Aplica regressão linear entre as variáveis de similaridade."""
    X = similarity_df['Dzubukuá - Arcaico (Semântica)'].values.reshape(-1, 1)
    y = similarity_df['Dzubukuá - Moderno (Semântica)'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)
    return y_pred, r2

# Função para aplicar PCA (Análise de Componentes Principais)
def aplicar_pca(similarity_df):
    """Reduz a dimensionalidade usando PCA para entender os padrões nas similaridades."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(similarity_df)
    return pca_result

# Função para gerar o gráfico de PCA com títulos e legendas claros
def grafico_pca(similarity_df, pca_result):
    """Plota os resultados da Análise de Componentes Principais (PCA) com legendas simplificadas."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Ajuste do tamanho do gráfico para melhorar a visualização
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=100)  # Tamanho dos pontos aumentado

    # Título claro e informativo
    ax.set_title('Distribuição das Similaridades', fontsize=16, pad=20)

    # Legendas mais compreensíveis
    ax.set_xlabel('Primeiro Padrão Principal', fontsize=14, labelpad=15)  # Eixo X
    ax.set_ylabel('Segundo Padrão Principal', fontsize=14, labelpad=15)   # Eixo Y

    # Adicionar grades leves para melhor visualização
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ajustar espaçamento para as margens e elementos do gráfico
    plt.tight_layout(pad=3.0)

    # Exibir gráfico no Streamlit
    st.pyplot(fig)

# Função para gerar gráficos de correlações
def grafico_matriz_correlacao(pearson_corr, spearman_corr, kendall_corr):
    """Gera gráficos para as correlações Pearson, Spearman e Kendall."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Correlação de Pearson')

    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Correlação de Spearman')

    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', ax=axs[2])
    axs[2].set_title('Correlação de Kendall')

    st.pyplot(fig)

# Função para gerar gráficos interativos com Plotly
def grafico_interativo_plotly(similarity_df):
    """Gera gráficos interativos com Plotly."""
    fig = px.scatter_matrix(similarity_df, dimensions=similarity_df.columns, title="Correlação entre Similaridades")
    st.plotly_chart(fig)

# Função para gerar gráficos de dispersão interativos com Plotly
def grafico_regressao_plotly(similarity_df, y_pred):
    """Gera gráfico interativo com a linha de regressão."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico (Semântica)'], 
                             y=similarity_df['Dzubukuá - Moderno (Semântica)'], 
                             mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico (Semântica)'], 
                             y=y_pred, mode='lines', name='Regressão Linear'))
    fig.update_layout(title="Regressão Linear - Dzubukuá vs. Moderno (Semântica)",
                      xaxis_title="Similaridade Dzubukuá - Arcaico (Semântica)",
                      yaxis_title="Similaridade Dzubukuá - Moderno (Semântica)")
    st.plotly_chart(fig)

# Função para calcular similaridade semântica usando Sentence-BERT
def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade semântica usando o modelo Sentence-BERT."""
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    embeddings = model.encode(all_sentences, batch_size=32)

    embeddings_dzubukua = embeddings[:len(sentences_dzubukua)]
    embeddings_arcaico = embeddings[len(sentences_dzubukua):len(sentences_dzubukua) + len(sentences_arcaico)]
    embeddings_moderno = embeddings[len(sentences_dzubukua) + len(sentences_arcaico):]

    similarity_arcaico_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_arcaico).diagonal()
    similarity_moderno_dzubukua = cosine_similarity(embeddings_dzubukua, embeddings_moderno).diagonal()
    similarity_arcaico_moderno = cosine_similarity(embeddings_arcaico, embeddings_moderno).diagonal()
    
    return similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno

# Função para calcular similaridade de N-gramas
def calcular_similaridade_ngramas(sentences_dzubukua, sentences_arcaico, sentences_moderno, n=2):
    """Calcula a similaridade lexical usando N-gramas e Jaccard Similarity."""
    from sklearn.metrics import jaccard_score
    from sklearn.feature_extraction.text import CountVectorizer

    # Função para gerar N-gramas binários
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), binary=True, analyzer='word').fit(sentences)
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
    ngramas_dzubukua = ngramas_dzubukua[:, :min(ngramas_dzubukua.shape[1], ngramas_arcaico.shape[1], ngramas_moderno.shape[1])]
    ngramas_arcaico = ngramas_arcaico[:, :ngramas_dzubukua.shape[1]]
    ngramas_moderno = ngramas_moderno[:, :ngramas_dzubukua.shape[1]]

    # Calculando a similaridade de Jaccard entre os N-gramas
    similarity_arcaico_dzubukua = [
        jaccard_score(ngramas_dzubukua[i], ngramas_arcaico[i], average='binary') 
        for i in range(num_frases)
    ]
    similarity_moderno_dzubukua = [
        jaccard_score(ngramas_dzubukua[i], ngramas_moderno[i], average='binary') 
        for i in range(num_frases)
    ]
    similarity_arcaico_moderno = [
        jaccard_score(ngramas_arcaico[i], ngramas_moderno[i], average='binary') 
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

# Função principal para rodar a aplicação no Streamlit
def main():
    st.title('Análises Estatísticas e Visualizações Avançadas para Dados Linguísticos')

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Extrair frases de cada idioma
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

        # Similaridade Semântica (Sentence-BERT)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_arcaico_dzubukua_sem, similarity_moderno_dzubukua_sem, similarity_arcaico_moderno_sem = calcular_similaridade_semantica(
            model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Lexical (N-gramas)
        similarity_arcaico_dzubukua_ng, similarity_moderno_dzubukua_ng, similarity_arcaico_moderno_ng = calcular_similaridade_ngramas(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Lexical (Word2Vec)
        similarity_arcaico_dzubukua_w2v, similarity_moderno_dzubukua_w2v, similarity_arcaico_moderno_w2v = calcular_similaridade_word2vec(
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
            'Arcaico - Moderno (Word2Vec)': similarity_arcaico_moderno_w2v
        })

        # Exibir o DataFrame das similaridades
        st.subheader("Similaridade Calculada entre as Três Línguas")
        st.dataframe(similarity_df)

        # Correlações Avançadas (Pearson, Spearman, Kendall)
        st.subheader("Correlação entre as Similaridades (Pearson, Spearman, Kendall)")
        pearson_corr, spearman_corr, kendall_corr = calcular_correlacoes_avancadas(similarity_df)
        grafico_matriz_correlacao(pearson_corr, spearman_corr, kendall_corr)

        # Regressão Linear entre Dzubukuá e Moderno
        st.subheader("Análise de Regressão Linear entre Dzubukuá e Português Moderno")
        y_pred, r2 = regressao_linear(similarity_df)
        st.write(f"Coeficiente de Determinação (R²) da Regressão Linear: {r2:.2f}")
        grafico_regressao_plotly(similarity_df, y_pred)

        # Gráfico interativo de correlações usando Plotly
        st.subheader("Gráfico Interativo de Correlações entre Similaridades")
        grafico_interativo_plotly(similarity_df)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_result = aplicar_pca(similarity_df)
        grafico_pca(similarity_df, pca_result)

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
        file_name='similaridades_semanticas_lexicais.csv',
        mime='text/csv',
    )

# Rodar a aplicação Streamlit
if __name__ == '__main__':
    main()
