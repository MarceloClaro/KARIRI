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

# Função para gerar gráficos de correlações semânticas
def grafico_matriz_semantica(pearson_corr, spearman_corr, kendall_corr):
    """Gera gráficos para as correlações Semânticas."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Correlação Semântica de Pearson', pad=20)

    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Correlação Semântica de Spearman', pad=20)

    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', ax=axs[2])
    axs[2].set_title('Correlação Semântica de Kendall', pad=20)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar gráficos de correlações lexicais
def grafico_matriz_lexical(pearson_corr, spearman_corr, kendall_corr):
    """Gera gráficos para as correlações Lexicais."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Correlação Lexical de Pearson', pad=20)

    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', ax=axs[1])
    axs[1].set_title('Correlação Lexical de Spearman', pad=20)

    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', ax=axs[2])
    axs[2].set_title('Correlação Lexical de Kendall', pad=20)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para calcular similaridade semântica usando Sentence-BERT
def calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade semântica usando o modelo Sentence-BERT."""
    all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
    embeddings = model.encode(all_sentences, batch_size=32, clean_up_tokenization_spaces=True)

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
    def ngramas(sentences, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), binary=True, analyzer='word').fit(sentences)
        ngram_matrix = vectorizer.transform(sentences).toarray()
        return ngram_matrix

    ngramas_dzubukua = ngramas(sentences_dzubukua, n)
    ngramas_arcaico = ngramas(sentences_arcaico, n)
    ngramas_moderno = ngramas(sentences_moderno, n)

    num_frases = min(len(ngramas_dzubukua), len(ngramas_arcaico), len(ngramas_moderno))

    ngramas_dzubukua = ngramas_dzubukua[:num_frases]
    ngramas_arcaico = ngramas_arcaico[:num_frases]
    ngramas_moderno = ngramas_moderno[:num_frases]

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

# Função principal para rodar a aplicação no Streamlit
def main():
    st.title('Análises Estatísticas e Visualizações Avançadas para Dados Linguísticos')

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
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_arcaico_dzubukua_sem, similarity_moderno_dzubukua_sem, similarity_arcaico_moderno_sem = calcular_similaridade_semantica(
            model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Similaridade Lexical (N-gramas)
        similarity_arcaico_dzubukua_ng, similarity_moderno_dzubukua_ng, similarity_arcaico_moderno_ng = calcular_similaridade_ngramas(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Criando DataFrame com as similaridades calculadas
        similarity_df = pd.DataFrame({
            'Dzubukuá - Arcaico (Semântica)': similarity_arcaico_dzubukua_sem,
            'Dzubukuá - Moderno (Semântica)': similarity_moderno_dzubukua_sem,
            'Arcaico - Moderno (Semântica)': similarity_arcaico_moderno_sem,
            'Dzubukuá - Arcaico (N-gramas)': similarity_arcaico_dzubukua_ng,
            'Dzubukuá - Moderno (N-gramas)': similarity_moderno_dzubukua_ng,
            'Arcaico - Moderno (N-gramas)': similarity_arcaico_moderno_ng
        })

        # Exibir o DataFrame das similaridades
        st.subheader("Similaridade Calculada entre as Três Línguas")
        st.dataframe(similarity_df)

        # Correlações Semânticas (Pearson, Spearman, Kendall)
        st.subheader("Correlação Semântica entre as Similaridades")
        pearson_corr_sem, spearman_corr_sem, kendall_corr_sem = calcular_correlacoes_avancadas(
            similarity_df[['Dzubukuá - Arcaico (Semântica)', 'Dzubukuá - Moderno (Semântica)', 'Arcaico - Moderno (Semântica)']]
        )
        grafico_matriz_semantica(pearson_corr_sem, spearman_corr_sem, kendall_corr_sem)

        # Correlações Lexicais (Pearson, Spearman, Kendall)
        st.subheader("Correlação Lexical entre as Similaridades")
        pearson_corr_lex, spearman_corr_lex, kendall_corr_lex = calcular_correlacoes_avancadas(
            similarity_df[['Dzubukuá - Arcaico (N-gramas)', 'Dzubukuá - Moderno (N-gramas)', 'Arcaico - Moderno (N-gramas)']]
        )
        grafico_matriz_lexical(pearson_corr_lex, spearman_corr_lex, kendall_corr_lex)

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

# Função principal para rodar a aplicação no Streamlit
if __name__ == '__main__':
    main()

