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
    from sklearn.linear_model import LinearRegression
    X = similarity_df['Dzubukuá - Arcaico'].values.reshape(-1, 1)
    y = similarity_df['Dzubukuá - Moderno'].values
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

# Função para gerar gráficos de margem de erro
def grafico_margem_erro(margem_erro):
    """Gera gráfico de barras para margem de erro."""
    fig, ax = plt.subplots()
    margem_erro.plot(kind='bar', color=['blue', 'green', 'red'], ax=ax)
    ax.set_title('Margem de Erro das Estimativas de Similaridade')
    ax.set_ylabel('Margem de Erro')
    st.pyplot(fig)

# Função para gerar gráficos de ANOVA
def grafico_anova(fvalue, pvalue):
    """Gera gráfico de barras com os valores F e P da ANOVA."""
    fig, ax = plt.subplots()
    ax.bar(['F-value', 'P-value'], [fvalue, pvalue], color=['blue', 'green'])
    ax.set_title('Resultados da ANOVA')
    ax.set_ylabel('Valor')
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
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico'], 
                             y=similarity_df['Dzubukuá - Moderno'], 
                             mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=similarity_df['Dzubukuá - Arcaico'], 
                             y=y_pred, mode='lines', name='Regressão Linear'))
    fig.update_layout(title="Regressão Linear - Dzubukuá vs. Moderno",
                      xaxis_title="Similaridade Dzubukuá - Arcaico",
                      yaxis_title="Similaridade Dzubukuá - Moderno")
    st.plotly_chart(fig)

# Função para realizar análise de componentes principais e plotar
def grafico_pca(similarity_df, pca_result):
    """Plota os resultados da Análise de Componentes Principais (PCA)."""
    fig = plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=50)
    plt.title('Análise de Componentes Principais (PCA)')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    st.pyplot(fig)

# Função para gerar Pairplot (visualiza múltiplas correlações)
def grafico_pairplot(similarity_df):
    """Gera um Pairplot para visualizar múltiplas correlações. Apenas colunas numéricas serão usadas."""
    # Verificar se há colunas numéricas no DataFrame
    numeric_df = similarity_df.select_dtypes(include=['float64', 'int64'])
    
    if numeric_df.empty:
        st.error("O Pairplot requer colunas numéricas. O DataFrame atual não contém colunas numéricas suficientes.")
    else:
        fig = sns.pairplot(numeric_df)
        st.pyplot(fig)

# Função para calcular as similaridades de cosseno
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

# Função para calcular a margem de erro
def calcular_margem_erro(similarity_df, confidence=0.95):
    """Calcula a margem de erro com base nas médias das similaridades."""
    mean = similarity_df.mean()
    std_err = similarity_df.sem()
    margin_of_error = std_err * stats.t.ppf((1 + confidence) / 2., len(similarity_df) - 1)
    return margin_of_error

# Função para realizar ANOVA
def calcular_anova(similarity_df):
    """Realiza ANOVA para verificar se há diferenças significativas entre as médias das similaridades."""
    fvalue, pvalue = stats.f_oneway(similarity_df['Dzubukuá - Arcaico'], 
                                    similarity_df['Dzubukuá - Moderno'], 
                                    similarity_df['Arcaico - Moderno'])
    return fvalue, pvalue

# Função para calcular Q-Exponencial
def calcular_q_exponencial(similarity_df):
    """Aplica o conceito de Q-exponencial para análise não-linear."""
    q = 1.5  # Valor de ajuste da função Q
    return np.exp(-q * similarity_df)

# Função para salvar o dataframe como CSV para download
def salvar_dataframe(similarity_df):
    """Permite o download do DataFrame em formato CSV."""
    csv = similarity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Similaridades em CSV",
        data=csv,
        file_name='similaridades_semanticas.csv',
        mime='text/csv',
    )

# Função para exibir e paginar dataset
def exibir_dataset(df):
    """Exibe o dataset com paginação."""
    total_rows = len(df)
    
    # Controle deslizante para selecionar o número de linhas
    linhas_exibir = st.slider("Quantas linhas deseja exibir?", min_value=5, max_value=total_rows, value=10, step=5)

   ----
    # Exibir as primeiras linhas
    st.write(f"Exibindo as primeiras {linhas_exibir} de {total_rows} linhas:")
    st.dataframe(df.head(linhas_exibir))

    # Opção para exibir todas as linhas
    if st.checkbox("Exibir todas as linhas (pode impactar a performance)"):
        st.write(df)

    # Botão para baixar o CSV completo
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar CSV completo",
        data=csv,
        file_name='dataset_completo.csv',
        mime='text/csv',
    )

# Função principal para rodar a aplicação no Streamlit
def main():
    """Função principal da aplicação no Streamlit."""
    # Título da aplicação
    st.title('Análises Estatísticas e Visualizações Avançadas para Dados Linguísticos')

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    # Se o arquivo foi carregado
    if uploaded_file is not None:
        # Carregar o arquivo CSV
        df = pd.read_csv(uploaded_file)

        # Exibir dataset
        st.write("Primeiras linhas do dataset:")
        st.dataframe(df.head())

        # Similaridade semântica usando Sentence-BERT
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno = calcular_similaridade_semantica(
            model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

        similarity_df = pd.DataFrame({
            'Dzubukuá - Arcaico': similarity_arcaico_dzubukua,
            'Dzubukuá - Moderno': similarity_moderno_dzubukua,
            'Arcaico - Moderno': similarity_arcaico_moderno
        })

        # Correlações Avançadas
        st.subheader("Correlação entre as Similaridades (Pearson, Spearman, Kendall)")
        pearson_corr, spearman_corr, kendall_corr = calcular_correlacoes_avancadas(similarity_df)
        grafico_matriz_correlacao(pearson_corr, spearman_corr, kendall_corr)

        # Regressão Linear
        st.subheader("Análise de Regressão Linear entre as Similaridades")
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

        # Pairplot para visualizar múltiplas correlações
        st.subheader("Pairplot para Visualizar Múltiplas Correlações")
        grafico_pairplot(similarity_df)

        # Margem de Erro
        st.subheader("Margem de Erro para as Estimativas de Similaridade")
        margem_erro = calcular_margem_erro(similarity_df)
        st.write(f"Margem de Erro: {margem_erro}")
        grafico_margem_erro(margem_erro)

        # ANOVA (Análise de Variância)
        st.subheader("Análise de Variância (ANOVA) entre as Similaridades")
        fvalue, pvalue = calcular_anova(similarity_df)
        st.write(f"F-value: {fvalue}, P-value: {pvalue}")
        grafico_anova(fvalue, pvalue)

        # Análise de Q-Exponencial
        st.subheader("Análise de Padrões Não-Lineares usando Q-Exponencial")
        q_exponencial_result = calcular_q_exponencial(similarity_df)
        st.write("Resultados da análise Q-Exponencial:")
        st.write(q_exponencial_result)

        # Perguntar se o usuário deseja baixar o CSV com os resultados
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

        # Exibir o dataset completo
        st.subheader("Dataset Completo")
        exibir_dataset(df)

# Rodar a aplicação Streamlit
if __name__ == '__main__':
    main()

