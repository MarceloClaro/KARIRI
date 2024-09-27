# Importar as bibliotecas necessárias
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import stats
import numpy as np

# Função para calcular as correlações de comprimento das frases
def calcular_correlacao_comprimento(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a correlação de comprimento entre as três línguas."""
    length_dzubukua = [len(x) for x in sentences_dzubukua]
    length_arcaico = [len(x) for x in sentences_arcaico]
    length_moderno = [len(x) for x in sentences_moderno]
    
    correlation_arcaico_dzubukua = pd.Series(length_dzubukua).corr(pd.Series(length_arcaico))
    correlation_moderno_dzubukua = pd.Series(length_dzubukua).corr(pd.Series(length_moderno))
    correlation_arcaico_moderno = pd.Series(length_arcaico).corr(pd.Series(length_moderno))
    
    return length_dzubukua, length_arcaico, length_moderno, correlation_arcaico_dzubukua, correlation_moderno_dzubukua, correlation_arcaico_moderno

# Função para calcular ANOVA
def calcular_anova(similarity_df):
    """Realiza ANOVA para verificar se há diferenças significativas entre as médias das similaridades."""
    fvalue, pvalue = stats.f_oneway(similarity_df['Dzubukuá - Arcaico'], similarity_df['Dzubukuá - Moderno'], similarity_df['Arcaico - Moderno'])
    return fvalue, pvalue

# Função para calcular margem de erro
def calcular_margem_erro(similarity_df, confidence=0.95):
    """Calcula a margem de erro com base nas médias das similaridades."""
    mean = similarity_df.mean()
    std_err = similarity_df.sem()
    margin_of_error = std_err * stats.t.ppf((1 + confidence) / 2., len(similarity_df) - 1)
    return margin_of_error

# Função para calcular Q-Exponencial (Tsallis)
def calcular_q_exponencial(similarity_df):
    """Aplica o conceito de Q-exponencial para análise não-linear."""
    q = 1.5  # Valor de ajuste da função Q
    return np.exp(-q * similarity_df)

# Função para gerar gráficos de dispersão de correlação de comprimento
def grafico_dispersao_comprimento(length_dzubukua, length_arcaico, length_moderno):
    """Gera gráfico de dispersão para visualização das correlações de comprimento."""
    fig, ax = plt.subplots()
    ax.scatter(length_dzubukua, length_arcaico, c='blue', label='Dzubukuá vs. Arcaico', alpha=0.6)
    ax.scatter(length_dzubukua, length_moderno, c='green', label='Dzubukuá vs. Moderno', alpha=0.6)
    ax.scatter(length_arcaico, length_moderno, c='red', label='Arcaico vs. Moderno', alpha=0.6)
    ax.set_title('Correlação de Comprimento de Frases entre Idiomas')
    ax.set_xlabel('Comprimento das Frases em Idioma Base')
    ax.set_ylabel('Comprimento Comparado')
    ax.legend(loc='upper left')
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

# Função para gerar gráficos de barras e histogramas
def grafico_barras_histogramas(similarity_df):
    """Gera gráficos de barras e histogramas para visualizar as similaridades."""
    # Gráfico de Barras
    fig, ax = plt.subplots()
    similarity_df.mean().plot(kind='bar', color=['blue', 'green', 'red'], ax=ax)
    ax.set_title('Média da Similaridade Semântica entre Idiomas')
    ax.set_ylabel('Similaridade de Cosseno')
    st.pyplot(fig)

    # Histogramas de distribuição
    fig, ax = plt.subplots()
    sns.histplot(similarity_df['Dzubukuá - Arcaico'], color='blue', label='Dzubukuá - Arcaico', kde=True, ax=ax)
    sns.histplot(similarity_df['Dzubukuá - Moderno'], color='green', label='Dzubukuá - Moderno', kde=True, ax=ax)
    sns.histplot(similarity_df['Arcaico - Moderno'], color='red', label='Arcaico - Moderno', kde=True, ax=ax)
    ax.set_title('Distribuição da Similaridade Semântica entre Idiomas')
    ax.legend()
    st.pyplot(fig)

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

# Função para gerar o heatmap de correlação
def grafico_heatmap(similarity_df):
    """Gera um heatmap de correlação entre as variáveis."""
    corr = similarity_df.corr()

    # Criar o heatmap
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap das Correlações')
    st.pyplot(fig)

# Função para perguntar ao usuário se deseja continuar
def perguntar_continuacao(mensagem):
    """Exibe uma pergunta ao usuário com uma opção de 'Sim' ou 'Não'."""
    return st.radio(mensagem, ('Sim', 'Não')) == 'Sim'

# Função principal para rodar a aplicação no Streamlit
def main():
    """Função principal da aplicação no Streamlit."""

    st.title('Análise de Correlação e Similaridade Semântica entre Dzubukuá, Português Arcaico e Português Moderno')

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    # Se o arquivo foi carregado
    if uploaded_file is not None:
        # Carregar o arquivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Exibir dataset com opção de paginação
        exibir_dataset(df)

        # Garantir que o número de frases seja o mesmo entre os três grupos
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

        min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
        sentences_dzubukua = sentences_dzubukua[:min_length]
        sentences_arcaico = sentences_arcaico[:min_length]
        sentences_moderno = sentences_moderno[:min_length]

        # Correlação de Comprimento
        st.subheader('Correlação de Comprimento de Frases')
        length_dzubukua, length_arcaico, length_moderno, correlation_arcaico_dzubukua, correlation_moderno_dzubukua, correlation_arcaico_moderno = calcular_correlacao_comprimento(sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Exibir correlações
        st.write(f'Correlação de comprimento Dzubukuá - Arcaico: {correlation_arcaico_dzubukua:.2f}')
        st.write(f'Correlação de comprimento Dzubukuá - Moderno: {correlation_moderno_dzubukua:.2f}')
        st.write(f'Correlação de comprimento Arcaico - Moderno: {correlation_arcaico_moderno:.2f}')

        # Mostrar gráfico de dispersão
        grafico_dispersao_comprimento(length_dzubukua, length_arcaico, length_moderno)

        # Perguntar se o usuário deseja continuar para o cálculo de similaridade semântica
        if perguntar_continuacao("Deseja continuar para o cálculo de similaridade semântica?"):
            # Calcular a similaridade semântica
            st.subheader('Cálculo de Similaridade Semântica usando Sentence-BERT')
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            similarity_arcaico_dzubukua, similarity_moderno_dzubukua, similarity_arcaico_moderno = calcular_similaridade_semantica(model, sentences_dzubukua, sentences_arcaico, sentences_moderno)

            # Criar dataframe para visualização e download
            similarity_df = pd.DataFrame({
                'Dzubukuá - Arcaico': similarity_arcaico_dzubukua,
                'Dzubukuá - Moderno': similarity_moderno_dzubukua,
                'Arcaico - Moderno': similarity_arcaico_moderno
            })

            # Mostrar gráficos de barras e histogramas
            grafico_barras_histogramas(similarity_df)

            # Mostrar estatísticas descritivas
            st.subheader('Estatísticas Descritivas das Similaridades de Cosseno')
            st.write(similarity_df.describe())

            # Calcular a margem de erro
            st.subheader('Margem de Erro para Similaridades')
            margin_of_error = calcular_margem_erro(similarity_df)
            st.write(margin_of_error)

            # Realizar ANOVA
            st.subheader('ANOVA: Análise de Variância')
            fvalue, pvalue = calcular_anova(similarity_df)
            st.write(f'F-value: {fvalue}, P-value: {pvalue}')

            # Aplicar Q-Exponencial
            st.subheader('Q-Exponencial para Análise Não-Linear')
            q_exponencial_df = calcular_q_exponencial(similarity_df)
            st.write(q_exponencial_df)

            # Perguntar se o usuário deseja visualizar o heatmap das correlações
            if perguntar_continuacao("Deseja visualizar o heatmap das correlações?"):
                st.subheader("Heatmap das Correlações")
                grafico_heatmap(similarity_df)

            # Perguntar se o usuário deseja baixar o CSV
            if perguntar_continuacao("Deseja baixar os resultados como CSV?"):
                salvar_dataframe(similarity_df)

        else:
            st.write("Operação finalizada. Selecione outro arquivo ou reinicie o processo.")

# Rodar a aplicação Streamlit
if __name__ == '__main__':
    main()
