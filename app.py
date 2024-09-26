# Instalar as bibliotecas necessárias

# Importar as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

# Função para calcular as correlações de comprimento das frases
def calcular_correlacao_comprimento(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    length_dzubukua = [len(x) for x in sentences_dzubukua]
    length_arcaico = [len(x) for x in sentences_arcaico]
    length_moderno = [len(x) for x in sentences_moderno]
    
    correlation_arcaico_dzubukua = pd.Series(length_dzubukua).corr(pd.Series(length_arcaico))
    correlation_moderno_dzubukua = pd.Series(length_dzubukua).corr(pd.Series(length_moderno))
    correlation_arcaico_moderno = pd.Series(length_arcaico).corr(pd.Series(length_moderno))
    
    return length_dzubukua, length_arcaico, length_moderno, correlation_arcaico_dzubukua, correlation_moderno_dzubukua, correlation_arcaico_moderno

# Função para gerar gráficos de dispersão de correlação de comprimento
def grafico_dispersao_comprimento(length_dzubukua, length_arcaico, length_moderno):
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
    # Gráfico de Barras para Similaridade Semântica
    fig, ax = plt.subplots()
    similarity_df.mean().plot(kind='bar', color=['blue', 'green', 'red'], ax=ax)
    ax.set_title('Média da Similaridade Semântica entre Idiomas')
    ax.set_ylabel('Similaridade de Cosseno')
    st.pyplot(fig)

    # Histogramas de distribuição de similaridade
    fig, ax = plt.subplots()
    sns.histplot(similarity_df['Dzubukuá - Arcaico'], color='blue', label='Dzubukuá - Arcaico', kde=True, ax=ax)
    sns.histplot(similarity_df['Dzubukuá - Moderno'], color='green', label='Dzubukuá - Moderno', kde=True, ax=ax)
    sns.histplot(similarity_df['Arcaico - Moderno'], color='red', label='Arcaico - Moderno', kde=True, ax=ax)
    ax.set_title('Distribuição da Similaridade Semântica entre Idiomas')
    ax.legend()
    st.pyplot(fig)

# Função para salvar o dataframe como CSV
def salvar_dataframe(similarity_df):
    csv = similarity_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Similaridades em CSV",
        data=csv,
        file_name='similaridades_semanticas.csv',
        mime='text/csv',
    )

# Função para gerar dicionário a partir do dataframe
def gerar_dicionario_de_similaridade(similarity_df):
    dicionario_similaridade = {
        'Dzubukuá - Arcaico': similarity_df['Dzubukuá - Arcaico'].tolist(),
        'Dzubukuá - Moderno': similarity_df['Dzubukuá - Moderno'].tolist(),
        'Arcaico - Moderno': similarity_df['Arcaico - Moderno'].tolist(),
        'Estatísticas': similarity_df.describe().to_dict()  # Estatísticas descritivas adicionadas como um sub-dicionário
    }
    return dicionario_similaridade

# Função para perguntar ao usuário se ele deseja continuar para o próximo módulo
def perguntar_continuacao(mensagem):
    return st.radio(mensagem, ('Sim', 'Não')) == 'Sim'

# Função principal para rodar a aplicação no Streamlit
def main():
    # Título da aplicação
    st.title('Análise de Correlação e Similaridade Semântica entre Dzubukuá, Português Arcaico e Português Moderno')
    
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        # Carregar o arquivo CSV
        df = pd.read_csv(uploaded_file)
        st.write("Primeiras linhas do dataset:", df.head())

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

        # Perguntar se o usuário deseja continuar para a próxima etapa
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

            # Perguntar se o usuário deseja continuar para a próxima etapa
            if perguntar_continuacao("Deseja gerar um dicionário de similaridades?"):
                # Gerar dicionário a partir do dataframe
                dicionario_similaridade = gerar_dicionario_de_similaridade(similarity_df)

                # Exibir o dicionário
                st.subheader('Dicionário de Similaridade Gerado')
                st.json(dicionario_similaridade)

                # Perguntar se o usuário deseja baixar o CSV
                if perguntar_continuacao("Deseja baixar os resultados como CSV?"):
                    salvar_dataframe(similarity_df)

        else:
            st.write("Operação finalizada. Selecione outro arquivo ou reinicie o processo.")

# Rodar a aplicação Streamlit
if __name__ == '__main__':
    main()
