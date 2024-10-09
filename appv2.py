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
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
import jellyfish

# Bibliotecas adicionais para as novas implementações
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway  # Para ANOVA
from scipy.optimize import curve_fit  # Para q-Exponencial

import os
import logging
import base64

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
    st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
else:
    st.warning("Imagem 'capa.png' não encontrada.")

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=200)
else:
    st.sidebar.text("Imagem do logotipo não encontrada.")

st.sidebar.title("Geomaker +IA")

# [Os expanders e o controle de áudio permanecem iguais ao código original]

# Certifique-se de que todas as funções estão definidas antes do main()
# Funções de cálculo de similaridades e análises estatísticas
# [Definições das funções permanecem as mesmas que no código original]

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
                    grafico_regressao_plotly(similarity_df, model_linear, y_pred_linear)
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
                elif analysis_name == "Análise de Componentes Principais (PCA)":
                    aplicar_pca_and_plot(similarity_df)
                elif analysis_name == "Análise de Agrupamentos (Clustering)":
                    analise_clustering_and_visualize(similarity_df)
                elif analysis_name == "Mapas de Correlações nas Áreas Lexical, Semântica e Fonológica":
                    mapas_de_correlacoes(similarity_df)
                else:
                    # Chamar a função normalmente
                    analysis_function(similarity_df)
                next_analysis()
        else:
            st.write("Todas as análises foram concluídas.")

        # Opção para baixar os resultados
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

# Funções adicionais necessárias
def aplicar_pca_and_plot(similarity_df):
    pca_result, explained_variance = aplicar_pca(similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore'))
    grafico_pca(similarity_df, pca_result, explained_variance)

def analise_clustering_and_visualize(similarity_df):
    similarity_df = analise_clustering(similarity_df)
    visualizar_clusters(similarity_df)

def mapas_de_correlacoes(similarity_df):
    st.markdown("### Correlações Semânticas")
    semantic_columns = ['Dzubukuá - Arcaico (Semântica)', 'Dzubukuá - Moderno (Semântica)', 'Arcaico - Moderno (Semântica)']
    semantic_df = similarity_df[semantic_columns]
    pearson_corr_sem, spearman_corr_sem, kendall_corr_sem = calcular_correlacoes_avancadas(semantic_df)
    grafico_matriz_correlacao(pearson_corr_sem, spearman_corr_sem, kendall_corr_sem)

    st.markdown("### Correlações Lexicais")
    lexical_columns = ['Dzubukuá - Arcaico (N-gramas)', 'Dzubukuá - Moderno (N-gramas)', 'Arcaico - Moderno (N-gramas)',
                       'Dzubukuá - Arcaico (Word2Vec)', 'Dzubukuá - Moderno (Word2Vec)', 'Arcaico - Moderno (Word2Vec)']
    lexical_df = similarity_df[lexical_columns]
    pearson_corr_lex, spearman_corr_lex, kendall_corr_lex = calcular_correlacoes_avancadas(lexical_df)
    grafico_matriz_correlacao(pearson_corr_lex, spearman_corr_lex, kendall_corr_lex)

    st.markdown("### Correlações Fonológicas")
    phonological_columns = ['Dzubukuá - Arcaico (Fonológica)', 'Dzubukuá - Moderno (Fonológica)', 'Arcaico - Moderno (Fonológica)']
    phonological_df = similarity_df[phonological_columns]
    pearson_corr_phon, spearman_corr_phon, kendall_corr_phon = calcular_correlacoes_avancadas(phonological_df)
    grafico_matriz_correlacao(pearson_corr_phon, spearman_corr_phon, kendall_corr_phon)

if __name__ == '__main__':
    main()
