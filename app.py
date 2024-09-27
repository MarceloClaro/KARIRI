# Importar as bibliotecas necessárias
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
import jellyfish
import re
import spacy
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.manifold import TSNE
import statsmodels.api as sm
from Levenshtein import distance as levenshtein_distance
from collections import Counter

# Tentar carregar o modelo de língua portuguesa para análise morfológica
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    st.error("O modelo 'pt_core_news_sm' não está instalado. Por favor, execute 'python -m spacy download pt_core_news_sm' no terminal para instalar o modelo e reinicie a aplicação.")
    st.stop()

# Função para limpeza e normalização de dados
def limpar_normalizar_texto(text):
    """Limpa e normaliza o texto, tratando variações ortográficas e caracteres especiais."""
    # Converter para minúsculas
    text = text.lower()
    # Remover diacríticos
    text = re.sub(r'[áàãâä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[íìîï]', 'i', text)
    text = re.sub(r'[óòõôö]', 'o', text)
    text = re.sub(r'[úùûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    # Remover caracteres especiais
    text = re.sub(r'[^a-z\s]', '', text)
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função para anotação linguística usando spaCy
def anotar_texto(text):
    """Anota o texto com informações morfológicas e sintáticas."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct]
    return ' '.join(tokens)

# Função para codificação fonética customizada
def codificacao_fonetica_customizada(text):
    """Codifica o texto usando regras fonéticas específicas das línguas estudadas."""
    # Exemplo de regras fonéticas simplificadas
    text = re.sub(r'ph', 'f', text)
    text = re.sub(r'th', 't', text)
    text = re.sub(r'gh', 'g', text)
    # Adicionar mais regras conforme necessário
    return text

# Função para cálculo de similaridade fonológica usando sequências de fonemas
def calcular_similaridade_fonologica_fonemas(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Calcula a similaridade fonológica usando sequências de fonemas e distância de Levenshtein."""

    def preprocess(sentence):
        # Limpar e normalizar
        sentence = limpar_normalizar_texto(sentence)
        # Codificação fonética customizada
        sentence = codificacao_fonetica_customizada(sentence)
        # Transformar em sequência de fonemas (simplificado)
        # Em uma implementação real, utilizar um transcritor fonético
        return list(sentence)

    # Preprocessar as frases
    sentences_dzubukua_phonemes = [preprocess(s) for s in sentences_dzubukua]
    sentences_arcaico_phonemes = [preprocess(s) for s in sentences_arcaico]
    sentences_moderno_phonemes = [preprocess(s) for s in sentences_moderno]

    # Garantir que as listas tenham o mesmo comprimento
    min_length = min(len(sentences_dzubukua_phonemes), len(sentences_arcaico_phonemes), len(sentences_moderno_phonemes))
    sentences_dzubukua_phonemes = sentences_dzubukua_phonemes[:min_length]
    sentences_arcaico_phonemes = sentences_arcaico_phonemes[:min_length]
    sentences_moderno_phonemes = sentences_moderno_phonemes[:min_length]

    # Calcular similaridades
    similarity_arcaico_dzubukua_phon = []
    similarity_moderno_dzubukua_phon = []
    similarity_arcaico_moderno_phon = []

    for i in range(min_length):
        s1 = sentences_dzubukua_phonemes[i]
        s2 = sentences_arcaico_phonemes[i]
        s3 = sentences_moderno_phonemes[i]

        # Similaridade entre Dzubukuá e Arcaico
        dist = levenshtein_distance(''.join(s1), ''.join(s2))
        max_len = max(len(s1), len(s2))
        similarity = 1 - (dist / max_len) if max_len > 0 else 1
        similarity_arcaico_dzubukua_phon.append(similarity)

        # Similaridade entre Dzubukuá e Moderno
        dist = levenshtein_distance(''.join(s1), ''.join(s3))
        max_len = max(len(s1), len(s3))
        similarity = 1 - (dist / max_len) if max_len > 0 else 1
        similarity_moderno_dzubukua_phon.append(similarity)

        # Similaridade entre Arcaico e Moderno
        dist = levenshtein_distance(''.join(s2), ''.join(s3))
        max_len = max(len(s2), len(s3))
        similarity = 1 - (dist / max_len) if max_len > 0 else 1
        similarity_arcaico_moderno_phon.append(similarity)

    return similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon

# Função para treinar embeddings customizados (FastText)
def treinar_embeddings(sentences):
    """Treina embeddings customizados usando FastText."""
    tokenized_sentences = [sentence.split() for sentence in sentences]
    model = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Função para calcular similaridade lexical usando embeddings customizados
def calcular_similaridade_embeddings_customizados(model, sentences1, sentences2):
    """Calcula a similaridade lexical usando embeddings customizados."""
    def sentence_vector(sentence, model):
        words = sentence.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    vectors1 = [sentence_vector(sentence, model) for sentence in sentences1]
    vectors2 = [sentence_vector(sentence, model) for sentence in sentences2]

    similarity = cosine_similarity(vectors1, vectors2).diagonal()
    return similarity

# Função para análise de correspondências sonoras
def analisar_correspondencias_sonoras(sentences_dzubukua, sentences_arcaico, sentences_moderno):
    """Analisa correspondências sonoras entre as línguas."""
    # Implementação de exemplo simplificado
    # Em uma aplicação real, utilizar análises fonológicas detalhadas
    correspondencias = []
    for dz, ar, mo in zip(sentences_dzubukua, sentences_arcaico, sentences_moderno):
        dz_phon = codificacao_fonetica_customizada(limpar_normalizar_texto(dz))
        ar_phon = codificacao_fonetica_customizada(limpar_normalizar_texto(ar))
        mo_phon = codificacao_fonetica_customizada(limpar_normalizar_texto(mo))
        correspondencias.append({'Dzubukuá': dz_phon, 'Arcaico': ar_phon, 'Moderno': mo_phon})
    return correspondencias

# Função para gerar árvore filogenética
def gerar_arvore_filogenetica(similarity_df):
    """Gera uma árvore filogenética baseada nas similaridades."""
    import plotly.figure_factory as ff

    # Calcular distância
    dist_matrix = 1 - similarity_df.corr()
    labels = similarity_df.columns.tolist()
    fig = ff.create_dendrogram(dist_matrix.values, orientation='left', labels=labels)
    fig.update_layout(width=800, height=600)
    st.subheader("Árvore Filogenética das Similaridades")
    st.plotly_chart(fig)

# Função para regressão linear avançada com verificação de suposições
def regressao_linear_avancada(X, y):
    """Aplica regressão linear e verifica as suposições do modelo."""
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)

    # Resíduos
    residuals = y - y_pred

    # Teste de normalidade dos resíduos
    stat, p_normal = stats.shapiro(residuals)

    # Teste de homocedasticidade (Breusch-Pagan)
    _, p_homoscedasticity, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, sm.add_constant(X))

    # Teste de significância estatística
    n = len(X)
    p = X.shape[1]
    dof = n - p - 1
    t_stat = reg.coef_ / (np.sqrt((1 - r2) / dof))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    # Exibir resultados dos testes
    st.write(f"Teste de Normalidade dos Resíduos (Shapiro-Wilk): p-valor = {p_normal:.4f}")
    st.write(f"Teste de Homocedasticidade (Breusch-Pagan): p-valor = {p_homoscedasticity:.4f}")

    return y_pred, r2, p_value[0], residuals

# Função para gerar gráfico de regressão personalizado
def grafico_regressao_plotly_custom(X, y, y_pred, r2, p_value):
    """Gera gráfico interativo com a linha de regressão e informações estatísticas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name='Regressão Linear'))
    fig.update_layout(
        title=f"Regressão Linear - R²: {r2:.2f}, p-valor: {p_value:.4f}",
        xaxis_title="Similaridade Dzubukuá - Arcaico (Embeddings)",
        yaxis_title="Similaridade Dzubukuá - Moderno (Embeddings)",
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14)),
        width=800,
        height=600,
        margin=dict(l=100, r=100, b=100, t=100),
        font=dict(size=12),
    )
    st.plotly_chart(fig)

# Função para gráficos de diagnóstico da regressão
def grafico_diagnostico_regressao(X, y, y_pred, residuals):
    """Gera gráficos de diagnóstico para a regressão linear."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Resíduos vs. Valores Ajustados
    axs[0].scatter(y_pred, residuals)
    axs[0].axhline(y=0, color='r', linestyle='--')
    axs[0].set_xlabel('Valores Ajustados')
    axs[0].set_ylabel('Resíduos')
    axs[0].set_title('Resíduos vs. Valores Ajustados')

    # Gráfico Q-Q dos Resíduos
    sm.qqplot(residuals, line='s', ax=axs[1])
    axs[1].set_title('Gráfico Q-Q dos Resíduos')

    plt.tight_layout()
    st.pyplot(fig)

# Função para visualizar embeddings
def visualizar_embeddings(model, words):
    """Visualiza os embeddings das palavras fornecidas."""
    word_vectors = []
    labels = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
            labels.append(word)
    if len(word_vectors) < 2:
        st.write("Não há palavras suficientes para visualização.")
        return

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(word_vectors)

    fig, ax = plt.subplots()
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    for i, label in enumerate(labels):
        ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    st.pyplot(fig)

# Função para gerar heatmap de correspondências sonoras
def gerar_heatmap_correspondencias(correspondencias_sonoras):
    """Gera um heatmap de correspondências sonoras entre as línguas."""
    pares_correspondencias = []
    for c in correspondencias_sonoras:
        pares_correspondencias.append((c['Dzubukuá'], c['Arcaico']))
        pares_correspondencias.append((c['Dzubukuá'], c['Moderno']))
        pares_correspondencias.append((c['Arcaico'], c['Moderno']))

    # Contar frequências
    contagem_pares = Counter(pares_correspondencias)

    # Criar DataFrame para o heatmap
    df_heatmap = pd.DataFrame(list(contagem_pares.items()), columns=['Pares', 'Frequência'])
    df_heatmap[['Lingua1', 'Lingua2']] = pd.DataFrame(df_heatmap['Pares'].tolist(), index=df_heatmap.index)
    pivot_table = df_heatmap.pivot_table(values='Frequência', index='Lingua1', columns='Lingua2', fill_value=0)

    # Plotar o heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=False, cmap='Blues', ax=ax)
    ax.set_title('Heatmap de Correspondências Sonoras')
    st.pyplot(fig)

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
    for i, txt in enumerate(similarity_df.index):
        ax.annotate(txt, (pca_result[i, 0], pca_result[i, 1]))
    ax.set_title('Análise de Componentes Principais (PCA)', fontsize=16, pad=20)
    ax.set_xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.set_ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

# Função para gerar o mapa de calor interativo da matriz de correlação
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

# Função principal para rodar a aplicação no Streamlit
def main():
    st.title('Análises Avançadas de Similaridade Linguística para Línguas Mortas')

    # Seção de Metodologia
    st.sidebar.title("Metodologia")
    st.sidebar.markdown("""
    Este aplicativo realiza análises avançadas de similaridade linguística entre línguas mortas e modernas. As etapas incluem:

    1. **Limpeza e Normalização de Dados:** Remoção de variações ortográficas e diacríticos para padronizar o texto.
    2. **Anotação Linguística:** Utilização do spaCy para anotar o texto com informações morfológicas.
    3. **Treinamento de Embeddings Customizados:** Treinamento de modelos FastText para capturar relações semânticas e morfológicas.
    4. **Análise Fonética e Fonológica:** Codificação fonética customizada e cálculo de similaridades fonológicas usando distância de Levenshtein.
    5. **Análise de Correspondências Sonoras:** Identificação de correspondências entre sons nas diferentes línguas.
    6. **Análises Estatísticas e Visualizações:** Aplicação de regressão linear, PCA, e geração de visualizações para interpretar os resultados.
    """)

    # Seção de Considerações Éticas
    st.sidebar.title("Considerações Éticas")
    st.sidebar.markdown("""
    Este estudo reconhece a importância de respeitar as comunidades e culturas associadas às línguas estudadas. Buscamos:

    - **Colaboração com Especialistas e Comunidades:** Envolver linguistas e membros das comunidades para validar e interpretar os resultados.
    - **Respeito Cultural:** Garantir que o uso dos dados linguísticos seja sensível às práticas e tradições culturais.
    - **Transparência e Atribuição:** Agradecer e atribuir adequadamente as contribuições de indivíduos e organizações envolvidos.
    """)

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Limpar e normalizar os textos
        df['Texto Limpo'] = df['Texto Original'].apply(limpar_normalizar_texto)
        df['Texto Anotado'] = df['Texto Limpo'].apply(anotar_texto)

        # Exibir exemplos de textos processados
        st.subheader("Exemplos de Textos Processados")
        exemplos_df = df[['Idioma', 'Texto Original', 'Texto Limpo', 'Texto Anotado']].head(5)
        st.dataframe(exemplos_df)

        # Extrair frases de cada idioma
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Anotado'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Anotado'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].apply(limpar_normalizar_texto).tolist()

        # Garantir que todas as listas tenham o mesmo comprimento
        min_length = min(len(sentences_dzubukua), len(sentences_arcaico), len(sentences_moderno))
        sentences_dzubukua = sentences_dzubukua[:min_length]
        sentences_arcaico = sentences_arcaico[:min_length]
        sentences_moderno = sentences_moderno[:min_length]

        # Treinar embeddings customizados
        st.info("Treinando embeddings customizados...")
        all_sentences = sentences_dzubukua + sentences_arcaico + sentences_moderno
        embeddings_model = treinar_embeddings(all_sentences)

        # Visualização dos embeddings
        st.subheader("Visualização dos Embeddings")
        palavras_para_visualizar = ['palavra1', 'palavra2', 'palavra3']  # Substitua por palavras relevantes
        visualizar_embeddings(embeddings_model, palavras_para_visualizar)

        # Similaridade Lexical (Embeddings Customizados)
        st.info("Calculando similaridade lexical (Embeddings Customizados)...")
        similarity_arcaico_dzubukua_emb = calcular_similaridade_embeddings_customizados(
            embeddings_model, sentences_dzubukua, sentences_arcaico)
        similarity_moderno_dzubukua_emb = calcular_similaridade_embeddings_customizados(
            embeddings_model, sentences_dzubukua, sentences_moderno)
        similarity_arcaico_moderno_emb = calcular_similaridade_embeddings_customizados(
            embeddings_model, sentences_arcaico, sentences_moderno)

        # Similaridade Fonológica (Sequências de Fonemas)
        st.info("Calculando similaridade fonológica (Sequências de Fonemas)...")
        similarity_arcaico_dzubukua_phon, similarity_moderno_dzubukua_phon, similarity_arcaico_moderno_phon = calcular_similaridade_fonologica_fonemas(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)

        # Exibir exemplos de codificação fonética
        st.subheader("Exemplos de Codificação Fonética")
        exemplos_fonetica = pd.DataFrame({
            'Dzubukuá': [codificacao_fonetica_customizada(s) for s in sentences_dzubukua[:5]],
            'Arcaico': [codificacao_fonetica_customizada(s) for s in sentences_arcaico[:5]],
            'Moderno': [codificacao_fonetica_customizada(s) for s in sentences_moderno[:5]]
        })
        st.dataframe(exemplos_fonetica)

        # Análise de Correspondências Sonoras
        st.info("Analisando correspondências sonoras...")
        correspondencias_sonoras = analisar_correspondencias_sonoras(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.subheader("Exemplos de Correspondências Sonoras")
        st.write(correspondencias_sonoras[:5])  # Exibir os 5 primeiros exemplos

        # Gerar heatmap de correspondências sonoras
        gerar_heatmap_correspondencias(correspondencias_sonoras)

        # Criando DataFrame com as similaridades calculadas
        similarity_df = pd.DataFrame({
            'Dzubukuá - Arcaico (Embeddings)': similarity_arcaico_dzubukua_emb,
            'Dzubukuá - Moderno (Embeddings)': similarity_moderno_dzubukua_emb,
            'Arcaico - Moderno (Embeddings)': similarity_arcaico_moderno_emb,
            'Dzubukuá - Arcaico (Fonológica)': similarity_arcaico_dzubukua_phon,
            'Dzubukuá - Moderno (Fonológica)': similarity_moderno_dzubukua_phon,
            'Arcaico - Moderno (Fonológica)': similarity_arcaico_moderno_phon
        })

        # Exibir o DataFrame das similaridades
        st.subheader("Similaridade Calculada entre as Três Línguas")
        st.dataframe(similarity_df)

        # Gráfico interativo de correlações usando Plotly
        st.subheader("Mapa de Correlação entre Similaridades")
        grafico_interativo_plotly(similarity_df)

        # Regressão Linear com verificação de suposições
        st.subheader("Análise de Regressão Linear entre Dzubukuá e Português Moderno (Embeddings)")
        X = similarity_df['Dzubukuá - Arcaico (Embeddings)'].values.reshape(-1, 1)
        y = similarity_df['Dzubukuá - Moderno (Embeddings)'].values
        y_pred, r2, p_value, residuals = regressao_linear_avancada(X, y)
        st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
        st.write(f"Valor-p da Regressão: {p_value:.4f}")
        grafico_regressao_plotly_custom(X, y, y_pred, r2, p_value)

        # Gráficos de diagnóstico da regressão
        grafico_diagnostico_regressao(X, y, y_pred, residuals)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_result, explained_variance = aplicar_pca(similarity_df)
        grafico_pca(similarity_df, pca_result, explained_variance)

        # Árvore Filogenética
        gerar_arvore_filogenetica(similarity_df)

        # Perguntar se o usuário deseja baixar os resultados como CSV
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)
    else:
        st.info("Por favor, faça o upload do arquivo CSV para iniciar a análise.")

if __name__ == '__main__':
    main()
