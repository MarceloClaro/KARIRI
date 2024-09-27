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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
import jellyfish
import re
import spacy
import networkx as nx
from ete3 import Tree, TreeStyle, NodeStyle

# Carregar modelo de língua portuguesa para análise morfológica
nlp = spacy.load('pt_core_news_sm')

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
    from Levenshtein import distance as levenshtein_distance

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
        dz = codificacao_fonetica_customizada(limpar_normalizar_texto(dz))
        ar = codificacao_fonetica_customizada(limpar_normalizar_texto(ar))
        mo = codificacao_fonetica_customizada(limpar_normalizar_texto(mo))
        correspondencias.append({'Dzubukuá': dz, 'Arcaico': ar, 'Moderno': mo})
    return correspondencias

# Função para gerar árvore filogenética
def gerar_arvore_filogenetica(similarity_df):
    """Gera uma árvore filogenética baseada nas similaridades."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, to_tree
    from ete3 import Tree, TreeStyle, NodeStyle

    # Calcular distância
    dist_matrix = 1 - similarity_df.corr()
    condensed_dist = pdist(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Construir a árvore
    tree = to_tree(linkage_matrix, rd=False)

    # Converter para formato ETE
    def build_ete_tree(scipy_tree, labels):
        if scipy_tree.is_leaf():
            return Tree(f"{labels[scipy_tree.id]};")
        else:
            left = build_ete_tree(scipy_tree.get_left(), labels)
            right = build_ete_tree(scipy_tree.get_right(), labels)
            new_tree = Tree()
            new_tree.add_child(left)
            new_tree.add_child(right)
            return new_tree

    labels = similarity_df.columns.tolist()
    ete_tree = build_ete_tree(tree, labels)

    # Visualizar a árvore
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.rotation = 90
    ts.scale = 20

    # Renderizar a árvore no Streamlit
    st.subheader("Árvore Filogenética das Similaridades")
    ete_tree.render("%%inline", w=800, tree_style=ts)

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

        # Limpar e normalizar os textos
        df['Texto Limpo'] = df['Texto Original'].apply(limpar_normalizar_texto)
        df['Texto Anotado'] = df['Texto Limpo'].apply(anotar_texto)

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

        # Análise de Correspondências Sonoras
        st.info("Analisando correspondências sonoras...")
        correspondencias_sonoras = analisar_correspondencias_sonoras(
            sentences_dzubukua, sentences_arcaico, sentences_moderno)
        st.subheader("Exemplos de Correspondências Sonoras")
        st.write(correspondencias_sonoras[:5])  # Exibir os 5 primeiros exemplos

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

        # Gráfico interativo de correlações usando Plotly (atualizado)
        st.subheader("Mapa de Correlação entre Similaridades")
        grafico_interativo_plotly(similarity_df)

        # Regressão Linear com verificação de suposições
        st.subheader("Análise de Regressão Linear entre Dzubukuá e Português Moderno (Embeddings)")
        X = similarity_df['Dzubukuá - Arcaico (Embeddings)'].values.reshape(-1, 1)
        y = similarity_df['Dzubukuá - Moderno (Embeddings)'].values
        y_pred, r2, p_value = regressao_linear_avancada(X, y)
        st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
        st.write(f"Valor-p da Regressão: {p_value:.4f}")
        grafico_regressao_plotly_custom(X.flatten(), y, y_pred, r2, p_value)

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_result, explained_variance = aplicar_pca(similarity_df)
        grafico_pca(similarity_df, pca_result, explained_variance)

        # Árvore Filogenética
        gerar_arvore_filogenetica(similarity_df)

        # Perguntar se o usuário deseja baixar os resultados como CSV
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

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
    _, p_homoscedasticity, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(residuals, X)

    # Teste de significância estatística
    n = len(X)
    p = X.shape[1]
    dof = n - p - 1
    t_stat = reg.coef_ / (np.sqrt((1 - r2) / dof))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    # Exibir resultados dos testes
    st.write(f"Teste de Normalidade dos Resíduos (Shapiro-Wilk): p-valor = {p_normal:.4f}")
    st.write(f"Teste de Homocedasticidade (Breusch-Pagan): p-valor = {p_homoscedasticity:.4f}")

    return y_pred, r2, p_value[0]

# Função para gerar gráfico de regressão personalizado
def grafico_regressao_plotly_custom(X, y, y_pred, r2, p_value):
    """Gera gráfico interativo com a linha de regressão e informações estatísticas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Regressão Linear'))
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
import statsmodels.api as sm
import statsmodels

# Função principal para rodar a aplicação no Streamlit
if __name__ == '__main__':
    main()
