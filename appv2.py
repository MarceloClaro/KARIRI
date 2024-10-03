# Importar as bibliotecas necessárias
import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
import jellyfish
from scipy.stats import pearsonr, kendalltau

# Modelo de Sentence-BERT
model_bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Função para tokenizar frases
def tokenizacao(frases):
    return [frase.split() for frase in frases]

# Função para calcular similaridade semântica usando Sentence-BERT
def calcular_similaridade_semantica(frases_dzubukua, frases_arcaico, frases_moderno):
    embeddings_dzubukua = model_bert.encode(frases_dzubukua)
    embeddings_arcaico = model_bert.encode(frases_arcaico)
    embeddings_moderno = model_bert.encode(frases_moderno)

    sim_dzubukua_arcaico = cosine_similarity(embeddings_dzubukua, embeddings_arcaico).diagonal()
    sim_dzubukua_moderno = cosine_similarity(embeddings_dzubukua, embeddings_moderno).diagonal()
    sim_arcaico_moderno = cosine_similarity(embeddings_arcaico, embeddings_moderno).diagonal()

    return sim_dzubukua_arcaico, sim_dzubukua_moderno, sim_arcaico_moderno

# Função para calcular a análise lexical usando Word2Vec e N-gramas
def calcular_similaridade_lexical(frases_dzubukua, frases_arcaico, frases_moderno):
    # Word2Vec
    tokenized_sentences = tokenizacao(frases_dzubukua + frases_arcaico + frases_moderno)
    model_w2v = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)

    def sentence_vector(sentence):
        words = sentence.split()
        word_vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model_w2v.vector_size)

    # N-gramas
    vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')
    def ngrams_similaridade(frases1, frases2):
        vec1 = vectorizer.fit_transform(frases1).toarray()
        vec2 = vectorizer.transform(frases2).toarray()
        return cosine_similarity(vec1, vec2).diagonal()

    w2v_dzubukua_arcaico = cosine_similarity([sentence_vector(f) for f in frases_dzubukua], [sentence_vector(f) for f in frases_arcaico]).diagonal()
    ngrams_dzubukua_arcaico = ngrams_similaridade(frases_dzubukua, frases_arcaico)

    return w2v_dzubukua_arcaico, ngrams_dzubukua_arcaico

# Função para calcular a análise fonológica com Soundex e distância de Levenshtein
def calcular_similaridade_fonologica(frases_dzubukua, frases_arcaico, frases_moderno):
    def soundex_levenshtein(f1, f2):
        soundex1 = jellyfish.soundex(f1)
        soundex2 = jellyfish.soundex(f2)
        return jellyfish.levenshtein_distance(soundex1, soundex2)

    similarity_fon_dzubukua_arcaico = [soundex_levenshtein(f1, f2) for f1, f2 in zip(frases_dzubukua, frases_arcaico)]
    similarity_fon_arcaico_moderno = [soundex_levenshtein(f1, f2) for f1, f2 in zip(frases_arcaico, frases_moderno)]
    similarity_fon_dzubukua_moderno = [soundex_levenshtein(f1, f2) for f1, f2 in zip(frases_dzubukua, frases_moderno)]

    return similarity_fon_dzubukua_arcaico, similarity_fon_arcaico_moderno, similarity_fon_dzubukua_moderno

# Função para calcular correlação de Pearson e Kendall
def calcular_correlacoes(similaridades_1, similaridades_2):
    corr_pearson, _ = pearsonr(similaridades_1, similaridades_2)
    corr_kendall, _ = kendalltau(similaridades_1, similaridades_2)
    return corr_pearson, corr_kendall

# Função para realizar regressão múltipla
def regressao_multipla(similaridades):
    X = similaridades[['Similaridade de Cosseno', 'Soundex', 'Distância de Levenshtein']]
    y = similaridades['Word2Vec']
    modelo = LinearRegression().fit(X, y)
    return modelo

# Função para realizar ANOVA
def analise_anova(frase_grupos):
    return f_oneway(*frase_grupos)

# Função para calcular margens de erro
def calcular_margem_erro(similaridades, z=1.96):
    erro_padrao = np.std(similaridades) / np.sqrt(len(similaridades))
    margem_erro = z * erro_padrao
    return margem_erro

# Função para criar o DataFrame final com todas as análises
def criar_dataset(frases_dzubukua, frases_arcaico, frases_moderno):
    token_dzubukua = tokenizacao(frases_dzubukua)
    token_arcaico = tokenizacao(frases_arcaico)
    token_moderno = tokenizacao(frases_moderno)

    sim_semantica = calcular_similaridade_semantica(frases_dzubukua, frases_arcaico, frases_moderno)
    sim_lexical = calcular_similaridade_lexical(frases_dzubukua, frases_arcaico, frases_moderno)
    sim_fonologica = calcular_similaridade_fonologica(frases_dzubukua, frases_arcaico, frases_moderno)

    # Correlação de Pearson e Kendall
    corr_pearson, corr_kendall = calcular_correlacoes(sim_semantica[0], sim_fonologica[0])

    # Regressão Múltipla
    dataset = pd.DataFrame({
        'Similaridade de Cosseno': sim_semantica[0],
        'Word2Vec': sim_lexical[0],
        'N-gramas': sim_lexical[1],
        'Soundex': sim_fonologica[0],
        'Distância de Levenshtein': sim_fonologica[0],
        'Correlação de Pearson': [corr_pearson] * len(frases_dzubukua),
        'Correlação de Kendall': [corr_kendall] * len(frases_dzubukua)
    })

    modelo_reg = regressao_multipla(dataset)

    # ANOVA
    anova_result = analise_anova([dataset['Similaridade de Cosseno'], dataset['Word2Vec'], dataset['N-gramas']])

    # Margem de Erro
    dataset['Margens de Erro'] = calcular_margem_erro(dataset['Similaridade de Cosseno'])

      # Criação do DataFrame com todas as análises incluindo as colunas do arquivo de entrada
    df = pd.DataFrame({
        'Idioma': ['Dzubukuá', 'Português Arcaico', 'Português Moderno'] * len(frases_dzubukua),
        'Texto Original': frases_dzubukua + frases_arcaico + frases_moderno,
        'Tradução para o Português Moderno': frases_moderno * 3,
        'Tokenização': [' '.join(token) for token in token_dzubukua + token_arcaico + token_moderno],
        'Similaridade de Cosseno': list(sim_semantica[0]) + list(sim_semantica[1]) + list(sim_semantica[2]),
        'Word2Vec': [list(w2v) for w2v in sim_lexical[0]] * 3,
        'N-gramas': list(sim_lexical[1]) * 3,
        'Soundex': list(sim_fonologica[0]) + list(sim_fonologica[1]) + list(sim_fonologica[2]),
        'Distância de Levenshtein': list(sim_fonologica[0]) + list(sim_fonologica[1]) + list(sim_fonologica[2]),
        'Correlação de Pearson': [corr_pearson] * len(frases_dzubukua) * 3,
        'Correlação de Kendall': [corr_kendall] * len(frases_dzubukua) * 3,
        'Regressão Múltipla': [modelo_reg.coef_.tolist()] * len(frases_dzubukua) * 3,
        'ANOVA': [anova_result.statistic] * len(frases_dzubukua) * 3,
        'Margens de Erro': [calcular_margem_erro(sim_semantica[0])] * len(frases_dzubukua) * 3,
        'Análise Fonológica': ['[IPA]'] * len(frases_dzubukua) * 3,
        'Análise Morfológica': ['Subst, Verbo, Obj'] * len(frases_dzubukua) * 3,
        'Análise Sintática': ['SVO'] * len(frases_dzubukua) * 3,
        'Glossário Cultural': ['Termo religioso'] * len(frases_dzubukua) * 3,
        'Justificativa da Tradução': ['Preserva divindade'] * len(frases_dzubukua) * 3,
        'Etimologia': ['Origem indígena de "Senhor"'] * len(frases_dzubukua) * 3
    })

    # Mostra o DataFrame no Streamlit
    st.write(df)

    # Opção de baixar o DataFrame como CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV", data=csv, file_name='analises_linguisticas.csv')

# Upload do arquivo CSV original para comparações
uploaded_file = st.file_uploader("Carregue o CSV com as colunas do usuário", type="csv")

if uploaded_file:
    df_original = pd.read_csv(uploaded_file)
    st.write("Dados originais carregados:")
    st.write(df_original)

    # Combina os dados originais com o novo dataset
    df_completo = pd.concat([df_original, df], axis=1)
    st.write("Dados combinados:")
    st.write(df_completo)

    # Opção de baixar o DataFrame completo
    csv_completo = df_completo.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV completo", data=csv_completo, file_name='analises_completas.csv')
