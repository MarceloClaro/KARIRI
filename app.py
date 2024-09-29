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
import streamlit as st
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

# Expander de Insights
with st.sidebar.expander("Insights do Projeto"):
    st.markdown("""
        ## Sobre o Projeto
        Este aplicativo realiza análises avançadas de similaridade linguística entre três idiomas: **Dzubukuá**, **Português Arcaico** e **Português Moderno**. Utilizamos técnicas de processamento de linguagem natural (PLN) e estatística para explorar as relações entre essas línguas.
        ...[Explicações detalhadas sobre o projeto]...
    """)

# Expander de Insights do Código
with st.sidebar.expander("Insights do Código"):
       
    st.latex(r'''
    \text{### Introdução}
    
    \text{Este aplicativo web foi desenvolvido para auxiliar no estudo de três idiomas: \textbf{Dzubukuá} (uma língua morta ou em risco de extinção), \textbf{Português Arcaico} e \textbf{Português Moderno}. Ele oferece uma análise detalhada das similaridades e diferenças entre esses idiomas, utilizando técnicas avançadas de processamento de linguagem natural (PLN) e métodos estatísticos. O principal objetivo é investigar como esses idiomas se relacionam em termos de significado, construção lexical e som.}
    
    \text{No decorrer da explicação, apresentaremos os métodos utilizados, com detalhes matemáticos e justificativas para cada análise, além de exemplos práticos e insights que podem ser extraídos dos resultados. Também serão discutidas as limitações de cada técnica aplicada.}
    
    \text{---}
    
    \text{### 1. Ferramentas Utilizadas}
    
    \text{Antes de começar as análises, é importante compreender as ferramentas que foram utilizadas para realizar o estudo:}
    
    \text{- \textbf{Pandas}: Usado para manipular dados e criar estruturas organizadas, como DataFrames.}
    
    \text{- \textbf{Streamlit}: Facilita a construção de uma interface web interativa.}
    
    \text{- \textbf{Matplotlib e Seaborn}: Auxiliam na visualização de gráficos e dados.}
    
    \text{- \textbf{Plotly}: Permite a criação de gráficos interativos.}
    
    \text{- \textbf{Scikit-learn}: Fornece algoritmos de aprendizado de máquina e estatística.}
    
    \text{- \textbf{Gensim}: Implementa o modelo \textbf{Word2Vec} para trabalhar com representações de palavras.}
    
    \text{- \textbf{Jellyfish}: Facilita cálculos de similaridade fonética.}
    
    \text{- \textbf{Statsmodels e Scipy}: Conjunto de ferramentas para realizar testes estatísticos e cálculos avançados.}
    
    \text{\textbf{Objetivo}: Utilizar essas bibliotecas para analisar como os três idiomas se comparam em diferentes aspectos, como semântica, estrutura lexical e fonologia.}
    
    \text{---}
    
    \text{### 2. Carregamento e Organização dos Dados}
    
    \text{O primeiro passo é carregar o arquivo CSV que contém as frases nos três idiomas. Esse arquivo deve conter as seguintes colunas:}
    
    \text{- \textbf{Idioma}: Identifica qual das três línguas a frase pertence.}
    
    \text{- \textbf{Texto Original}: A frase no idioma original.}
    
    \text{- \textbf{Tradução para o Português Moderno}: A tradução da frase no Português Moderno.}
    
    \text{\textbf{Objetivo}: Extrair essas frases para que possamos compará-las em diferentes níveis e medir as similaridades linguísticas entre os três idiomas.}
    
    \text{---}
    
    \text{### 3. Cálculo das Similaridades}
    
    \text{As similaridades entre as frases de cada idioma são medidas em três níveis principais: \textbf{semântica} (significado), \textbf{lexical} (construção das palavras) e \textbf{fonológica} (som).}
    
    \text{#### 3.1 Similaridade Semântica com Sentence-BERT}
    
    \text{\textbf{O que é}: O \textbf{Sentence-BERT} é um modelo de linguagem que transforma frases em vetores, permitindo que possamos medir o quanto duas frases têm significados parecidos.}
    
    \text{\textbf{Como funciona}:}
    
    \text{1. \textbf{Geração de Embeddings}: Para cada frase, o modelo gera um vetor (ou "embedding") que representa o significado dessa frase. Isso é como um "resumo" matemático do conteúdo semântico da frase.}
    
    \text{Seja \( s_i \) a \( i \)-ésima frase, obtemos um vetor \( \vec{v}_i \in \mathbb{R}^d \).}
    
    \text{2. \textbf{Cálculo da Similaridade de Cosseno}: A semelhança entre duas frases é calculada comparando os seus vetores usando a métrica de \textbf{similaridade de cosseno}:}
    
    \( \text{similaridade}(\vec{v}_i, \vec{v}_j) = \cos(\theta) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \|\vec{v}_j\|} \)
    
    \text{Onde:}
    
    \text{- \( \vec{v}_i \cdot \vec{v}_j \) é o produto escalar dos vetores.}
    
    \text{- \( \|\vec{v}_i\| \) é a norma (comprimento) do vetor \( \vec{v}_i \).}
    
    \text{\textbf{Objetivo}: Descobrir se duas frases em diferentes idiomas possuem o mesmo significado, mesmo que escritas de forma diferente.}
    
    \text{\textbf{Exemplo}: Se houver uma alta similaridade semântica entre uma frase em Dzubukuá e sua tradução no Português Arcaico, isso pode significar que o significado dessa frase foi preservado ao longo do tempo.}
    
    \text{---}
    
    \text{#### 3.2 Similaridade Lexical com N-gramas}
    
    \text{\textbf{O que é}: Um \textbf{N-grama} é uma sequência de \( N \) caracteres que aparece em uma palavra ou frase. Por exemplo, na palavra "casa", os \textbf{bigramas} (N=2) seriam: "ca", "as" e "sa".}
    
    \text{\textbf{Como funciona}:}
    
    \text{1. \textbf{Extração de N-gramas}: O sistema divide cada frase em N-gramas de caracteres.}
    
    \text{Exemplo de bigramas da palavra "casa": \{ "ca", "as", "sa" \}.}
    
    \text{2. \textbf{Representação Vetorial}: Cada frase é representada como um vetor binário indicando a presença ou ausência de cada N-grama possível.}
    
    \text{3. \textbf{Cálculo do Coeficiente de Sorensen-Dice}: Para medir essa similaridade, usamos a fórmula:}
    
    \( \text{SDC}(A, B) = \frac{2 \times |A \cap B|}{|A| + |B|} \)
    
    \text{Onde:}
    
    \text{- \( A \) e \( B \) são os conjuntos de N-gramas das duas frases.}
    
    \text{- \( |A| \) é o número total de N-gramas na frase \( A \).}
    
    \text{- \( |A \cap B| \) é o número de N-gramas em comum entre \( A \) e \( B \).}
    
    \text{\textbf{Objetivo}: Avaliar a semelhança na construção das palavras entre as frases dos três idiomas.}
    
    \text{\textbf{Exemplo}: Se houver uma alta similaridade lexical entre o Português Arcaico e Moderno, isso pode indicar que muitos padrões de palavras foram mantidos ao longo dos séculos.}
    
    \text{---}
    
    \text{#### 3.3 Similaridade Lexical com Word2Vec}
    
    \text{\textbf{O que é}: O \textbf{Word2Vec} é um modelo que cria vetores para palavras baseados no contexto em que elas aparecem. Esse modelo aprende os significados das palavras ao observar como elas são usadas em diferentes frases.}
    
    \text{\textbf{Como funciona}:}
    
    \text{1. \textbf{Tokenização das Frases}: Cada frase é dividida em palavras individuais.}
    
    \text{2. \textbf{Treinamento do Modelo}: O \textbf{Word2Vec} é treinado para gerar um vetor para cada palavra \( \vec{w}_i \), baseado nas palavras que a cercam no texto.}
    
    \text{3. \textbf{Representação das Frases}: Para representar uma frase inteira, o sistema calcula a média dos vetores de todas as palavras da frase:}
    
    \( \vec{v}_{\text{frase}} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i \)
    
    \text{Onde:}
    
    \text{- \( n \) é o número de palavras na frase.}
    
    \text{- \( \vec{w}_i \) é o vetor da \( i \)-ésima palavra.}
    
    \text{4. \textbf{Cálculo da Similaridade de Cosseno}: A similaridade entre frases é medida pela similaridade de cosseno entre seus vetores médios, assim como no método semântico.}
    
    \text{\textbf{Objetivo}: Identificar semelhanças lexicais entre os idiomas com base no contexto em que as palavras aparecem.}
    
    \text{\textbf{Exemplo}: O modelo pode capturar o fato de que, mesmo que a grafia das palavras tenha mudado, seu uso em certos contextos permaneceu similar.}
    
    \text{---}
    
    \text{#### 3.4 Similaridade Fonológica}
    
    \text{\textbf{O que é}: A análise fonológica se preocupa com o som das palavras. Podemos avaliar a semelhança de como as palavras \textbf{soam}, mesmo que sua escrita seja diferente.}
    
    \text{\textbf{Como funciona}:}
    
    \text{1. \textbf{Codificação Fonética}: Cada palavra é convertida em um código fonético usando a técnica \textbf{Soundex}, que atribui um código baseado no som da palavra.}
    
    \text{2. \textbf{Cálculo da Distância de Levenshtein}: A \textbf{distância de Levenshtein} mede quantas mudanças (inserções, deleções ou substituições) são necessárias para transformar uma palavra em outra.}
    
    \( D(S_1, S_2) = \text{Número mínimo de operações para transformar } S_1 \text{ em } S_2 \)
    
    \text{3. \textbf{Cálculo da Similaridade}: A distância entre os sons das frases é normalizada para fornecer uma pontuação de similaridade de 0 a 1:}
    
    \( \text{Similaridade} = 1 - \frac{D(S_1, S_2)}{\max(\text{len}(S_1), \text{len}(S_2))} \)
    
    \text{Onde:}
    
    \text{- \( D(S_1, S_2) \) é a distância de Levenshtein entre as sequências fonéticas.}
    
    \text{- \( \text{len}(S_i) \) é o comprimento da sequência fonética \( S_i \).}
    
    \text{\textbf{Objetivo}: Avaliar o quanto as palavras de dois idiomas diferentes soam de forma semelhante, independentemente de sua ortografia.}
    
    \text{\textbf{Exemplo}: Se as palavras de Dzubukuá e Português Arcaico tiverem uma alta similaridade fonológica, isso pode sugerir que o som dessas línguas permaneceu semelhante ao longo do tempo, mesmo com mudanças na grafia.}
    
    \text{---}
    
    \text{### 4. Análises Estatísticas e Visualizações}
    
    \text{Após calcular as similaridades, podemos realizar análises estatísticas para explorar a relação entre as diferentes medidas de similaridade.}
    
    \text{#### 4.1 Cálculo de Correlações}
    
    \text{As \textbf{correlações} medem como duas variáveis se movem juntas. Usamos três tipos de correlação: \textbf{Pearson}, \textbf{Spearman} e \textbf{Kendall}.}
    
    \text{\textbf{Fórmulas}:}
    
    \text{1. \textbf{Correlação de Pearson}:}
    
    \( r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n}(Y_i - \bar{Y})^2}} \)
    
    \text{2. \textbf{Correlação de Spearman}:}
    
    \( \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)} \)
    
    \text{Onde:}
    
    \text{- \( d_i \) é a diferença entre os postos (ranks) de \( X_i \) e \( Y_i \).}
    
    \text{3. \textbf{Correlação de Kendall}:}
    
    \( \tau = \frac{C - D}{\frac{1}{2} n(n - 1)} \)
    
    \text{Onde:}
    
    \text{- \( C \) é o número de pares concordantes.}
    
    \text{- \( D \) é o número de pares discordantes.}
    
    \text{\textbf{Objetivo}: Entender se, por exemplo, a similaridade lexical e semântica entre as frases estão relacionadas. Se duas variáveis têm uma correlação forte, significa que há uma relação entre elas.}
    
    \text{---}
    
    \text{#### 4.2 Regressão Linear}
    
    \text{A \textbf{regressão linear} é usada para determinar se há uma relação direta entre duas variáveis, como entre a similaridade semântica de Dzubukuá e Português Arcaico e a similaridade com o Português Moderno.}
    
    \text{\textbf{Modelo}:}
    
    \( y = \beta_0 + \beta_1 x + \epsilon \)
    
    \text{Onde:}
    
    \text{- \( y \) é a variável dependente.}
    
    \text{- \( x \) é a variável independente.}
    
    \text{- \( \beta_0 \) é o intercepto.}
    
    \text{- \( \beta_1 \) é o coeficiente angular.}
    
    \text{- \( \epsilon \) é o termo de erro.}
    
    \text{\textbf{Objetivo}: Avaliar se uma medida de similaridade pode ser usada para prever outra, indicando uma possível relação causal ou dependente.}
    
    \text{---}
    
    \text{#### 4.3 Regressão Múltipla}
    
    \text{A \textbf{regressão múltipla} estende a regressão linear para considerar várias variáveis de uma vez.}
    
    \text{\textbf{Modelo}:}
    
    \( y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon \)
    
    \text{Onde:}
    
    \text{- \( y \) é a variável dependente.}
    
    \text{- \( x_1, x_2, \dots, x_p \) são as variáveis independentes.}
    
    \text{- \( \beta_0 \) é o intercepto.}
    
    \text{- \( \beta_1, \beta_2, \dots, \beta_p \) são os coeficientes das variáveis independentes.}
    
    \text{\textbf{Objetivo}: Entender quais medidas de similaridade (semântica, lexical ou fonológica) mais influenciam a relação entre os idiomas.}
    
    \text{---}
    
    \text{#### 4.4 Análise de Variância (ANOVA)}
    
    \text{A \textbf{ANOVA} compara as médias de grupos diferentes para ver se há uma diferença significativa.}
    
    \text{\textbf{Estatística F}:}
    
    \( F = \frac{\text{Variância entre os grupos}}{\text{Variância dentro dos grupos}} \)
    
    \text{Onde:}
    
    \text{- A variância entre os grupos mede a diferença das médias de cada grupo em relação à média geral.}
    
    \text{- A variância dentro dos grupos mede a dispersão dos dados dentro de cada grupo.}
    
    \text{\textbf{Objetivo}: Entender se, por exemplo, as similaridades entre Dzubukuá e Português Arcaico são significativamente diferentes daquelas entre Dzubukuá e Português Moderno.}
    
    \text{---}
    
    \text{### 5. Considerações e Limitações}
    
    \text{As análises realizadas fornecem uma visão detalhada das semelhanças entre Dzubukuá, Português Arcaico e Português Moderno. Entretanto, é importante considerar que:}
    
    \text{- \textbf{A qualidade dos dados} é crucial: Análises imprecisas podem ocorrer se os dados não forem representativos.}
    
    \text{- \textbf{Assunções estatísticas} devem ser verificadas: Algumas análises requerem normalidade e independência dos dados.}
    
    \text{- \textbf{Multicolinearidade} pode influenciar as regressões: Se as variáveis de similaridade estiverem muito correlacionadas, pode ser difícil determinar a importância individual de cada uma.}
    
    \text{---}
    
    \text{### Conclusão}
    
    \text{Este estudo sobre a comparação de três idiomas oferece insights valiosos sobre a evolução linguística. Ele mostra como as línguas podem se manter semelhantes ou divergir ao longo do tempo, não apenas em termos de palavras, mas também no significado e som.}
    
    \text{Resultados como esses nos ajudam a entender a história linguística e cultural dos povos que falavam essas línguas, oferecendo uma visão mais profunda sobre a preservação de conceitos e ideias ao longo do tempo.}
    ''')


# Imagem e Contatos
if os.path.exists("eu.ico"):
    st.sidebar.image("eu.ico", width=80)
else:
    st.sidebar.text("Imagem do contato não encontrada.")

st.sidebar.write("""
Projeto Geomaker + IA 
- Professor: Marcelo Claro.
Contatos: marceloclaro@gmail.com
Whatsapp: (88)981587145
Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
""")

# _____________________________________________
# Controle de Áudio
st.sidebar.title("Controle de Áudio")

# Dicionário de arquivos de áudio, com nomes amigáveis mapeando para o caminho do arquivo
mp3_files = {
    "Áudio explicativo 1": "kariri.mp3",
}

# Lista de arquivos MP3 para seleção
selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))

# Controle de opção de repetição
loop = st.sidebar.checkbox("Repetir áudio")

# Botão de Play para iniciar o áudio
play_button = st.sidebar.button("Play")

# Placeholder para o player de áudio
audio_placeholder = st.sidebar.empty()

# Função para verificar se o arquivo existe
def check_file_exists(mp3_path):
    if not os.path.exists(mp3_path):
        st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
        return False
    return True

# Se o botão Play for pressionado e um arquivo de áudio estiver selecionado
if play_button and selected_mp3:
    mp3_path = mp3_files[selected_mp3]
    
    # Verificação da existência do arquivo
    if check_file_exists(mp3_path):
        try:
            # Abrindo o arquivo de áudio no modo binário
            with open(mp3_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                
                # Codificando o arquivo em base64 para embutir no HTML
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Controle de loop (repetição)
                loop_attr = "loop" if loop else ""
                
                # Gerando o player de áudio em HTML
                audio_html = f"""
                <audio id="audio-player" controls autoplay {loop_attr}>
                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                  Seu navegador não suporta o elemento de áudio.
                </audio>
                """
                
                # Inserindo o player de áudio na interface
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        
        except FileNotFoundError:
            st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")
#______________________________________________________________________________________-



# Certifique-se de que todas as funções estão definidas antes do main()
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
        intersection = np.sum(np.minimum(a, b))
        total = np.sum(a) + np.sum(b)
        return 2 * intersection / total if total > 0 else 0

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
            # Codificação fonética usando Soundex
            s1_phonetic = ''.join(jellyfish.soundex(s1))
            s2_phonetic = ''.join(jellyfish.soundex(s2))
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

# Função para calcular correlações de Pearson, Spearman e Kendall
def calcular_correlacoes_avancadas(similarity_df):
    """Calcula as correlações de Pearson, Spearman e Kendall entre as variáveis de similaridade."""
    pearson_corr = similarity_df.corr(method='pearson')
    spearman_corr = similarity_df.corr(method='spearman')
    kendall_corr = similarity_df.corr(method='kendall')
    return pearson_corr, spearman_corr, kendall_corr

# Função para realizar regressão linear com teste de significância e diagnósticos
def regressao_linear(similarity_df):
    """Aplica regressão linear entre duas variáveis de similaridade e realiza testes de significância estatística."""
    X = similarity_df['Dzubukuá - Arcaico (Semântica)']
    y = similarity_df['Dzubukuá - Moderno (Semântica)']
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    y_pred = model.predict(X_const)

    # Obter intervalos de confiança
    intervalo_confianca = model.conf_int(alpha=0.05)

    # Diagnósticos de resíduos
    residuos = model.resid
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuos, kde=True, ax=axs[0])
    axs[0].set_title('Distribuição dos Resíduos')
    stats.probplot(residuos, dist="norm", plot=axs[1])
    axs[1].set_title('QQ-Plot dos Resíduos')
    st.pyplot(fig)

    return model, y_pred

# Função para realizar regressão múltipla com diagnósticos
def regressao_multipla(similarity_df):
    """Aplica regressão múltipla entre as variáveis de similaridade e realiza testes de significância estatística."""
    # Definir variáveis independentes (X) e variável dependente (y)
    X = similarity_df.drop(columns=['Dzubukuá - Moderno (Semântica)', 'Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore')
    y = similarity_df['Dzubukuá - Moderno (Semântica)']

    # Verificar multicolinearidade usando VIF
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1) for i in range(len(X.columns))]
    st.write("Fatores de Inflação da Variância (VIF):")
    st.dataframe(vif_data)

    # Ajustar o modelo
    model = sm.OLS(y, X_const).fit()

    # Diagnósticos de resíduos
    residuos = model.resid
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuos, kde=True, ax=axs[0])
    axs[0].set_title('Distribuição dos Resíduos da Regressão Múltipla')
    stats.probplot(residuos, dist="norm", plot=axs[1])
    axs[1].set_title('QQ-Plot dos Resíduos da Regressão Múltipla')
    st.pyplot(fig)

    return model

# Outras funções necessárias (defina todas antes do main)
def aplicar_pca(similarity_df):
    """Reduz a dimensionalidade usando PCA para entender os padrões nas similaridades."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(similarity_df)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

def grafico_pca(similarity_df, pca_result, explained_variance):
    """Plota os resultados da Análise de Componentes Principais (PCA)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=100)
    ax.set_title('Análise de Componentes Principais (PCA)', fontsize=16, pad=20)
    ax.set_xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.set_ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.2f}% da variância)', fontsize=14, labelpad=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

def analise_clustering(similarity_df):
    """Realiza análise de clustering utilizando K-Means e DBSCAN."""
    # Padronizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(similarity_df)

    # K-Means clustering
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(data_scaled)
        distortions.append(kmeanModel.inertia_)

    # Plotar o gráfico do método Elbow
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('Número de Clusters')
    ax.set_ylabel('Distortion')
    ax.set_title('Método Elbow para Determinação do Número Ótimo de Clusters')
    st.pyplot(fig)

    # Escolher k com base no método Elbow ou outras considerações
    k = 3  # Ajuste conforme necessário
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_scaled)
    similarity_df['Cluster_KMeans'] = kmeans_labels

    # Avaliar o modelo K-Means usando o coeficiente de silhueta
    silhouette_avg = silhouette_score(data_scaled, kmeans_labels)
    st.write(f"Coeficiente de Silhueta para K-Means: {silhouette_avg:.4f}")

    # DBSCAN clustering
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data_scaled)
    similarity_df['Cluster_DBSCAN'] = dbscan_labels

    # Avaliar o modelo DBSCAN (ignorar ruído, label = -1)
    labels_unique = np.unique(dbscan_labels)
    labels_unique = labels_unique[labels_unique != -1]
    if len(labels_unique) > 1:
        silhouette_avg_dbscan = silhouette_score(data_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        st.write(f"Coeficiente de Silhueta para DBSCAN: {silhouette_avg_dbscan:.4f}")
    else:
        st.write("DBSCAN não encontrou clusters significativos.")

    return similarity_df

def visualizar_clusters(similarity_df):
    """Visualiza os clusters obtidos pelo K-Means e DBSCAN."""
    # PCA para reduzir a dimensionalidade
    features = similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore')
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(features)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para K-Means
    scatter = axs[0].scatter(data_pca[:, 0], data_pca[:, 1], c=similarity_df['Cluster_KMeans'], cmap='Set1', s=50)
    axs[0].set_title('Clusters K-Means', fontsize=16)
    axs[0].set_xlabel('Componente Principal 1', fontsize=14)
    axs[0].set_ylabel('Componente Principal 2', fontsize=14)
    axs[0].grid(True, linestyle='--', linewidth=0.5)

    # Gráfico para DBSCAN
    scatter = axs[1].scatter(data_pca[:, 0], data_pca[:, 1], c=similarity_df['Cluster_DBSCAN'], cmap='Set1', s=50)
    axs[1].set_title('Clusters DBSCAN', fontsize=16)
    axs[1].set_xlabel('Componente Principal 1', fontsize=14)
    axs[1].set_ylabel('Componente Principal 2', fontsize=14)
    axs[1].grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

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

def grafico_regressao_plotly(similarity_df, model, y_pred):
    """Gera gráfico interativo com a linha de regressão e informações estatísticas."""
    X = similarity_df['Dzubukuá - Arcaico (Semântica)']
    y = similarity_df['Dzubukuá - Moderno (Semântica)']
    intervalo_confianca = model.conf_int(alpha=0.05)
    r2 = model.rsquared
    p_value = model.pvalues[1]  # p-valor do coeficiente da variável independente

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Dados'))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Regressão Linear'))
    fig.update_layout(
        title=f"Regressão Linear - R²: {r2:.2f}, p-valor: {p_value:.4f}<br>Intervalo de Confiança do Coeficiente: [{intervalo_confianca.iloc[1,0]:.4f}, {intervalo_confianca.iloc[1,1]:.4f}]",
        xaxis_title="Similaridade Dzubukuá - Arcaico (Semântica)",
        yaxis_title="Similaridade Dzubukuá - Moderno (Semântica)",
        xaxis=dict(title_font=dict(size=14), tickangle=-45),
        yaxis=dict(title_font=dict(size=14)),
        width=800,
        height=600,
        margin=dict(l=100, r=100, b=100, t=100),
        font=dict(size=12),
    )
    st.plotly_chart(fig)

def grafico_dendrograma(similarity_df):
    """Gera um dendrograma para visualizar relações hierárquicas entre as variáveis."""
    from scipy.cluster.hierarchy import linkage

    linked = linkage(similarity_df.T, 'ward', metric='euclidean')
    labelList = similarity_df.columns
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linked, labels=labelList, ax=ax, orientation='top')
    ax.set_title('Dendrograma das Similaridades', fontsize=16, pad=20)
    ax.set_xlabel('Variáveis', fontsize=14, labelpad=15)
    ax.set_ylabel('Distância Euclidiana', fontsize=14, labelpad=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

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

def testes_hipotese(similarity_df):
    """Realiza testes de hipóteses estatísticas nas similaridades."""
    # Teste t para duas amostras independentes
    t_stat, p_value = stats.ttest_ind(
        similarity_df['Dzubukuá - Arcaico (Semântica)'],
        similarity_df['Dzubukuá - Moderno (Semântica)']
    )
    st.write(f"Teste t para duas amostras independentes (Semântica):")
    st.write(f"Estatística t: {t_stat:.4f}")
    st.write(f"Valor-p: {p_value:.4e}")
    if p_value < 0.05:
        st.write("Conclusão: Há uma diferença significativa entre as médias das similaridades semânticas.")
    else:
        st.write("Conclusão: Não há evidência suficiente para afirmar que as médias diferem.")

def analise_anova(similarity_df):
    """Realiza ANOVA para comparar as médias das similaridades entre grupos."""
    # Exemplo usando as similaridades semânticas
    f_stat, p_value = f_oneway(
        similarity_df['Dzubukuá - Arcaico (Semântica)'],
        similarity_df['Dzubukuá - Moderno (Semântica)'],
        similarity_df['Arcaico - Moderno (Semântica)']
    )
    st.write(f"Resultado da ANOVA:")
    st.write(f"Estatística F: {f_stat:.4f}")
    st.write(f"Valor-p: {p_value:.4e}")
    if p_value < 0.05:
        st.write("Conclusão: As médias das similaridades semânticas diferem significativamente entre os grupos.")
    else:
        st.write("Conclusão: Não há evidência suficiente para afirmar que as médias diferem entre os grupos.")

def ajuste_q_exponencial(data):
    """Ajusta uma distribuição q-exponencial aos dados e retorna os parâmetros."""
    def q_exponential(x, a, b, q):
        return a * (1 - (1 - q) * b * x) ** (1 / (1 - q))

    hist, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Ajuste dos parâmetros
    params, _ = curve_fit(q_exponential, bin_centers, hist, maxfev=10000)
    a, b, q = params
    return a, b, q

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
        sentences_dzubukua = df[df['Idioma'] == 'Dzubukuá']['Texto Original'].tolist()
        sentences_arcaico = df[df['Idioma'] == 'Português Arcaico']['Texto Original'].tolist()
        sentences_moderno = df['Tradução para o Português Moderno'].tolist()

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

        # Gráfico interativo de correlações usando Plotly
        st.subheader("Mapa de Correlação entre Similaridades")
        grafico_interativo_plotly(similarity_df)

        # Regressão Linear entre Dzubukuá e Moderno
        st.subheader("Análise de Regressão Linear entre Dzubukuá e Português Moderno (Semântica)")
        model_linear, y_pred_linear = regressao_linear(similarity_df)
        st.write(model_linear.summary())
        grafico_regressao_plotly(similarity_df, model_linear, y_pred_linear)

        # Regressão Múltipla
        st.subheader("Análise de Regressão Múltipla")
        model_multipla = regressao_multipla(similarity_df)
        st.write(model_multipla.summary())

        # Testes de Hipóteses
        st.subheader("Testes de Hipóteses Estatísticas")
        testes_hipotese(similarity_df)

        # ANOVA
        st.subheader("Análise de Variância (ANOVA)")
        analise_anova(similarity_df)

        # Ajuste q-Exponencial
        st.subheader("Ajuste q-Exponencial")
        # Usando a similaridade semântica como exemplo
        a, b, q = ajuste_q_exponencial(similarity_df['Dzubukuá - Moderno (Semântica)'])
        st.write(f"Parâmetros ajustados:")
        st.write(f"a = {a:.4f}")
        st.write(f"b = {b:.4f}")
        st.write(f"q = {q:.4f}")
        st.write("O parâmetro q indica o grau de não-extensividade da distribuição, relevante em sistemas complexos.")

        # Análise de Componentes Principais (PCA)
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_result, explained_variance = aplicar_pca(similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN'], errors='ignore'))
        grafico_pca(similarity_df, pca_result, explained_variance)

        # Análise de Agrupamentos (Clustering)
        st.subheader("Análise de Agrupamentos (Clustering)")
        similarity_df = analise_clustering(similarity_df)
        visualizar_clusters(similarity_df)

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
        grafico_dendrograma(similarity_df.drop(columns=['Cluster_KMeans', 'Cluster_DBSCAN']))

        # Perguntar se o usuário deseja baixar os resultados como CSV
        if st.checkbox("Deseja baixar os resultados como CSV?"):
            salvar_dataframe(similarity_df)

if __name__ == '__main__':
    main()
