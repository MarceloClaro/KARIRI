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

# Função principal para rodar a aplicação no Streamlit
import os
import logging


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
st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')

st.sidebar.image("logo.png", width=200)
# Exibe uma imagem na sidebar. O arquivo "logo.png" é a imagem que representa a marca ou o tema da aplicação, neste caso, provavelmente o logotipo do Geomaker.
# - `width=200`: Define a largura da imagem, ajustando-a para caber na barra lateral sem ocupar muito espaço.
# Vantagem: A imagem dá uma identidade visual para a aplicação, tornando-a mais profissional e personalizada.
# Desvantagem: Se a imagem não estiver disponível ou o caminho estiver incorreto, pode gerar um erro ou deixar um espaço vazio na interface.
# Possível solução: Implementar uma verificação para garantir que a imagem existe antes de carregá-la. Caso não exista, exibir uma imagem padrão ou mensagem alternativa. 
# Exemplo de código para verificar a existência da imagem:
# ```python
# import os
# if os.path.exists("logo.png"):
#     st.sidebar.image("logo.png", width=200)
# else:
#     st.sidebar.text("Imagem não encontrada.")
# ```

st.sidebar.title("Geomaker +IA")
# Exibe um título na barra lateral com o nome "Geomaker +IA".
# Vantagem: O título fornece uma identificação clara do propósito ou do nome da aplicação. No contexto, o título "Geomaker +IA" indica uma ferramenta de inteligência artificial (IA) associada ao Geomaker.
# Desvantagem: O título é estático e não se adapta dinamicamente ao conteúdo. Se o nome do projeto ou da aplicação mudar, o código precisará ser ajustado manualmente.
# Possível solução: Tornar o título dinâmico, utilizando uma variável que permita alterar o nome facilmente, como em um painel de administração ou via um arquivo de configuração.
# Exemplo:
# ```python
# app_title = "Geomaker +IA"
# st.sidebar.title(app_title)
# ```

# Vantagens gerais:
# 1. **Identidade visual**: O uso da imagem e do título ajuda a criar uma identidade visual clara para a aplicação, tornando-a facilmente reconhecível pelos usuários.
# 2. **Organização de informações**: A sidebar permite que informações importantes, como logotipos e títulos, fiquem visíveis independentemente de onde o usuário esteja navegando na interface principal.
# 3. **Otimização de espaço**: Colocar o logotipo e o título na barra lateral ajuda a liberar espaço na área principal da aplicação, mantendo o foco nas funcionalidades.

# Desvantagens gerais:
# 1. **Dependência de arquivos externos**: O uso de uma imagem externa como o logotipo cria uma dependência que pode causar problemas se o arquivo não estiver disponível.
# 2. **Conteúdo estático**: Tanto o logotipo quanto o título são fixos e não se adaptam a mudanças contextuais na aplicação. Se houver uma alteração no nome da marca ou do logotipo, o código precisará ser alterado manualmente.
# 3. **Layout não responsivo**: A largura fixa da imagem pode ser inadequada para telas menores, o que pode prejudicar a experiência de usuários em dispositivos móveis ou telas pequenas.

# Possíveis soluções:
# 1. **Verificação de disponibilidade da imagem**: Implementar um código que verifica se o arquivo da imagem está disponível e, caso não esteja, exibir uma mensagem alternativa ou uma imagem padrão.
# 2. **Título dinâmico**: Tornar o título da aplicação dinâmico, utilizando variáveis configuráveis que podem ser alteradas facilmente, sem necessidade de mudar o código diretamente.
# 3. **Responsividade**: Melhorar a responsividade da imagem e do layout da sidebar, ajustando automaticamente a largura da imagem com base no tamanho da tela.

# Inovações:
# 1. **Barra lateral interativa**: Tornar a barra lateral mais interativa, permitindo que o usuário escolha temas ou personalize a interface, como a escolha de um logotipo diferente ou o título da aplicação.
# 2. **Suporte a modo escuro**: Adicionar uma opção para alternar entre modos claro e escuro diretamente da barra lateral, tornando a interface mais agradável para diferentes preferências de usuários.
# 3. **Notificações na sidebar**: Incorporar uma área para exibição de notificações ou atualizações na barra lateral, informando o usuário sobre novas funcionalidades ou mensagens importantes.

# _____________________________________________

with st.sidebar.expander("Insights do Projeto"):
    # A função `st.sidebar.expander` cria um elemento de "expansão" na barra lateral, que pode ser aberto ou fechado pelo usuário.
    # - "Insights do Código": É o título do expander, que indica que ele contém informações detalhadas sobre o código.
    # Vantagem: O uso de um expander permite organizar informações que não precisam ser exibidas o tempo todo, economizando espaço visual.
    # Desvantagem: Informações importantes podem ficar escondidas se o usuário não expandir o conteúdo, dificultando o acesso rápido a dados essenciais.
    # Possível solução: Para informações críticas, fornecer uma indicação visual clara para o usuário sobre a importância de expandir o conteúdo.
    # Exemplo: Adicionar uma seta ou destaque que incentive o usuário a expandir.
    
    st.markdown("""
        ## Sobre o Projeto
        Este aplicativo realiza análises avançadas de similaridade linguística entre três idiomas: **Dzubukuá**, **Português Arcaico** e **Português Moderno**. Utilizamos técnicas de processamento de linguagem natural (PLN) e estatística para explorar as relações entre essas línguas.

        ## Objetivos das Análises
        - **Similaridade Semântica**: Avaliar o quão semelhantes são as sentenças em termos de significado.
        - **Similaridade Lexical**: Comparar as palavras e estruturas de caracteres entre as línguas.
        - **Similaridade Fonológica**: Analisar a semelhança na pronúncia e sons das palavras.

        ## Possíveis Interpretações dos Resultados
        ### Similaridade Semântica
        Utilizando modelos como o **Sentence-BERT**, medimos a proximidade de significado entre sentenças correspondentes. Por exemplo, se uma frase em Dzubukuá tem alta similaridade semântica com sua tradução em Português Moderno, isso indica que, apesar das diferenças linguísticas, o conceito transmitido é semelhante.

        *Exemplo*: Se a frase Dzubukuá "Umake zuka" tem alta similaridade com "O sol nasce", podemos inferir que a tradução captura bem o significado original.

        ### Similaridade Lexical
        A análise lexical com **N-gramas** e **Word2Vec** nos permite entender como as palavras e suas estruturas se relacionam entre as línguas.

        - **N-gramas**: Se o coeficiente de similaridade for alto entre Português Arcaico e Moderno, pode indicar que a ortografia e construção de palavras permaneceram relativamente constantes ao longo do tempo.
        - **Word2Vec**: Captura contextos semânticos das palavras. Similaridades altas podem sugerir empréstimos linguísticos ou influências culturais.

        ### Similaridade Fonológica
        Avaliamos como os sons das palavras se comparam entre as línguas usando codificação fonética e distâncias de edição.

        *Exemplo*: Se "coração" em Português Moderno e "coraçon" em Português Arcaico têm alta similaridade fonológica, isso reflete a evolução da pronúncia e escrita ao longo do tempo.

        ### Análises Estatísticas
        - **Correlações**: Identificam relações entre diferentes medidas de similaridade. Correlações fortes podem indicar que mudanças em uma dimensão (por exemplo, semântica) estão associadas a mudanças em outra (por exemplo, lexical).
        - **Regressões**: Modelam relações entre variáveis. Uma regressão linear significativa entre similaridades semânticas de Dzubukuá-Português Arcaico e Dzubukuá-Português Moderno pode sugerir que as traduções modernas preservam elementos semânticos do arcaico.
        - **Testes de Hipóteses e ANOVA**: Verificam se as diferenças observadas são estatisticamente significativas. Isso ajuda a validar se as similaridades ou diferenças não ocorrem ao acaso.

        ### Análise de Componentes Principais (PCA)
        Reduz a dimensionalidade dos dados para identificar padrões. Componentes principais que explicam grande parte da variância podem revelar fatores subjacentes importantes nas similaridades linguísticas.

        ### Clustering (Agrupamento)
        Agrupa dados com base em características semelhantes.

        - **K-Means**: Separa os dados em k clusters distintos. Por exemplo, frases que formam um cluster podem compartilhar características linguísticas específicas.
        - **DBSCAN**: Identifica clusters de alta densidade e é útil para detectar outliers.

        ### Ajuste q-Exponencial
        Modela distribuições de dados que não seguem uma distribuição normal. O parâmetro *q* indica o grau de não-extensividade, relevante em sistemas complexos como a evolução de línguas.

        ## Considerações para Leigos
        - **Semelhanças e Diferenças Linguísticas**: As análises ajudam a entender como línguas evoluem e influenciam umas às outras.
        - **Importância Cultural**: Estudar o Dzubukuá pode revelar aspectos culturais e históricos importantes, especialmente ao compará-lo com o Português Arcaico e Moderno.
        - **Evolução da Linguagem**: Observando as similaridades, podemos inferir como certas palavras e estruturas mudaram ou permaneceram ao longo do tempo.

        ## Exemplos Práticos
        - **Tradução e Preservação**: Se uma palavra em Dzubukuá não tem equivalente direto em Português Moderno, mas encontra correspondência no Português Arcaico, isso pode indicar perda ou mudança de conceitos culturais.
        - **Educação e Pesquisa**: As ferramentas e análises apresentadas podem ser utilizadas por estudantes e pesquisadores para aprofundar o conhecimento em linguística histórica e comparativa.

        ## Conclusão
        Este aplicativo oferece uma forma interativa de explorar e compreender as complexas relações entre línguas, combinando técnicas modernas de análise de dados com estudos linguísticos tradicionais.

        **Nota**: Os resultados das análises devem ser interpretados com cautela e, preferencialmente, com apoio de especialistas em linguística para insights mais profundos.
        """)
    # A função `st.markdown` permite exibir um texto com formatação Markdown dentro do expander. Neste caso, ela está sendo usada para fornecer uma introdução ao código e sua funcionalidade.
    # Vantagem: O Markdown permite uma apresentação estruturada e organizada do conteúdo, incluindo listas, negrito, e outros elementos de formatação.
    # Desvantagem: O Markdown não é tão flexível quanto HTML para personalizações mais avançadas de estilo e pode limitar a formatação visual.
    # Possível solução: Se mais personalização for necessária, usar `st.markdown` com o parâmetro `unsafe_allow_html=True` para incorporar elementos HTML mais avançados.
    # Exemplo: `st.markdown("<h1>Agentes Alan Kay</h1>", unsafe_allow_html=True)`

   
# _____________________________________________
with st.sidebar.expander("Insights do Código"):
    # O comando `st.sidebar.expander` cria uma seção expansível na barra lateral com o título "Insights do Código".
    # Vantagem: Permite ao usuário ocultar ou expandir informações conforme necessário, economizando espaço e evitando sobrecarregar a interface.
    # Desvantagem: Se o conteúdo dentro do `expander` for essencial, o usuário pode deixar de visualizá-lo se não souber que há informações ali.
    # Possível solução: Tornar o título mais chamativo, sugerindo que contém informações valiosas, como "Clique para ver os insights importantes do código".

    st.markdown("""
    **Introdução**

Este documento fornece uma explicação detalhada das análises linguísticas realizadas por meio de um código Python que compara três idiomas: **Dzubukuá** (uma língua morta ou em risco de extinção), **Português Arcaico** e **Português Moderno**. O objetivo é entender as similaridades e diferenças entre esses idiomas em termos semânticos, lexicais e fonológicos, utilizando técnicas avançadas de processamento de linguagem natural (PLN) e estatística.

Serão apresentados os métodos utilizados, incluindo fórmulas matemáticas, justificativas para cada análise, objetivos específicos, exemplos e possíveis insights. Ao final, serão discutidas as limitações e considerações técnicas de cada abordagem.

---

**1. Importação das Bibliotecas Necessárias**

O código começa importando várias bibliotecas que fornecem as ferramentas necessárias para realizar as análises:

- **Pandas**: Manipulação de dados em estruturas de dados como DataFrames.
- **Streamlit**: Criação de interfaces web interativas.
- **Matplotlib e Seaborn**: Visualização gráfica dos dados.
- **Plotly**: Gráficos interativos.
- **Scikit-learn**: Algoritmos de aprendizado de máquina e estatística.
- **Gensim**: Implementação de modelos Word2Vec.
- **Jellyfish**: Funções para cálculos fonéticos e distâncias de edição.
- **Statsmodels e Scipy**: Ferramentas estatísticas avançadas.

**Objetivo:** Fornecer as ferramentas necessárias para realizar análises de similaridade linguística, cálculos estatísticos, visualizações e modelos de aprendizado de máquina.

---

**2. Carregamento e Preparação dos Dados**

O código permite que o usuário carregue um arquivo CSV contendo as frases nos três idiomas de interesse. As colunas esperadas são:

- **Idioma**
- **Texto Original**
- **Tradução para o Português Moderno**

**Objetivo:** Extrair as frases correspondentes de cada idioma para serem usadas nas análises subsequentes.

---

**3. Cálculo das Similaridades**

As similaridades entre as frases dos três idiomas são calculadas em termos semânticos, lexicais e fonológicos.

### 3.1 Similaridade Semântica com Sentence-BERT

**Função:** `calcular_similaridade_semantica`

**Metodologia:**

- **Modelo Sentence-BERT:** Um modelo de linguagem que gera embeddings (vetores) que capturam o significado semântico de sentenças.

**Passos:**

1. **Geração dos Embeddings:**

   - Para cada frase, o modelo gera um vetor de dimensão fixa que representa o significado semântico da frase.

   - Matemática: Seja \( S = \{s_1, s_2, \dots, s_n\} \) o conjunto de sentenças. O modelo transforma cada sentença \( s_i \) em um vetor \( \vec{v}_i \in \mathbb{R}^d \), onde \( d \) é a dimensionalidade do embedding.

2. **Cálculo da Similaridade de Cosseno:**

   - A similaridade entre duas sentenças é calculada usando a **similaridade de cosseno** entre seus embeddings.

   - Fórmula da Similaridade de Cosseno entre vetores \( \vec{v}_i \) e \( \vec{v}_j \):

     \[
     \text{similaridade}(\vec{v}_i, \vec{v}_j) = \cos(\theta) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \, \|\vec{v}_j\|}
     \]

   - Onde:

     - \( \vec{v}_i \cdot \vec{v}_j \) é o produto escalar dos vetores.
     - \( \|\vec{v}_i\| \) é a norma (magnitude) do vetor \( \vec{v}_i \).

**Objetivo:** Medir o quanto as frases correspondentes nos diferentes idiomas são semelhantes em termos de significado.

**Justificativa:** A similaridade semântica é crucial para entender se as frases, apesar de diferenças lexicais ou fonéticas, carregam o mesmo significado. Isso é especialmente importante ao comparar uma língua morta com línguas modernas para entender a transmissão de conceitos e ideias.

**Exemplo e Possíveis Insights:**

- Uma alta similaridade semântica entre Dzubukuá e Português Arcaico pode indicar que certos conceitos foram preservados ao longo do tempo.
- Diferenças semânticas podem revelar mudanças culturais ou a perda de certos conceitos.

---

### 3.2 Similaridade Lexical com N-gramas

**Função:** `calcular_similaridade_ngramas`

**Metodologia:**

- **N-gramas de caracteres:** Sequências de N caracteres extraídas das frases.

**Passos:**

1. **Extração de N-gramas:**

   - Para cada frase, extrai-se um conjunto de N-gramas de caracteres.

   - Exemplo para N=2 (bigramas) da palavra "casa": \{"ca", "as", "sa"\}.

2. **Representação Vetorial:**

   - Cada frase é representada como um vetor binário indicando a presença ou ausência de cada N-grama possível.

3. **Cálculo do Coeficiente de Sorensen-Dice:**

   - Métrica de similaridade entre dois conjuntos baseada na sobreposição.

   - Fórmula:

     \[
     \text{SDC}(A, B) = \frac{2 |A \cap B|}{|A| + |B|}
     \]

   - Onde:

     - \( A \) e \( B \) são os conjuntos de N-gramas das duas frases.
     - \( |A| \) é o número de N-gramas em \( A \).
     - \( |A \cap B| \) é o número de N-gramas comuns a \( A \) e \( B \).

**Objetivo:** Avaliar a similaridade das frases em termos de estrutura lexical, ou seja, a construção das palavras.

**Justificativa:** As línguas podem compartilhar palavras semelhantes ou ter evoluído de formas que preservam padrões lexicais. A análise de N-gramas captura essas semelhanças ou diferenças.

**Exemplo e Possíveis Insights:**

- Uma alta similaridade lexical entre Português Arcaico e Moderno é esperada devido à evolução da língua.
- Baixa similaridade entre Dzubukuá e Português pode indicar origens linguísticas diferentes.

---

### 3.3 Similaridade Lexical com Word2Vec

**Função:** `calcular_similaridade_word2vec`

**Metodologia:**

- **Modelo Word2Vec:** Gera embeddings para palavras com base no contexto em que aparecem.

**Passos:**

1. **Tokenização das Frases:**

   - As frases são divididas em palavras.

2. **Treinamento do Modelo Word2Vec:**

   - O modelo aprende vetores de palavras \( \vec{w} \) de forma que palavras com contextos semelhantes tenham vetores próximos.

3. **Representação das Frases:**

   - Cada frase é representada pela média dos vetores das palavras que a compõem.

   - Fórmula:

     \[
     \vec{v}_{\text{frase}} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i
     \]

     Onde \( n \) é o número de palavras na frase.

4. **Cálculo da Similaridade de Cosseno:**

   - Similar ao método semântico, calcula-se a similaridade entre os vetores das frases.

**Objetivo:** Capturar similaridades lexicais considerando o contexto das palavras.

**Justificativa:** Palavras que aparecem em contextos semelhantes podem indicar semelhanças linguísticas que não são evidentes apenas pela ortografia.

**Exemplo e Possíveis Insights:**

- Pode revelar empréstimos linguísticos ou raízes comuns.
- Diferenças podem indicar evolução semântica das palavras.

---

### 3.4 Similaridade Fonológica

**Função:** `calcular_similaridade_fonologica`

**Metodologia:**

- **Codificação Fonética (Soundex):** Converte palavras em códigos que representam sons.
- **Distância de Levenshtein:** Mede o número mínimo de operações necessárias para transformar uma string em outra.

**Passos:**

1. **Codificação Fonética das Frases:**

   - Cada frase é convertida em uma sequência de códigos fonéticos.

2. **Cálculo da Distância de Levenshtein:**

   - Calcula-se a distância entre as sequências codificadas de duas frases.

   - A distância \( D \) entre duas strings \( S_1 \) e \( S_2 \) é dada por:

     \[
     D(S_1, S_2) = \text{Número mínimo de inserções, deleções ou substituições para transformar } S_1 \text{ em } S_2
     \]

3. **Normalização da Similaridade:**

   - A similaridade é normalizada entre 0 e 1:

     \[
     \text{Similaridade} = 1 - \frac{D(S_1, S_2)}{\max(\text{len}(S_1), \text{len}(S_2))}
     \]

**Objetivo:** Avaliar o quanto as frases soam semelhantes, independentemente da escrita.

**Justificativa:** Mesmo que a ortografia seja diferente, as línguas podem compartilhar sons semelhantes, indicando possíveis influências ou origens comuns.

**Exemplo e Possíveis Insights:**

- Similaridades fonológicas podem sugerir contatos históricos entre povos.
- Diferenças podem indicar caminhos evolutivos distintos.

---

**4. Análises Estatísticas e Visualizações**

Após calcular as similaridades, diversas análises estatísticas são realizadas para entender as relações entre elas.

### 4.1 Cálculo de Correlações

**Função:** `calcular_correlacoes_avancadas`

**Metodologia:**

- Calcula as correlações de **Pearson**, **Spearman** e **Kendall** entre as medidas de similaridade.

**Fórmulas:**

1. **Correlação de Pearson:**

   - Mede a relação linear entre duas variáveis.

   - Fórmula:

     \[
     r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n}(Y_i - \bar{Y})^2}}
     \]

   - Onde \( \bar{X} \) e \( \bar{Y} \) são as médias de \( X \) e \( Y \).

2. **Correlação de Spearman:**

   - Mede a relação monotônica (não necessariamente linear) entre duas variáveis usando os postos (ranks) dos dados.

   - Fórmula:

     \[
     \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
     \]

     Onde \( d_i \) é a diferença entre os postos de \( X_i \) e \( Y_i \).

3. **Correlação de Kendall:**

   - Mede a concordância entre os rankings de duas variáveis.

   - Fórmula:

     \[
     \tau = \frac{C - D}{\frac{1}{2} n(n - 1)}
     \]

     Onde:

     - \( C \) é o número de pares concordantes.
     - \( D \) é o número de pares discordantes.

**Objetivo:** Identificar se há relações significativas entre diferentes medidas de similaridade.

**Justificativa:** Correlações podem indicar que certas dimensões de similaridade estão relacionadas, sugerindo que uma pode ser preditiva da outra.

**Exemplo e Possíveis Insights:**

- Uma alta correlação entre similaridades semânticas e lexicais pode indicar que palavras semelhantes carregam significados semelhantes.
- Baixas correlações podem sugerir que as dimensões analisadas capturam aspectos independentes das línguas.

---

### 4.2 Regressão Linear

**Função:** `regressao_linear`

**Metodologia:**

- Ajusta um modelo de regressão linear simples entre duas variáveis de similaridade.

**Modelo:**

- Fórmula do modelo de regressão linear:

  \[
  y = \beta_0 + \beta_1 x + \epsilon
  \]

  Onde:

  - \( y \) é a variável dependente.
  - \( x \) é a variável independente.
  - \( \beta_0 \) é o intercepto.
  - \( \beta_1 \) é o coeficiente angular.
  - \( \epsilon \) é o termo de erro.

**Passos:**

1. **Estimação dos Parâmetros:**

   - Utiliza o método dos **Mínimos Quadrados Ordinários (OLS)** para estimar \( \beta_0 \) e \( \beta_1 \).

   - As estimativas são dadas por:

     \[
     \hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
     \]

     \[
     \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
     \]

2. **Teste de Significância:**

   - Verifica se o coeficiente \( \beta_1 \) é significativamente diferente de zero usando um teste t.

   - Estatística t:

     \[
     t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}
     \]

     Onde \( SE(\hat{\beta}_1) \) é o erro padrão de \( \hat{\beta}_1 \).

3. **Análise dos Resíduos:**

   - Verifica se os resíduos \( \hat{\epsilon}_i = y_i - \hat{y}_i \) seguem uma distribuição normal e se há homocedasticidade (variância constante dos resíduos).

**Objetivo:** Avaliar se há uma relação linear significativa entre duas medidas de similaridade.

**Justificativa:** Se uma medida pode ser predita por outra, isso simplifica a análise e pode indicar causalidade ou dependência.

**Exemplo e Possíveis Insights:**

- Se a similaridade semântica entre Dzubukuá e Arcaico prediz a similaridade com o Moderno, isso sugere uma continuidade semântica ao longo do tempo.

---

### 4.3 Regressão Múltipla

**Função:** `regressao_multipla`

**Metodologia:**

- Extensão da regressão linear para múltiplas variáveis independentes.

**Modelo:**

- Fórmula do modelo de regressão múltipla:

  \[
  y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
  \]

**Passos:**

1. **Estimação dos Parâmetros:**

   - Utiliza OLS para estimar os coeficientes \( \beta_i \).

2. **Verificação de Multicolinearidade:**

   - Calcula o **Fator de Inflação da Variância (VIF)** para cada variável independente.

   - Fórmula do VIF para a variável \( x_j \):

     \[
     \text{VIF}_j = \frac{1}{1 - R_j^2}
     \]

     Onde \( R_j^2 \) é o coeficiente de determinação da regressão de \( x_j \) sobre todas as outras variáveis independentes.

   - Valores de VIF acima de 5 ou 10 indicam multicolinearidade preocupante.

3. **Teste de Significância dos Coeficientes:**

   - Cada coeficiente \( \beta_i \) é testado para verificar se é significativamente diferente de zero.

**Objetivo:** Entender como múltiplas medidas de similaridade influenciam conjuntamente a variável dependente.

**Justificativa:** As línguas são sistemas complexos onde várias dimensões podem interagir. A regressão múltipla captura essas interações.

**Exemplo e Possíveis Insights:**

- Pode identificar quais medidas de similaridade têm maior impacto na similaridade semântica com o Português Moderno.
- Multicolinearidade pode indicar que algumas medidas são redundantes.

---

### 4.4 Análise de Variância (ANOVA)

**Função:** `analise_anova`

**Metodologia:**

- Compara as médias de três ou mais grupos para verificar se há diferenças estatisticamente significativas.

**Fórmula:**

- Estatística F:

  \[
  F = \frac{\text{Variância entre grupos}}{\text{Variância dentro dos grupos}}
  \]

- Onde:

  - Variância entre grupos (MSG):

    \[
    \text{MSG} = \frac{\sum_{k=1}^{K} n_k (\bar{X}_k - \bar{X})^2}{K - 1}
    \]

  - Variância dentro dos grupos (MSE):

    \[
    \text{MSE} = \frac{\sum_{k=1}^{K} \sum_{i=1}^{n_k} (X_{ik} - \bar{X}_k)^2}{N - K}
    \]

  - \( K \) é o número de grupos.
  - \( n_k \) é o tamanho do grupo \( k \).
  - \( N \) é o total de observações.

**Objetivo:** Determinar se as médias das similaridades diferem significativamente entre os pares de línguas.

**Justificativa:** Identificar se as diferenças observadas são devidas ao acaso ou refletem diferenças reais entre as línguas.

**Exemplo e Possíveis Insights:**

- Uma ANOVA significativa sugere que pelo menos um par de línguas difere em sua similaridade média.
- Pode motivar análises post-hoc para identificar quais pares diferem.

---

### 4.5 Testes de Hipóteses

**Função:** `testes_hipotese`

**Metodologia:**

- Realiza o **teste t para duas amostras independentes**.

**Fórmula:**

- Estatística t:

  \[
  t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}
  \]

  Onde:

  - \( \bar{X}_1 \) e \( \bar{X}_2 \) são as médias das amostras.
  - \( S_1^2 \) e \( S_2^2 \) são as variâncias das amostras.
  - \( n_1 \) e \( n_2 \) são os tamanhos das amostras.

**Objetivo:** Testar se as médias de duas medidas de similaridade são significativamente diferentes.

**Justificativa:** Validar hipóteses específicas sobre as relações entre as línguas.

**Exemplo e Possíveis Insights:**

- Verificar se a similaridade semântica entre Dzubukuá e Arcaico é diferente da similaridade entre Dzubukuá e Moderno.

---

### 4.6 Análise de Componentes Principais (PCA)

**Função:** `aplicar_pca`

**Metodologia:**

- Reduz a dimensionalidade dos dados transformando as variáveis originais em componentes principais ortogonais que explicam a maior parte da variância.

**Passos:**

1. **Centralização dos Dados:**

   - Subtrai-se a média de cada variável:

     \[
     X_{\text{centralizado}} = X - \bar{X}
     \]

2. **Cálculo da Matriz de Covariância:**

   - Matriz \( \mathbf{C} \) de covariâncias entre as variáveis:

     \[
     \mathbf{C} = \frac{1}{n - 1} X_{\text{centralizado}}^T X_{\text{centralizado}}
     \]

3. **Autovalores e Autovetores:**

   - Calcula-se os autovalores \( \lambda \) e autovetores \( \vec{e} \) de \( \mathbf{C} \):

     \[
     \mathbf{C} \vec{e} = \lambda \vec{e}
     \]

4. **Componentes Principais:**

   - Os autovetores correspondem aos componentes principais.
   - Os autovalores indicam a variância explicada por cada componente.

**Objetivo:** Identificar padrões nos dados e reduzir a complexidade dimensional.

**Justificativa:** Facilita a visualização dos dados e pode revelar agrupamentos ou relações não evidentes nas dimensões originais.

**Exemplo e Possíveis Insights:**

- Componentes que explicam grande parte da variância podem representar combinações de medidas de similaridade significativas.
- Visualização em 2D pode revelar clusters de frases semelhantes.

---

### 4.7 Análise de Agrupamentos (Clustering)

**Funções:** `analise_clustering` e `visualizar_clusters`

**Metodologia:**

- **K-Means Clustering:** Agrupa os dados em \( k \) clusters baseados na minimização da soma das distâncias quadradas dentro dos clusters.

**Passos K-Means:**

1. **Inicialização:**

   - Seleciona \( k \) centroides iniciais aleatoriamente.

2. **Atribuição:**

   - Cada ponto é atribuído ao cluster com o centróide mais próximo.

3. **Atualização:**

   - Recalcula-se o centróide de cada cluster como a média dos pontos atribuídos a ele.

4. **Iteração:**

   - Repete os passos 2 e 3 até a convergência (quando os centroides não mudam significativamente).

**Método Elbow:**

- Avalia a **inércia** (soma das distâncias quadradas dentro dos clusters) para diferentes valores de \( k \).
- O ponto onde a redução na inércia começa a diminuir (formando um "cotovelo") é escolhido como \( k \) ótimo.

**Objetivo:** Identificar agrupamentos naturais nos dados.

**Justificativa:** Pode revelar grupos de frases com características semelhantes, indicando padrões linguísticos.

**Exemplo e Possíveis Insights:**

- Clusters podem representar grupos de frases que evoluíram de maneira semelhante.
- Comparar clusters entre métodos (K-Means e DBSCAN) aumenta a robustez da análise.

---

### 4.8 Ajuste de Distribuição q-Exponencial

**Função:** `ajuste_q_exponencial`

**Metodologia:**

- Ajusta uma distribuição q-exponencial aos dados, relevante em sistemas complexos.

**Fórmula:**

- Função q-exponencial:

  \[
  f(x) = a \left[1 - (1 - q) b x \right]^{\frac{1}{1 - q}}
  \]

  Onde:

  - \( a \), \( b \) e \( q \) são parâmetros de ajuste.
  - \( q \) é o parâmetro de entropia não-extensiva de Tsallis.

**Objetivo:** Modelar a distribuição dos dados de similaridade e capturar comportamentos não-exponenciais.

**Justificativa:** Dados de sistemas complexos muitas vezes não seguem distribuições exponenciais clássicas. O ajuste q-exponencial pode fornecer um modelo mais preciso.

**Exemplo e Possíveis Insights:**

- Valores de \( q \neq 1 \) indicam desvios da distribuição exponencial, sugerindo a presença de efeitos de memória ou interações de longo alcance.

---

### 4.9 Visualizações

**Funções:** `grafico_interativo_plotly`, `grafico_regressao_plotly`, `grafico_dendrograma`, `grafico_matriz_correlacao`

**Objetivo:** Fornecer representações visuais dos resultados para facilitar a interpretação.

**Justificativa:** Visualizações permitem identificar padrões, tendências e anomalias que podem não ser evidentes apenas com números.

---

**5. Considerações e Resalvas Técnicas**

- **Qualidade dos Dados:**

  - A confiabilidade das análises depende da qualidade e representatividade dos dados.

- **Assunções Estatísticas:**

  - Algumas análises assumem normalidade dos dados, homocedasticidade, independência, entre outras.

- **Multicolinearidade:**

  - Pode afetar a interpretação dos coeficientes em regressões múltiplas.

- **Número de Observações:**

  - Amostras pequenas podem não fornecer poder estatístico suficiente.

- **Interpretação Cautelosa:**

  - Correlações não implicam causalidade.
  - Resultados devem ser interpretados no contexto linguístico e histórico.

---

**Conclusão**

As análises realizadas fornecem uma visão abrangente das similaridades entre Dzubukuá, Português Arcaico e Português Moderno. Ao combinar técnicas de processamento de linguagem natural e estatística, é possível extrair insights sobre a evolução linguística, influências entre línguas e preservação de conceitos ao longo do tempo.

É importante interpretar os resultados considerando as limitações e contexto, buscando sempre complementar as análises quantitativas com conhecimentos qualitativos em linguística e história.

---

**Referências**

- Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics. *Journal of Statistical Physics*, 52(1-2), 479-487.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Jurafsky, D., & Martin, J. H. (2009). *Speech and Language Processing*. Prentice Hall.
    """)
   
    # Informações de contato
    st.sidebar.image("eu.ico", width=80)
    # Exibe uma imagem (provavelmente uma foto ou ícone do criador) na barra lateral.
    # Vantagem: A presença de uma imagem de contato dá uma personalização e identidade à aplicação, além de humanizar o projeto.
    # Desvantagem: Se a imagem não estiver disponível, pode gerar um erro ou deixar um espaço vazio na interface.
    # Possível solução: Incluir uma verificação para garantir que a imagem existe antes de carregá-la, e fornecer uma imagem alternativa se não estiver disponível.

    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.
    """)
    # Mostra uma breve descrição e o nome do responsável pelo projeto.
    # Vantagem: Facilita a identificação do criador ou responsável pela aplicação.
    # Desvantagem: O texto é fixo e não permite personalização dinâmica.
    # Possível solução: Tornar esse bloco dinâmico, permitindo que o responsável possa ser alterado conforme o projeto se expande ou outros colaboradores entram no projeto.

    st.sidebar.write("""
    Contatos: marceloclaro@gmail.com
    Whatsapp: (88)981587145
    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)
# _____________________________________________
# Controle de Áudio

st.sidebar.title("Controle de Áudio")

# O comando `st.sidebar.title` exibe um título na barra lateral. Neste caso, o título "Controle de Áudio" informa ao usuário que há uma seção dedicada ao controle de áudio.
# Vantagem: A criação de um título claro e destacado facilita a navegação e a compreensão da interface pelo usuário.
# Desvantagem: O título é estático e não reflete o conteúdo dinâmico, como a lista de arquivos ou o estado atual (por exemplo, se o áudio está tocando).
# Possível solução: Tornar o título dinâmico, atualizando-o conforme o áudio selecionado ou o estado do player, como "Controle de Áudio (Tocando)".
# Exemplo:
# ```python
# st.sidebar.title(f"Controle de Áudio (Tocando: {selected_mp3})" if play_button else "Controle de Áudio")
# ```

# Lista de arquivos MP3
mp3_files = {
    "Áudio explicação técnica": "kariri.mp3",
    
}
# O dicionário `mp3_files` mapeia nomes amigáveis de arquivos de áudio (como "Instrução de uso") para os respectivos nomes dos arquivos MP3.
# Vantagem: Facilita a exibição de nomes descritivos para os arquivos de áudio, permitindo ao usuário escolher com base no contexto, sem ver os nomes reais dos arquivos.
# Desvantagem: O código assume que todos os arquivos listados estão presentes no sistema, sem verificação prévia.
# Possível solução: Implementar uma verificação para garantir que todos os arquivos de áudio listados estão disponíveis antes de exibi-los na interface.
# Exemplo:
# ```python
# valid_mp3_files = {k: v for k, v in mp3_files.items() if os.path.exists(v)}
# ```

# Controle de seleção de música
selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))  # Certifique-se de usar 'options'
# O widget `st.sidebar.radio` cria um botão de seleção (radio button) na barra lateral, permitindo que o usuário escolha entre as opções fornecidas (os nomes amigáveis de MP3).
# Vantagem: Simples e intuitivo, o controle radio limita a seleção a uma opção por vez, o que é ideal quando o usuário precisa escolher apenas um arquivo de áudio.
# Desvantagem: Se houver muitos arquivos de áudio, a lista pode ficar muito longa, dificultando a navegação.
# Possível solução: Adicionar uma funcionalidade de busca ou agrupamento para facilitar a seleção de arquivos em listas grandes.
# Exemplo:
# ```python
# selected_mp3 = st.sidebar.selectbox("Escolha uma música", options=list(mp3_files.keys()))  # Para listas maiores
# ```

# Opção de loop
loop = st.sidebar.checkbox("Repetir áudio")
# O widget `st.sidebar.checkbox` cria uma caixa de seleção que permite ao usuário definir se deseja repetir a música (loop).
# Vantagem: O controle é direto e funcional, permitindo que o usuário ative ou desative a repetição de forma simples.
# Desvantagem: Não há feedback visual imediato sobre o estado atual da música (se está em loop ou não) no player de áudio.
# Possível solução: Atualizar dinamicamente o título ou o player para refletir o estado de loop, como "Repetindo: Sim/Não" na interface.
# Exemplo:
# ```python
# st.sidebar.text(f"Repetindo: {'Sim' if loop else 'Não'}")
# ```

# Botão de play
play_button = st.sidebar.button("Play")
# O widget `st.sidebar.button` cria um botão de "Play" que, quando clicado, dispara o carregamento e a execução do áudio selecionado.
# Vantagem: A interação via botão é intuitiva e tradicional para controles de áudio.
# Desvantagem: Não há indicação visual clara de que o áudio está sendo tocado ou foi pausado, além do próprio botão.
# Possível solução: Alterar o rótulo do botão para "Pausar" enquanto o áudio está tocando ou adicionar um botão de "Parar".
# Exemplo:
# ```python
# play_button = st.sidebar.button("Pausar" if is_playing else "Play")
# ```

# Carregar e exibir o player de áudio
audio_placeholder = st.sidebar.empty()  # Placeholder para o player de áudio
# `st.sidebar.empty()` cria um espaço reservado (placeholder) onde o player de áudio será exibido. Este espaço vazio será preenchido posteriormente com o player HTML.
# Vantagem: O placeholder permite a exibição condicional de conteúdo (como o player de áudio) de maneira controlada e dinâmica.
# Desvantagem: O uso de um placeholder vazio pode causar confusão visual se o espaço não for preenchido adequadamente após a interação.
# Possível solução: Exibir uma mensagem de "Aguardando seleção de música" antes de o áudio ser carregado, para que o espaço não fique visualmente vazio.
# Exemplo:
# ```python
# audio_placeholder.text("Selecione uma música para tocar")
# ```

if play_button and selected_mp3:
    # A condição `if play_button and selected_mp3:` verifica se o botão "Play" foi clicado e se uma música foi selecionada. Somente nesse caso o áudio será carregado.
    # Vantagem: Garante que o áudio só seja carregado quando o usuário realizar as duas ações necessárias (selecionar e clicar em play).
    # Desvantagem: Não há feedback caso o usuário clique no botão sem selecionar uma música.
    # Possível solução: Adicionar uma mensagem de erro ou aviso caso o usuário clique em "Play" sem selecionar um áudio.
    # Exemplo:
    # ```python
    # if play_button and not selected_mp3:
    #     st.sidebar.error("Por favor, selecione uma música antes de clicar em play.")
    # ```

    mp3_path = mp3_files[selected_mp3]
    # `mp3_path` armazena o caminho do arquivo MP3 correspondente à música selecionada pelo usuário.
    # Vantagem: A extração do caminho do arquivo com base no nome selecionado é eficiente e intuitiva.
    # Desvantagem: O caminho é assumido como correto sem verificação prévia.
    # Possível solução: Verificar se o arquivo realmente existe antes de prosseguir com o carregamento do áudio, evitando erros de arquivo inexistente.
    # Exemplo:
    # ```python
    # if not os.path.exists(mp3_path):
    #     st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
    # ```

    try:
        with open(mp3_path, "rb") as audio_file:
            # Abre o arquivo de áudio no modo de leitura binária (rb).
            # Vantagem: O uso do modo binário é essencial para ler corretamente o conteúdo do arquivo MP3.
            # Desvantagem: Se o arquivo estiver corrompido ou indisponível, pode gerar uma exceção que não está sendo tratada adequadamente.
            # Possível solução: Ampliar o tratamento de exceções para lidar com outros possíveis erros além de `FileNotFoundError`, como permissões ou arquivos corrompidos.

            audio_bytes = audio_file.read()
            # Lê o conteúdo do arquivo MP3 em bytes.
            # Vantagem: O uso de bytes permite manipulação e transmissão eficiente dos dados do áudio.
            # Desvantagem: Não há validação para garantir que o conteúdo lido seja realmente um arquivo MP3 válido.
            # Possível solução: Adicionar uma verificação básica para garantir que o arquivo é válido ou está no formato esperado.

            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            # Codifica o áudio lido em base64 para ser embutido no HTML.
            # Vantagem: A codificação base64 permite a inserção do áudio diretamente em HTML, sem a necessidade de um servidor de mídia externo.
            # Desvantagem: Arquivos de áudio grandes podem gerar strings base64 enormes, sobrecarregando a transferência de dados e o carregamento da página.
            # Possível solução: Limitar o tamanho dos arquivos MP3 ou adicionar suporte a streaming em vez de carregar o arquivo inteiro.
            # Exemplo:
            # ```python
            # if len(audio_bytes) > MAX_SIZE:
            #     st.sidebar.error("O arquivo de áudio é muito grande.")
            # ```

            loop_attr = "loop" if loop else ""
            # A variável `loop_attr` é uma string condicional que define o atributo `loop` no player de áudio HTML.
            # Vantagem: Condicional simples e clara para habilitar ou desabilitar a repetição do áudio.
            # Desvantagem: Se o atributo `loop` for omitido acidentalmente, o comportamento pode não ser o esperado.
            # Possível solução: Garantir que o atributo seja aplicado corretamente, adicionando mais verificações sobre o estado do checkbox de loop.

            audio_html = f"""
            <audio id="audio-player" controls autoplay {loop_attr}>
              <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
              Seu navegador não suporta o elemento de áudio.
            </audio>
            """
            # A string `audio_html` define o player de áudio HTML com controle de autoplay e loop, se necessário.
            # Vantagem: Usar HTML embutido para exibir o player de áudio é eficiente e direto, sem a necessidade de bibliotecas externas.
            # Desvantagem: Dependente do suporte do navegador para HTML5 e a tag <audio>. Alguns navegadores ou versões podem não suportar.
            # Possível solução: Adicionar um fallback para navegadores que não suportam a tag <audio>, ou instruções para atualização de navegador.

            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
            # A função `audio_placeholder.markdown` insere o HTML gerado no espaço reservado da barra lateral, exibindo o player de áudio.
            # Vantagem: O uso de `unsafe_allow_html=True` permite injetar HTML diretamente, permitindo uma personalização completa da interface de áudio.
            # Desvantagem: O uso de `unsafe_allow_html` pode abrir brechas de segurança se o conteúdo HTML não for validado corretamente.
            # Possível solução: Certificar-se de que o HTML gerado é seguro, especialmente se partes dele forem dinâmicas ou controladas pelo usuário.

    except FileNotFoundError:
        audio_placeholder.error(f"Arquivo {mp3_path} não encontrado.")
        # Se o arquivo MP3 não for encontrado, uma mensagem de erro é exibida no espaço reservado (audio_placeholder).
        # Vantagem: A função de erro alerta o usuário de maneira clara quando um arquivo está ausente.
        # Desvantagem: O tratamento de exceção está limitado ao `FileNotFoundError`, mas pode haver outros erros de I/O que não são capturados.
        # Possível solução: Ampliar o bloco `except` para capturar outros tipos de erros de arquivo.
        # Exemplo:
        # ```python
        # except (FileNotFoundError, IOError) as e:
        #     audio_placeholder.error(f"Erro ao carregar o arquivo: {str(e)}")
        # ```

# _____________________________________________




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
