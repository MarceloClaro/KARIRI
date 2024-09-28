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

with st.sidebar.expander("Insights do Código"):
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
        """))
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
    O código do Agentes Alan Kay é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.
    """)
    # A função `st.markdown` permite que o texto seja exibido em formato Markdown. Esse bloco de texto fornece uma introdução à análise do código.
    # Vantagem: Markdown é uma maneira simples e eficiente de formatar texto com títulos, listas e links, facilitando a leitura.
    # Desvantagem: Markdown é limitado em termos de interatividade e formatação avançada comparado ao HTML puro.
    # Possível solução: Oferecer uma interface mais interativa usando botões ou links que levem a seções específicas de análise detalhada.

    # Seções detalhadas do código (Inovações, Pontos Positivos, Limitações)
    st.markdown("""
    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas do modelo de linguagem.
    """)
    # Vantagem: Organiza as inovações do código de forma clara, ajudando o usuário a entender as funcionalidades mais avançadas da aplicação.
    # Desvantagem: A lista é puramente informativa e não interativa.
    # Possível solução: Adicionar exemplos práticos que mostrem como cada inovação funciona na prática, permitindo ao usuário experimentar as funcionalidades ao mesmo tempo que lê sobre elas.

    st.markdown("""
    **Pontos positivos:**
    - Personalização: O aplicativo permite que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas de acordo com suas necessidades.
    - Precisão: A integração com a API Groq e o refinamento de respostas garantem que as respostas sejam precisas e relevantes para a consulta.
    - Flexibilidade: O código é flexível o suficiente para permitir que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas.
    """)
    # O bloco de pontos positivos destaca as principais qualidades do código.
    # Vantagem: Facilita a visualização dos aspectos fortes da aplicação, ajudando a destacar suas capacidades para os usuários.
    # Desvantagem: Não há links para exemplos ou tutoriais que expliquem como tirar proveito dessas qualidades na prática.
    # Possível solução: Criar links para tutoriais ou guias interativos que mostrem como configurar e utilizar as opções de personalização e flexibilidade.

    st.markdown("""
    **Limitações:**
    - Dificuldade de uso: O aplicativo pode ser difícil de usar para os usuários que não têm experiência com modelos de linguagem ou API.
    - Limitações de token: O código tem limitações em relação ao número de tokens que podem ser processados pelo modelo de linguagem.
    - Necessidade de treinamento adicional: O modelo de linguagem pode precisar de treinamento adicional para lidar com consultas mais complexas ou específicas.
    """)
    # Apresenta as limitações da aplicação de maneira clara e objetiva.
    # Vantagem: Transparência sobre as limitações do sistema, permitindo que os usuários entendam onde podem surgir problemas ou desafios.
    # Desvantagem: Não há soluções sugeridas para lidar com essas limitações diretamente na aplicação.
    # Possível solução: Incluir sugestões ou links para documentação que explique como superar essas limitações, como usar técnicas de compressão de tokens ou melhorar o desempenho de modelos com grandes conjuntos de dados.

    st.markdown("""
    **Importância de ter colocado instruções em chinês:**
    A linguagem chinesa tem uma densidade de informação mais alta do que muitas outras línguas, o que significa que os modelos de linguagem precisam processar menos tokens para entender o contexto e gerar respostas precisas. Isso torna a linguagem chinesa mais apropriada para a utilização de modelos de linguagem com baixa quantidade de tokens. Portanto, ter colocado instruções em chinês no código é um recurso importante para garantir que o aplicativo possa lidar com consultas em chinês de forma eficaz.
    """)
    # Explica a importância de otimizar o processamento em línguas como o chinês, que tem alta densidade de informação por token.
    # Vantagem: Aponta uma solução inteligente para maximizar o uso eficiente de tokens em modelos de linguagem.
    # Desvantagem: Focado em um caso específico (língua chinesa), sem explorar outros cenários em diferentes línguas ou dialetos.
    # Possível solução: Ampliar a análise para incluir outras línguas que também podem ter características vantajosas no processamento por tokens, ou oferecer suporte a múltiplas línguas com estratégias específicas.

    st.markdown("""
    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais.
    """)
    # O texto de resumo oferece uma visão final sobre a aplicação, destacando sua inovação e sugerindo melhorias.
    # Vantagem: Conclui a análise de forma clara e com recomendações.
    # Desvantagem: O texto é genérico e não explora possíveis planos de ação detalhados para melhorar a aplicação.
    # Possível solução: Oferecer um plano de ação prático com etapas recomendadas para superar as limitações, como a expansão de capacidade de processamento de tokens ou a simplificação da interface para usuários iniciantes.

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
    # Exibe as informações de contato do responsável (e-mail, WhatsApp e Instagram).
    # Vantagem: Facilita a comunicação entre os usuários e o criador, permitindo feedback direto ou consultas.
    # Desvantagem: Informações de contato fixas podem se desatualizar com o tempo (por exemplo, mudanças de número ou e-mail).
    # Possível solução: Usar variáveis externas para armazenar essas informações, facilitando a atualização em caso de mudanças. Exemplo:
    # ```python
    # contato_email = "marceloclaro@gmail.com"
    # contato_whatsapp = "(88)981587145"
    # contato_instagram = "https://www.instagram.com/marceloclaro.geomaker/"
    # st.sidebar.write(f"Contatos: {contato_email}\nWhatsapp: {contato_whatsapp}\nInstagram: {contato_instagram}")
    # ```

# Vantagens gerais:
# 1. **Informação organizada**: A estrutura da sidebar, com seções separadas para insights e informações de contato, mantém tudo organizado e fácil de acessar.
# 2. **Conteúdo expansível**: O uso de `expander` para ocultar/mostrar informações melhora a usabilidade, especialmente em interfaces mais densas.
# 3. **Contato fácil**: O usuário pode entrar em contato rapidamente com o criador via diferentes canais, o que ajuda a criar uma comunidade em torno do projeto.

# Desvantagens gerais:
# 1. **Conteúdo estático**: Algumas partes do código são estáticas e exigem alterações manuais caso as informações mudem (como contato ou títulos).
# 2. **Falta de interatividade**: Embora a barra lateral tenha muitas informações, faltam elementos interativos que incentivem a navegação ou a experimentação.
# 3. **Dependência de arquivos externos**: A ausência ou falha ao carregar as imagens externas pode prejudicar a aparência da interface.

# Possíveis soluções:
# 1. **Conteúdo dinâmico**: Usar variáveis configuráveis para facilitar a atualização de informações, como os dados de contato e descrições.
# 2. **Links interativos**: Adicionar mais links interativos que direcionem para seções ou funcionalidades relevantes da aplicação.
# 3. **Verificação de imagem**: Implementar verificações automáticas para garantir que as imagens estejam disponíveis antes de carregá-las.

# Inovações:
# 1. **Barra lateral interativa**: Transformar a barra lateral em uma área interativa com menus, temas personalizados e opções dinâmicas.
# 2. **Feedback do usuário**: Adicionar uma seção de feedback onde os usuários possam enviar diretamente sugestões ou comentários através da sidebar.
# 3. **Análises em tempo real**: Incluir widgets que mostrem estatísticas ou insights em tempo real, como o número de usuários que utilizaram o sistema ou desempenho dos modelos de IA.

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
    st.title('Análises Avançadas de Similaridade Linguística para Línguas Mortas')

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
