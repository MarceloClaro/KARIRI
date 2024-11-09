import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import openai
import pandas as pd
from jiwer import wer  # Biblioteca para medir a taxa de erro de palavras (WER)

# Configuração da barra lateral no Streamlit
st.sidebar.title("Configurações da Aplicação")
api_key = st.sidebar.text_input("Chave da API da OpenAI", type="password")
psm_mode = st.sidebar.selectbox("Modo de Segmentação de Página (PSM)", [3, 6, 11], index=1)
oem_mode = st.sidebar.selectbox("Modo do Mecanismo OCR (OEM)", [0, 1, 2, 3], index=3)
ocr_language = st.sidebar.selectbox("Idioma para OCR", ['por', 'eng', 'spa', 'fra'], index=0, format_func=lambda x: {'por': 'Português', 'eng': 'Inglês', 'spa': 'Espanhol', 'fra': 'Francês'}[x])
file_format = st.sidebar.multiselect("Formatos de Arquivo Permitidos", ['jpg', 'png', 'jpeg', 'tiff'], default=['jpg', 'png', 'jpeg'])

# Verificar se a chave da API foi fornecida
if api_key:
    openai.api_key = api_key
else:
    st.warning("Por favor, insira a chave da API da OpenAI para continuar.")

# Função para pré-processar a imagem para OCR
def preprocess_image(image):
    image = image.convert('L')  # Converter para escala de cinza
    image = image.filter(ImageFilter.MedianFilter())  # Aplicar filtro de mediana para redução de ruído
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Aumentar o contraste
    return image

# Função para realizar OCR com configurações avançadas
def ocr_image_advanced(file_path, oem, psm, lang):
    custom_config = f'--oem {oem} --psm {psm}'
    image = Image.open(file_path)
    image = preprocess_image(image)  # Aplicar pré-processamento
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text

# Função para calcular a taxa de erro de palavras (WER)
def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

# Função para processar texto com OpenAI
def process_text_with_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-4" ou "gpt-3.5-turbo" conforme necessário
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Função para criar DataFrame com análise da OpenAI
def create_dataframe_with_openai_analysis(text):
    analysis_prompt = f"Analise o seguinte texto extraído da imagem e forneça: Análise Semântica, Análise Morfológica, Análise Fonológica, Contexto Cultural e outras informações relevantes:\n\n{text}"
    analysis_result = process_text_with_openai(analysis_prompt)
    
    data = {
        'Idioma': 'Português Moderno',  # Defina ou ajuste conforme necessário
        'Texto Original': text,
        'Tradução para o Português Moderno': '',  # Pode ser preenchido conforme a necessidade
        'Análise Fonológica': '',  # Pode ser preenchido com a resposta da OpenAI
        'Análise Morfológica': '',  # Pode ser preenchido com a resposta da OpenAI
        'Análise Sintática': '',  # Pode ser preenchido com a resposta da OpenAI
        'Análise Semântica': analysis_result,
        'IPA': '',
        'Análise Fonética': '',
        'Glossário (Palavra/Frase)': '',
        'Contexto Cultural': '',
        'Justificativa da Tradução': '',
        'Etimologia': '',
        'Correlato Gramatical': ''
    }
    return pd.DataFrame([data])

# Interface do Streamlit para upload de imagem
st.title("OCR e Análise Linguística com OpenAI")
uploaded_file = st.file_uploader("Faça upload de uma imagem", type=file_format)

if uploaded_file:
    # Realizar OCR com configurações avançadas e criar DataFrame com análise da OpenAI
    ocr_result = ocr_image_advanced(uploaded_file, oem_mode, psm_mode, ocr_language)
    st.subheader("Texto extraído da imagem:")
    st.text(ocr_result)

    # Validar a extração com um texto de referência (exemplo fictício)
    reference_text = st.text_area("Texto de referência para validação WER (opcional):")
    if reference_text:
        error_rate = calculate_wer(reference_text, ocr_result)
        st.write(f"Taxa de erro de palavras (WER): {error_rate:.2%}")

    # Criar o DataFrame com a análise da OpenAI
    if st.button("Analisar Texto"):
        df = create_dataframe_with_openai_analysis(ocr_result)
        st.subheader("Resultado da Análise:")
        st.dataframe(df)

        # Salvar o DataFrame em CSV
        output_csv_path = 'ocr_openai_analysis_advanced.csv'
        df.to_csv(output_csv_path, index=False)
        st.success(f"Arquivo CSV gerado com sucesso: {output_csv_path}")
        st.download_button(
            label="Baixar CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='ocr_openai_analysis_advanced.csv',
            mime='text/csv'
        )
