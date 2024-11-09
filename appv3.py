sudo apt update
sudo apt install libtesseract-dev libleptonica-dev

import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import openai
import pandas as pd
from jiwer import wer, cer  # Adicionada taxa de erro de caracteres (CER)
import logging
import traceback
import time
from fpdf import FPDF  # Biblioteca para criar relatórios em PDF

# Configuração do log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='ocr_analysis.log')

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

# Função para verificar a qualidade da imagem
def check_image_quality(image):
    width, height = image.size
    if width < 500 or height < 500:
        st.warning("A imagem pode estar com baixa qualidade. Considere usar uma imagem de maior resolução.")
        logging.info("Imagem com qualidade abaixo do esperado.")
    else:
        logging.info("Qualidade da imagem verificada e aprovada.")

# Função para pré-processar a imagem para OCR
def preprocess_image(image):
    try:
        logging.info("Pré-processamento de imagem iniciado.")
        image = image.convert('L')  # Converter para escala de cinza
        image = image.filter(ImageFilter.MedianFilter())  # Aplicar filtro de mediana para redução de ruído
        image = ImageOps.equalize(image)  # Equalização de histograma para melhorar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Aumentar o contraste
        logging.info("Pré-processamento de imagem concluído.")
        return image
    except Exception as e:
        logging.error(f"Erro no pré-processamento da imagem: {e}")
        st.error("Erro no pré-processamento da imagem. Verifique a qualidade da imagem.")
        return None

# Função para realizar OCR com configurações avançadas
def ocr_image_advanced(file_path, oem, psm, lang):
    try:
        start_time = time.time()
        logging.info(f"Realizando OCR na imagem com OEM={oem} e PSM={psm}.")
        custom_config = f'--oem {oem} --psm {psm}'
        image = Image.open(file_path)
        check_image_quality(image)  # Verificar qualidade da imagem
        image = preprocess_image(image)  # Aplicar pré-processamento
        if image is None:
            return ""
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        logging.info("OCR realizado com sucesso.")
        end_time = time.time()
        logging.info(f"Tempo de execução do OCR: {end_time - start_time:.2f} segundos.")
        return text
    except Exception as e:
        logging.error(f"Erro durante o OCR: {e}\n{traceback.format_exc()}")
        st.error("Erro durante o OCR. Verifique a imagem e as configurações.")
        return ""

# Função para calcular a taxa de erro de palavras (WER) e caracteres (CER)
def calculate_errors(reference, hypothesis):
    try:
        word_error = wer(reference, hypothesis)
        char_error = cer(reference, hypothesis)
        logging.info(f"WER calculado: {word_error:.2%}, CER calculado: {char_error:.2%}")
        return word_error, char_error
    except Exception as e:
        logging.error(f"Erro ao calcular WER/CER: {e}\n{traceback.format_exc()}")
        st.error("Erro ao calcular WER/CER. Verifique o texto de referência.")
        return None, None

# Função para processar texto com OpenAI
def process_text_with_openai(prompt):
    try:
        start_time = time.time()
        logging.info("Enviando texto para análise com OpenAI.")
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-4" ou "gpt-3.5-turbo" conforme necessário
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        logging.info(f"Tempo de execução da análise com OpenAI: {end_time - start_time:.2f} segundos.")
        logging.info("Análise com OpenAI concluída.")
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Erro durante a análise com OpenAI: {e}\n{traceback.format_exc()}")
        st.error("Erro durante a análise com OpenAI. Verifique a chave da API e a conectividade.")
        return ""

# Função para criar DataFrame com análise da OpenAI
def create_dataframe_with_openai_analysis(text):
    try:
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
        logging.info("DataFrame criado com a análise da OpenAI.")
        return pd.DataFrame([data])
    except Exception as e:
        logging.error(f"Erro ao criar DataFrame com análise da OpenAI: {e}\n{traceback.format_exc()}")
        st.error("Erro ao criar DataFrame com a análise da OpenAI.")
        return pd.DataFrame()

# Função para gerar relatório em PDF
def generate_pdf_report(ocr_text, wer_value, cer_value):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Análise de OCR e OpenAI", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, f"Texto extraído do OCR:\n{ocr_text}\n")
    pdf.cell(0, 10, f"Taxa de erro de palavras (WER): {wer_value:.2%}")
    pdf.cell(0, 10, f"Taxa de erro de caracteres (CER): {cer_value:.2%}", ln=True)

    pdf.output("relatorio_ocr_openai.pdf")
    logging.info("Relatório em PDF gerado com sucesso.")

# Interface do Streamlit para upload de imagem
st.title("OCR e Análise Linguística com OpenAI")
uploaded_file = st.file_uploader("Faça upload de uma imagem", type=file_format)

if uploaded_file:
    # Realizar OCR com configurações avançadas e criar DataFrame com análise da OpenAI
    ocr_result = ocr_image_advanced(uploaded_file, oem_mode, psm_mode, ocr_language)
    st.subheader("Texto extraído da imagem:")
    st.text(ocr_result)

    # Validar a extração com um texto de referência (opcional)
    reference_text = st.text_area("Texto de referência para validação WER/CER (opcional):")
    if reference_text:
        word_error, char_error = calculate_errors(reference_text, ocr_result)
        if word_error is not None and char_error is not None:
            st.write(f"Taxa de erro de palavras (WER): {word_error:.2%}")
            st.write(f"Taxa de erro de caracteres (CER): {char_error:.2%}")
            
            # Gerar relatório em PDF
            if st.button("Gerar Relatório em PDF"):
                generate_pdf_report(ocr_result, word_error, char_error)
                with open("relatorio_ocr_openai.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Baixar Relatório em PDF",
                        data=pdf_file,
                        file_name="relatorio_ocr_openai.pdf",
                        mime="application/pdf"
                    )

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

