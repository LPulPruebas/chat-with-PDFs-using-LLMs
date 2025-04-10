import os
import logging
import asyncio
import time
from typing import List, Union

import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai # Importar google genai

# Langchain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Cambio: Usar embeddings de Google
from langchain_google_genai import ChatGoogleGenerativeAI # Asegurar importación correcta
from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage # Para manejar el historial de chat correctamente

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(asctime)s %(message)s"
)
logger = logging.getLogger(__name__)

# --- Google API Key Configuration ---
# Intenta obtener la clave de st.secrets primero, luego de variables de entorno
# --- Google API Key Configuration ---
# Intenta obtener la clave directamente de st.secrets (Cloud o local toml) o env var (local)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key no encontrada. Configúrala en los secretos de Streamlit Cloud (Settings -> Secrets) o localmente como variable de entorno (GOOGLE_API_KEY).")
    logger.error("GOOGLE_API_KEY no encontrada ni en st.secrets ni en os.getenv.")
    st.stop()
else:
    # Opcional: No loguear la clave misma, solo que se encontró
    logger.info("GOOGLE_API_KEY encontrada.")



# Configura la API de Google GenAI
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google GenAI configurado exitosamente.")
except Exception as e:
    st.error(f"Error configurando Google GenAI: {e}")
    logger.error(f"Error configurando Google GenAI: {e}")
    st.stop()
# --- Fin Configuración API Key ---


def get_pdf_text(pdf_docs: Union[str, list]) -> str: # Corregido: Retorna str, no List[str]
    """
    Extrae el texto de uno o varios archivos PDF.

    ### Arguments
    - `pdf_docs`: Una ruta de archivo (str) o una lista de objetos BytesIO (de st.file_uploader).

    ### Return
    Un string con todo el texto extraído de los PDFs.
    """
    text = ""
    try:
        if isinstance(pdf_docs, list):
            # Si es una lista (múltiples archivos subidos)
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        elif isinstance(pdf_docs, str):
            # Si es una ruta de archivo (no usado en este script con st.file_uploader)
             pdf_reader = PdfReader(pdf_docs)
             for page in pdf_reader.pages:
                 page_text = page.extract_text()
                 if page_text:
                     text += page_text
        else:
            # Si es un solo archivo subido (objeto BytesIO)
             pdf_reader = PdfReader(pdf_docs)
             for page in pdf_reader.pages:
                 page_text = page.extract_text()
                 if page_text:
                     text += page_text

        logger.info(f"Texto extraído de PDF(s), longitud: {len(text)} caracteres.")
        if not text:
             logger.warning("No se extrajo texto de los PDFs. ¿Están vacíos o son PDFs de imágenes sin OCR?")
             st.warning("No se pudo extraer texto de los PDFs. Asegúrate de que no sean solo imágenes.")

    except Exception as e:
        logger.error(f"Error al leer PDF: {e}")
        st.error(f"Error al procesar PDF: {e}")
        text = "" # Asegurarse de retornar string vacío en caso de error

    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Divide el texto en chunks usando RecursiveCharacterTextSplitter.

    ### Arguments
    - `text`: El texto (string) a dividir.

    ### Return
    Una lista de chunks de texto.
    """
    if not text:
        logger.warning("Se intentó dividir un texto vacío.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # Ligeramente aumentado, puedes ajustar
        chunk_overlap=200,
        length_function = len,
        is_separator_regex = False,
        separators=["\n\n", "\n", ".", ",", " ", ""], # Separadores comunes
    )

    chunks = text_splitter.split_text(text)
    logger.info(f"Texto dividido en {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks: List[str], persist_path: str = "faiss_index"):
    if not text_chunks:
        logger.error("No hay chunks de texto para crear el vector store.")
        st.error("No se pudo procesar el texto del documento para crear la base de conocimiento.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", maxBatchSize= 100)
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Guardar el índice en disco
        vector_store.save_local(persist_path)
        logger.info(f"Vector store FAISS guardado en '{persist_path}'.")
        return vector_store

    except Exception as e:
        logger.error(f"Error creando el vector store: {e}")
        st.error(f"Error al crear la base de conocimiento (vector store): {e}")
        return None

def load_vectorstore(persist_path: str = "faiss_index"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # 👈 habilita la deserialización
)
        logger.info(f"Vector store FAISS cargado desde '{persist_path}'.")
        return vector_store
    except Exception as e:
        logger.error(f"Error cargando vector store: {e}")
        st.error(f"No se pudo cargar la base de conocimiento guardada: {e}")
        return None

def get_conversation_chain(vectorstore):
    """
    Configura la cadena de conversación con Gemini y memoria.
    """
    if vectorstore is None:
        logger.error("No se puede crear la cadena de conversación sin un vector store.")
        st.error("Fallo al inicializar el asistente. No se pudo crear la base de conocimiento.")
        return None

    try:
        # --- LLM de Google Gemini ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", # <-- USA EL NOMBRE DEL MODELO ESTABLE
            temperature=0.3
            # convert_system_message_to_human=True # <-- QUITAR ESTO (obsoleto en nuevas versiones)
        )
        logger.info("LLM ChatGoogleGenerativeAI (gemini-1.0-pro) inicializado.")
        # --- FIN LLM ---

        retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) # Obtener 5 chunks relevantes
        logger.info("Retriever creado desde el vector store.")

        memory = ConversationBufferMemory(
            # llm=llm, # Puedes quitar el llm de la memoria si no necesitas resumen complejo
            input_key="question",
            output_key="answer",
            memory_key="chat_history",
            return_messages=True # Importante: retorna objetos HumanMessage/AIMessage
        )
        logger.info("Memoria de conversación inicializada.")

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True, # Opcional: útil para ver qué chunks se usaron
            verbose=False # Poner en True para debugging detallado en consola
        )
        logger.info("Cadena ConversationalRetrievalChain creada.")
        return chain

    except Exception as e:
        logger.error(f"Error creando la cadena de conversación: {e}")
        st.error(f"Error al inicializar el asistente de conversación: {e}")
        return None


def handle_user_input(user_question):
    """
    Maneja la entrada del usuario, llama a la cadena y muestra la conversación.
    """
    if st.session_state.conversation is None:
        st.warning("El asistente no está listo. Por favor, procesa los documentos primero.")
        logger.warning("Intento de usar handle_user_input sin conversación inicializada.")
        return

    try:
        # Asegurarse de que chat_history sea una lista (puede ser None al inicio)
        current_chat_history = st.session_state.chat_history or []

        # --- Llamada a la cadena usando invoke ---
        response = st.session_state.conversation.invoke({
            'question': user_question,
            'chat_history': current_chat_history
        })
        logger.info("Respuesta recibida de la cadena de conversación via invoke.")
        # --- Fin Llamada ---

        # Actualizar el historial en session_state
        # La cadena devuelve el historial actualizado (incluyendo la última Q&A)
        st.session_state.chat_history = response['chat_history']

        # --- Mostrar la conversación ---
        # Limpiar el área de chat y volver a mostrar todo el historial actualizado
        st.session_state.messages_placeholder.empty() # Limpia el contenedor
        with st.session_state.messages_placeholder.container(): # Vuelve a escribir en el contenedor
            if st.session_state.chat_history:
                 for i, message in enumerate(st.session_state.chat_history):
                    # Usar isinstance para determinar el tipo de mensaje
                    if isinstance(message, HumanMessage):
                        with st.chat_message(name="User", avatar="💃"):
                            st.write(message.content)
                    elif isinstance(message, AIMessage):
                         with st.chat_message(name="J.A.A.F.A.R.", avatar="🤖"):
                             st.write(message.content)
                    else: # Por si acaso hay otros tipos de mensajes
                        logger.warning(f"Tipo de mensaje inesperado en chat_history: {type(message)}")
                        st.write(f"*{message.content}*") # Mostrar de alguna forma

            # Opcional: Mostrar documentos fuente si están disponibles
            # if 'source_documents' in response and response['source_documents']:
            #    with st.expander("Fuentes consultadas"):
            #        for doc in response['source_documents']:
            #             st.write(f"- Chunk de '{doc.metadata.get('source', 'PDF desconocido')}':")
            #             st.caption(doc.page_content[:200] + "...") # Muestra inicio del chunk

    except Exception as e:
        logger.error(f"Error durante la ejecución de la cadena o mostrando la respuesta: {e}")
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")


def main():
    st.set_page_config(
        page_title="Chat con múltiples PDFs (Gemini)", # Título actualizado
        page_icon="📚",
        layout="wide"
    )

    # --- Inicialización de Session State ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        logger.info("Inicializando st.session_state.conversation a None.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # Se inicializará como lista vacía si es necesario
        logger.info("Inicializando st.session_state.chat_history a None.")
    if "pdf_processed" not in st.session_state:
         st.session_state.pdf_processed = False # Para saber si se hizo clic en "Process"
         logger.info("Inicializando st.session_state.pdf_processed a False.")
    # Contenedor para mensajes de chat (para poder limpiar y re-dibujar)
    if "messages_placeholder" not in st.session_state:
         st.session_state.messages_placeholder = st.empty()


    with st.sidebar:
        st.subheader("Gestión de Documentos")

        pdf_docs = st.file_uploader(
            "Sube tus PDFs aquí", accept_multiple_files=True, type="pdf"
        )

        if st.button("Guardar documentos en BD"):
             if not pdf_docs:
                 st.warning("Sube al menos un PDF.")
             else:
                 with st.spinner("Procesando y guardando documentos..."):
                     # --- INICIO: Añadir comprobación de event loop AQUÍ también ---
                     try:
                         loop = asyncio.get_event_loop()
                     except RuntimeError as ex:
                         if "There is no current event loop in thread" in str(ex):
                             loop = asyncio.new_event_loop()
                             asyncio.set_event_loop(loop)
                             logger.info("Creado nuevo event loop para el hilo de guardado.")
                     # --- FIN: Añadir comprobación de event loop ---

                     raw_text = get_pdf_text(pdf_docs)
                     if raw_text:
                         text_chunks = get_text_chunks(raw_text)
                         if text_chunks:
                             # get_vectorstore puede usar embeddings que necesitan el loop
                             vectorstore = get_vectorstore(text_chunks, persist_path="faiss_index")
                             if vectorstore: # Comprobar si se creó correctamente
                                 st.success("Documentos guardados en la base de conocimiento.")
                             # else: el error ya se mostró dentro de get_vectorstore
                         else:
                             st.error("No se pudo dividir el texto.")
                     else:
                         st.error("No se extrajo texto válido de los PDFs.")


        if st.button("Iniciar Chat con documentos guardados"):
            with st.spinner("Cargando documentos guardados..."):
                # --- INICIO: Añadir comprobación de event loop ---
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError as ex:
                    if "There is no current event loop in thread" in str(ex):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        logger.info("Creado nuevo event loop para el hilo de inicio de chat.")
                # --- FIN: Añadir comprobación de event loop ---

                vectorstore = load_vectorstore("faiss_index")
                if vectorstore:
                    # Esta es la llamada que probablemente causaba el error original
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation: # Verificar que la cadena se creó
                        st.session_state.chat_history = []
                        st.session_state.pdf_processed = True
                        st.success("¡Chat listo! Puedes hacer preguntas.")
                    # else: el error ya se mostró dentro de get_conversation_chain
                else:
                    st.error("No se pudo cargar la base de conocimiento.")


        # --- Área Principal de Chat ---
        st.header("Chatea con tus PDFs usando Google Gemini 💬")
        st.write("Sube tus documentos PDF en la barra lateral, haz clic en 'Procesar Documentos' y luego haz tus preguntas aquí.")

        # Mostrar mensajes existentes (usando el placeholder)
        with st.session_state.messages_placeholder.container():
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if isinstance(message, HumanMessage):
                        with st.chat_message(name="User", avatar="💃"):
                            st.write(message.content)
                    elif isinstance(message, AIMessage):
                        with st.chat_message(name="J.A.A.F.A.R.", avatar="🤖"):
                            st.write(message.content)

    # Input del usuario
    user_question = st.chat_input("Haz una pregunta sobre tus documentos...")

    if user_question:
        if not st.session_state.pdf_processed or st.session_state.conversation is None:
            st.warning('Primero guarda o carga los documentos.')
        else:
            handle_user_input(user_question)


if __name__ == '__main__':
    # Asegurarse de que la clave API esté disponible antes de llamar a main()
    if GOOGLE_API_KEY:
        main()
    else:
        # El mensaje de error ya se mostró al inicio si la clave no existe
        logger.error("Ejecución detenida porque GOOGLE_API_KEY no está configurada.")
