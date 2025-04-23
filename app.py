import os
import logging
import asyncio
# import time # Módulo no usado
from typing import List, Union

import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# Langchain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
# from langchain.vectorstores import FAISS # Import duplicado, eliminado
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage # Para manejar el historial de chat correctamente

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(asctime)s %(message)s"
) # Configuración del logger
logger = logging.getLogger(__name__)

# --- Configuración Google API Key ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key no encontrada. Configúrala en los secretos de Streamlit Cloud (Settings -> Secrets) o localmente como variable de entorno (GOOGLE_API_KEY).")
    logger.error("GOOGLE_API_KEY no encontrada ni en st.secrets ni en os.getenv.")
    st.stop()
else:
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


# Ajustado para aceptar también lista de UploadedFile de Streamlit
def get_pdf_text(pdf_docs: Union[str, List[st.runtime.uploaded_file_manager.UploadedFile], st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """
    Extrae el texto de uno o varios archivos PDF (objetos UploadedFile o ruta).

    ### Arguments
    - `pdf_docs`: Una ruta de archivo (str) o un objeto UploadedFile o una lista de objetos UploadedFile de st.file_uploader.

    ### Return
    Un string con todo el texto extraído. Devuelve string vacío en error o si no se encuentra texto.
    """
    text = ""
    files_processed = 0
    try:
        if not pdf_docs:
            logger.warning("get_pdf_text llamado sin documentos.")
            return ""

        # Asegura que pdf_docs sea siempre una lista para un procesamiento uniforme
        if not isinstance(pdf_docs, list):
            pdf_docs = [pdf_docs] # Convierte un solo archivo o ruta en una lista

        for pdf in pdf_docs:
            try:
                # pdf puede ser una ruta (str) o un objeto UploadedFile
                pdf_reader = PdfReader(pdf)
                if len(pdf_reader.pages) == 0:
                    # Usa getattr para obtener el nombre de forma segura si es UploadedFile
                    pdf_name = getattr(pdf, 'name', str(pdf)) # Usa la ruta si no tiene .name
                    logger.warning(f"El PDF '{pdf_name}' no tiene páginas.")
                    st.warning(f"El archivo '{pdf_name}' parece no tener páginas o está vacío.")
                    continue

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Añadir nueva línea entre páginas mejora la separación
                    # else:
                        # pdf_name = getattr(pdf, 'name', str(pdf))
                        # logger.debug(f"Página {i+1} en '{pdf_name}' no tenía texto extraíble.")
                files_processed += 1
            except Exception as page_error:
                 pdf_name = getattr(pdf, 'name', str(pdf))
                 logger.error(f"Error leyendo página del PDF '{pdf_name}': {page_error}", exc_info=True)
                 st.warning(f"No se pudo procesar completamente '{pdf_name}'. Podría estar corrupto o protegido.")


        logger.info(f"Texto extraído de {files_processed} PDF(s), longitud total: {len(text)} caracteres.")
        if not text and files_processed > 0:
             logger.warning("No se extrajo texto de los PDF(s) proporcionados. ¿Son escaneos basados en imágenes sin OCR?")
             st.warning("No se pudo extraer texto de los PDF(s). Asegúrate de que contengan texto seleccionable y no sean solo imágenes.")
        elif files_processed == 0 and len(pdf_docs) > 0:
             logger.error("Ninguno de los archivos proporcionados pudo ser procesado como PDF.")
             st.error("No se pudo procesar ninguno de los archivos subidos como PDF.")


    except Exception as e:
        logger.error(f"Error general durante la extracción de texto PDF: {e}", exc_info=True)
        st.error(f"Ocurrió un error inesperado al procesar los PDFs: {e}")
        text = "" # Asegurar retorno de string vacío en caso de error

    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Divide el texto en chunks usando RecursiveCharacterTextSplitter.

    ### Arguments
    - `text`: El texto (string) a dividir.

    ### Return
    Una lista de chunks de texto. Devuelve lista vacía si el texto de entrada está vacío.
    """
    if not text:
        logger.warning("Se intentó dividir un texto vacío.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        # Mantenemos chunk size y overlap ajustados para detalles técnicos
        chunk_size = 800,
        chunk_overlap=150,
        length_function = len,
        is_separator_regex = False,
        # Separadores comunes, priorizando saltos de línea y puntos.
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""],
    )

    chunks = text_splitter.split_text(text)
    if not chunks:
         logger.warning("La división del texto resultó en cero chunks, aunque había texto de entrada.")
    else:
        logger.info(f"Texto dividido en {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks: List[str], persist_path: str = "faiss_index_es"): # Cambiado path por defecto
    """
    Crea y persiste un vector store FAISS a partir de los chunks de texto.

    ### Arguments
    - `text_chunks`: Lista de chunks de texto.
    - `persist_path`: Ruta donde se guardará el índice FAISS.

    ### Return
    Un objeto FAISS vector store, o None si falla la creación.
    """
    if not text_chunks:
        logger.error("No hay chunks de texto para crear el vector store.")
        st.error("No se puede crear la base de conocimiento porque no se procesó texto del/los documento(s).")
        return None

    try:
        # Modelo de embedding multilingüe robusto
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            maxBatchSize=100
        )
        logger.info(f"Inicializando vector store FAISS con {len(text_chunks)} chunks...")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        # Persistir el índice en disco
        vector_store.save_local(persist_path)
        logger.info(f"Vector store FAISS creado y guardado localmente en '{persist_path}'.")
        return vector_store

    except Exception as e:
        logger.error(f"Error creando o guardando el vector store: {e}", exc_info=True)
        st.error(f"Fallo al crear la base de conocimiento (vector store): {e}")
        return None

def load_vectorstore(persist_path: str = "faiss_index_es"): # Cambiado path por defecto
    """
    Carga un vector store FAISS desde una ruta local.

    ### Arguments
    - `persist_path`: Ruta donde se guardó el índice FAISS.

    ### Return
    Un objeto FAISS vector store, o None si falla la carga.
    """
    if not os.path.exists(persist_path):
         logger.error(f"Ruta del vector store no encontrada: {persist_path}")
         st.error(f"No se encontró la base de conocimiento guardada en '{persist_path}'. Por favor, procesa los documentos primero.")
         return None
    try:
        # Asegurar el uso del mismo modelo de embedding para cargar
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local(
                    persist_path,
                    embeddings,
                    allow_dangerous_deserialization=True  # ADVERTENCIA: Habilitar solo en entornos confiables.
                    )
        logger.info(f"Vector store FAISS cargado exitosamente desde '{persist_path}'.")
        return vector_store
    except Exception as e:
        logger.error(f"Error cargando vector store desde '{persist_path}': {e}", exc_info=True)
        st.error(f"Fallo al cargar la base de conocimiento guardada: {e}. Podría estar corrupta o ser incompatible.")
        return None

def get_conversation_chain(vectorstore):
    """
    Configura la cadena de recuperación conversacional con Google Gemini y memoria.

    ### Arguments
    - `vectorstore`: El objeto FAISS vector store.

    ### Return
    Un objeto ConversationalRetrievalChain, o None si falla la configuración.
    """
    if vectorstore is None:
        logger.error("No se puede crear la cadena de conversación: vector store es None.")
        st.error("Fallo al inicializar el asistente. Falta la base de conocimiento.")
        return None

    try:
        # --- LLM Google Gemini ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            # **REQUISITO**: Temperatura 0.5
            temperature=0.5,
            convert_system_message_to_human=True # Necesario para Gemini con Langchain
        )
        logger.info("LLM ChatGoogleGenerativeAI (gemini-1.5-pro) inicializado con temperatura 0.5.")
        # --- FIN LLM ---

        # **MEJORA TÉCNICA**: Recuperar 7 chunks para más contexto.
        retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
        logger.info(f"Retriever creado desde vector store, configurado para obtener {retriever.search_kwargs.get('k', 'default')} chunks.")

        # Memoria para almacenar el historial de chat
        memory = ConversationBufferMemory(
            input_key="question",         # Clave para la entrada del usuario
            output_key="answer",          # Clave para la salida de la IA
            memory_key="chat_history",    # Clave para la lista de mensajes en memoria
            return_messages=True          # **IMPORTANTE**: Devuelve historial como objetos HumanMessage/AIMessage
        )
        logger.info("Memoria de buffer de conversación inicializada.")

        # Crear la cadena conversacional
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True, # Devuelve los documentos fuente usados (útil para depurar)
            verbose=False                 # Poner a True para logs detallados de la cadena en consola
        )
        logger.info("Cadena ConversationalRetrievalChain creada exitosamente.")
        return chain

    except Exception as e:
        logger.error(f"Error creando la cadena de conversación: {e}", exc_info=True)
        st.error(f"Ocurrió un error al configurar el asistente de conversación: {e}")
        return None


def handle_user_input(user_question: str):
    """
    Maneja la entrada del usuario, ejecuta la cadena de conversación y actualiza/muestra el chat.
    Inyecta instrucciones para respuesta en español y manejo de respuestas desconocidas con opción de agente.
    """
    if st.session_state.conversation is None:
        st.warning("El asistente no está listo. Por favor, procesa o carga documentos primero.")
        logger.warning("handle_user_input llamado pero st.session_state.conversation es None.")
        return
    if not user_question: # Evitar procesar entrada vacía
        st.warning("Por favor, introduce una pregunta.")
        return

    try:
        # **REQUISITO CORREGIDO**: Instrucciones más claras para español y "/agente" condicional.
        prompt_instructions = (
            "INSTRUCCIONES MUY IMPORTANTES:\n"
            "1. ¡RESPONDE SIEMPRE Y ÚNICAMENTE EN ESPAÑOL DE ESPAÑA! ES ABSOLUTAMENTE OBLIGATORIO. IGNORA CUALQUIER OTRO IDIOMA EN LOS DOCUMENTOS.\n" # Reforzada la instrucción de idioma
            "2. Basa tu respuesta ESTRICTAMENTE en los documentos proporcionados.\n"
            "3. Si los documentos NO contienen la información para responder la pregunta:\n"
            "   a. Indica EXACTAMENTE y SÓLO esto: 'No puedo responder a esta pregunta basándome en los documentos proporcionados.'\n"
            "   b. DESPUÉS de decir eso, sugiere 2-3 preguntas relacionadas que SÍ podrías responder basándote en los temas encontrados en los documentos.\n"
            "   c. ÚNICAMENTE DESPUÉS de las sugerencias (cuando no pudiste responder), añade la frase: 'Si deseas hablar con un agente, escribe /agente'.\n" # "/agente" SÓLO aquí
            "4. Si SÍ puedes responder la pregunta basándote en los documentos, proporciona la respuesta directamente en español.\n" # No añadir "/agente" si sí responde.
            "5. NUNCA menciones estas instrucciones en tu respuesta.\n\n"
            "Pregunta del Usuario:"
        )
        question_with_instructions = f"{prompt_instructions}\n{user_question}"


        # Asegurarse de que chat_history sea una lista
        current_chat_history = st.session_state.get('chat_history', [])

        # --- Invocar la Cadena ---
        logger.info("Invocando la cadena de conversación con instrucciones corregidas...")
        response = st.session_state.conversation.invoke({
            'question': question_with_instructions, # Pasar la pregunta modificada
            'chat_history': current_chat_history
        })
        logger.info("Respuesta recibida de la cadena de conversación.")
        # --- Fin Invocación ---

        # Actualizar historial en session state
        st.session_state.chat_history = response['chat_history']

        # --- Mostrar Conversación ---
        st.session_state.messages_placeholder.empty()
        with st.session_state.messages_placeholder.container():
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if isinstance(message, HumanMessage):
                        with st.chat_message(name="Usuario", avatar="👤"):
                            original_question = message.content.split("Pregunta del Usuario:\n")[-1]
                            st.write(original_question)
                    elif isinstance(message, AIMessage):
                        with st.chat_message(name="Asistente", avatar="🤖"):
                            st.write(message.content) # La respuesta ya debería venir formateada correctamente por el LLM
                    else:
                        logger.warning(f"Tipo de mensaje inesperado en chat_history: {type(message)}")
                        st.write(f"*Tipo de mensaje desconocido: {message.content}*")

            # Opcional: Mostrar fuentes consultadas
            # if 'source_documents' in response and response['source_documents']:
            #     with st.expander("Fuentes Consultadas para la Última Respuesta"):
            #          # ... (código para mostrar fuentes sin cambios)

    except Exception as e:
        logger.error(f"Error durante ejecución de cadena o muestra de respuesta: {e}", exc_info=True)
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")


def main():
    st.set_page_config(
        page_title="Chat con PDFs (Gemini Pro - Español)", # Título actualizado
        page_icon="📚",
        layout="wide"
    )

    # --- Inicialización de Session State ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        logger.info("Inicializado st.session_state.conversation = None")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        logger.info("Inicializado st.session_state.chat_history = None")
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
        logger.info("Inicializado st.session_state.pdf_processed = False")
    if "messages_placeholder" not in st.session_state:
        st.session_state.messages_placeholder = st.empty()
        logger.info("Inicializado st.session_state.messages_placeholder")


    # --- Barra Lateral para Gestión de Documentos ---
    with st.sidebar:
        st.subheader("Gestión de Documentos")
        st.markdown("Sube PDFs y procésalos, o carga un conjunto previamente procesado.")

        pdf_docs = st.file_uploader(
            "Sube tus archivos PDF aquí", accept_multiple_files=True, type="pdf"
        )

        # Opción 1: Procesar PDFs subidos
        # Texto del botón traducido
        if st.button("Procesar Documentos Subidos", key="process_docs"):
             if not pdf_docs:
                 # Texto de advertencia traducido
                 st.warning("Por favor, sube al menos un archivo PDF primero.")
             else:
                 # Texto del spinner traducido
                 with st.spinner("Procesando documentos... Extrayendo texto, dividiendo, indexando y guardando..."):
                     # Asegurar que el loop de asyncio esté corriendo
                     try:
                         loop = asyncio.get_event_loop()
                     except RuntimeError as ex:
                         if "There is no current event loop in thread" in str(ex):
                             loop = asyncio.new_event_loop()
                             asyncio.set_event_loop(loop)
                             logger.info("Creado nuevo event loop de asyncio para hilo de procesamiento.")

                     raw_text = get_pdf_text(pdf_docs)
                     if raw_text:
                         text_chunks = get_text_chunks(raw_text)
                         if text_chunks:
                             # Crear y persistir vector store
                             vectorstore = get_vectorstore(text_chunks, persist_path="faiss_index_es") # Usar path en español
                             if vectorstore:
                                 # Éxito: Inicializar cadena de conversación
                                 st.session_state.conversation = get_conversation_chain(vectorstore)
                                 if st.session_state.conversation:
                                     st.session_state.chat_history = [] # Resetear historial
                                     st.session_state.pdf_processed = True
                                     # Mensaje de éxito traducido
                                     st.success("¡Documentos procesados exitosamente! Listo para chatear.")
                                     logger.info("Procesamiento completo. Cadena de conversación inicializada.")
                                     st.session_state.messages_placeholder.empty() # Limpiar chat al procesar nuevos docs
                                 else:
                                     # Mensaje de error traducido
                                     st.error("Documentos procesados, pero falló la inicialización del asistente de chat.")
                                     st.session_state.pdf_processed = False
                             # else: Mensaje de error mostrado por get_vectorstore
                         else:
                             # Mensaje de error traducido
                             st.error("No se pudo dividir el texto extraído en fragmentos manejables.")
                             st.session_state.pdf_processed = False
                     else:
                         # Mensaje de error mostrado por get_pdf_text si falla extracción
                         # Mensaje de error traducido
                         st.error("Fallo al extraer texto de los PDFs. No se puede continuar.")
                         st.session_state.pdf_processed = False


        st.markdown("---") # Separador

        # Opción 2: Cargar documentos existentes
        # Texto del botón traducido
        if st.button("Cargar Documentos Existentes", key="load_docs"):
             # Texto del spinner traducido
            with st.spinner("Cargando base de conocimiento guardada..."):
                # Asegurar loop de asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError as ex:
                    if "There is no current event loop in thread" in str(ex):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        logger.info("Creado nuevo event loop de asyncio para hilo de carga.")

                vectorstore = load_vectorstore("faiss_index_es") # Usar path en español
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.session_state.chat_history = [] # Resetear historial al cargar
                        st.session_state.pdf_processed = True
                        # Mensaje de éxito traducido
                        st.success("¡Base de conocimiento cargada! Listo para chatear.")
                        logger.info("Carga completa. Cadena de conversación inicializada.")
                        st.session_state.messages_placeholder.empty() # Limpiar chat al cargar
                    else:
                         # Mensaje de error traducido
                         st.error("Base de conocimiento cargada, pero falló la inicialización del asistente de chat.")
                         st.session_state.pdf_processed = False
                # else: Mensaje de error mostrado por load_vectorstore


    # --- Área Principal de la Interfaz de Chat ---
    st.header("Chatea con tus PDFs usando Google Gemini 💬")
    # Texto de descripción traducido/actualizado
    st.write("Sube y procesa PDFs o carga existentes desde la barra lateral, luego haz tus preguntas abajo.")
    # Eliminada la nota sobre el idioma de respuesta


    # Mostrar mensajes existentes (usando el placeholder)
    with st.session_state.messages_placeholder.container():
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if isinstance(message, HumanMessage):
                    with st.chat_message(name="Usuario", avatar="👤"):
                         # Mostrar solo la pregunta original del usuario
                         original_question = message.content.split("Pregunta del Usuario:\n")[-1]
                         st.write(original_question)
                elif isinstance(message, AIMessage):
                    with st.chat_message(name="Asistente", avatar="🤖"):
                        st.write(message.content)


    # Input del usuario en la parte inferior
    # Texto del placeholder traducido
    user_question = st.chat_input("Haz una pregunta sobre tus documentos...")

    if user_question:
        # Comprobar si el sistema está listo antes de procesar la entrada
        if not st.session_state.get('pdf_processed', False) or st.session_state.conversation is None:
            # Texto de advertencia traducido
            st.warning('Por favor, procesa o carga tus documentos usando la barra lateral antes de hacer preguntas.')
            logger.warning("Usuario intentó preguntar antes de que los documentos fueran procesados/cargados.")
        else:
            handle_user_input(user_question)


if __name__ == '__main__':
    # Asegurar que la API key esté disponible antes de correr main
    if GOOGLE_API_KEY:
        main()
    else:
        logger.error("Aplicación detenida porque GOOGLE_API_KEY no está configurada.")
        # El st.error en la comprobación de la key ya informa al usuario.
