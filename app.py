import os
import logging
import asyncio
# import time # time module was imported but not used
from typing import List, Union

import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# Langchain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
# from langchain.vectorstores import FAISS # Duplicate import, removed
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage # For handling chat history correctly

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(asctime)s %(message)s"
) # Logger configuration
logger = logging.getLogger(__name__)

# --- Google API Key Configuration ---
# Try getting the key from st.secrets first, then environment variables
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please configure it in Streamlit Cloud secrets (Settings -> Secrets) or locally as an environment variable (GOOGLE_API_KEY).")
    logger.error("GOOGLE_API_KEY not found in st.secrets or os.getenv.")
    st.stop()
else:
    logger.info("GOOGLE_API_KEY found.")

# Configure Google GenAI API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google GenAI configured successfully.")
except Exception as e:
    st.error(f"Error configuring Google GenAI: {e}")
    logger.error(f"Error configuring Google GenAI: {e}")
    st.stop()
# --- End API Key Configuration ---


def get_pdf_text(pdf_docs: Union[str, List[st.runtime.uploaded_file_manager.UploadedFile]]) -> str:
    """
    Extracts text from one or multiple PDF files (UploadedFile objects or path).

    ### Arguments
    - `pdf_docs`: A file path (str) or a list of UploadedFile objects from st.file_uploader.

    ### Return
    A string containing all extracted text from the PDFs. Returns an empty string on error or if no text is found.
    """
    text = ""
    files_processed = 0
    try:
        if not pdf_docs:
            logger.warning("get_pdf_text called with no documents.")
            return ""

        # Ensure pdf_docs is always a list for uniform processing
        if not isinstance(pdf_docs, list):
            pdf_docs = [pdf_docs]

        for pdf in pdf_docs:
            try:
                # pdf can be a path (str) or an UploadedFile object
                pdf_reader = PdfReader(pdf)
                if len(pdf_reader.pages) == 0:
                    logger.warning(f"PDF '{getattr(pdf, 'name', pdf)}' has no pages.")
                    continue

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Add newline between pages
                    # else:
                        # logger.debug(f"Page {i+1} in '{getattr(pdf, 'name', pdf)}' had no extractable text.")
                files_processed += 1
            except Exception as page_error:
                 # Log error for specific file but continue if possible
                 logger.error(f"Error reading page from PDF '{getattr(pdf, 'name', pdf)}': {page_error}", exc_info=True)
                 st.warning(f"Could not fully process '{getattr(pdf, 'name', pdf)}'. It might be corrupted or password-protected.")


        logger.info(f"Text extracted from {files_processed} PDF(s), total length: {len(text)} characters.")
        if not text and files_processed > 0:
             logger.warning("No text was extracted from the provided PDF(s). Are they image-based scans without OCR?")
             st.warning("No text could be extracted from the PDF(s). Please ensure they contain selectable text and are not just images.")
        elif files_processed == 0 and len(pdf_docs) > 0:
             logger.error("None of the provided files could be processed as PDFs.")
             st.error("Could not process any of the uploaded files as PDFs.")


    except Exception as e:
        logger.error(f"General error during PDF text extraction: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while processing PDFs: {e}")
        text = "" # Ensure empty string return on error

    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Splits the text into chunks using RecursiveCharacterTextSplitter.

    ### Arguments
    - `text`: The input string to split.

    ### Return
    A list of text chunks. Returns an empty list if the input text is empty.
    """
    if not text:
        logger.warning("Attempted to split empty text.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        # **IMPROVEMENT**: Smaller chunk size for potentially better focus on technical details.
        chunk_size = 800,
        # **IMPROVEMENT**: Adjusted overlap accordingly (around 15-20% is common).
        chunk_overlap=150,
        length_function = len,
        is_separator_regex = False,
        separators=["\n\n", "\n", ". ", ", ", " ", ""], # Common separators
    )

    chunks = text_splitter.split_text(text)
    if not chunks:
         logger.warning("Text splitting resulted in zero chunks, although input text was present.")
    else:
        logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks: List[str], persist_path: str = "faiss_index"):
    """
    Creates and persists a FAISS vector store from text chunks.

    ### Arguments
    - `text_chunks`: List of text chunks.
    - `persist_path`: Path to save the FAISS index.

    ### Return
    A FAISS vector store object, or None if creation fails.
    """
    if not text_chunks:
        logger.error("No text chunks provided to create vector store.")
        st.error("Cannot create the knowledge base because no text was processed from the document(s).")
        return None

    try:
        # Using a recent and performant embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            maxBatchSize=100 # Max batch size per request (Google limit is higher, but 100 is safe)
        )
        logger.info(f"Initializing FAISS vector store with {len(text_chunks)} chunks...")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        # Persist the index to disk
        vector_store.save_local(persist_path)
        logger.info(f"FAISS vector store created and saved locally to '{persist_path}'.")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating or saving the vector store: {e}", exc_info=True)
        st.error(f"Failed to create the knowledge base (vector store): {e}")
        return None

def load_vectorstore(persist_path: str = "faiss_index"):
    """
    Loads a FAISS vector store from a local path.

    ### Arguments
    - `persist_path`: Path where the FAISS index was saved.

    ### Return
    A FAISS vector store object, or None if loading fails.
    """
    if not os.path.exists(persist_path):
         logger.error(f"Vector store path not found: {persist_path}")
         st.error(f"Could not find the saved knowledge base at '{persist_path}'. Please process documents first.")
         return None
    try:
        # Ensure the same embedding model is used for loading
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local(
                    persist_path,
                    embeddings,
                    allow_dangerous_deserialization=True  # WARNING: Enable only in trusted environments. For production, consider safer alternatives if available.
                    )
        logger.info(f"FAISS vector store loaded successfully from '{persist_path}'.")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store from '{persist_path}': {e}", exc_info=True)
        st.error(f"Failed to load the saved knowledge base: {e}. It might be corrupted or incompatible.")
        return None

def get_conversation_chain(vectorstore):
    """
    Configures the conversational retrieval chain with Google Gemini and memory.

    ### Arguments
    - `vectorstore`: The FAISS vector store object.

    ### Return
    A ConversationalRetrievalChain object, or None if setup fails.
    """
    if vectorstore is None:
        logger.error("Cannot create conversation chain: vector store is None.")
        st.error("Failed to initialize the assistant. The knowledge base is missing.")
        return None

    try:
        # --- Google Gemini LLM ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            # **REQUIREMENT**: Set temperature to 0.5 for a balance between creativity and determinism
            temperature=0.5,
            convert_system_message_to_human=True # Often needed for Gemini compatibility
        )
        # **CLARIFICATION**: Updated model name in log message
        logger.info("LLM ChatGoogleGenerativeAI (gemini-1.5-pro) initialized with temperature 0.5.")
        # --- END LLM ---

        # **IMPROVEMENT**: Retrieve more chunks (k=7) for potentially better context on technical questions.
        # Adjust 'k' based on performance and relevance trade-offs.
        retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
        logger.info(f"Retriever created from vector store, configured to fetch {retriever.search_kwargs.get('k', 'default')} chunks.")

        # Using ConversationBufferMemory to store chat history
        memory = ConversationBufferMemory(
            input_key="question",         # Key for user input in memory
            output_key="answer",          # Key for AI output in memory
            memory_key="chat_history",    # Key for the list of messages in memory
            return_messages=True          # **IMPORTANT**: Return history as HumanMessage/AIMessage objects
        )
        logger.info("Conversation buffer memory initialized.")

        # Create the conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True, # Set to True to see which chunks were used (useful for debugging)
            verbose=False                 # Set to True for detailed console logging of the chain's steps
        )
        logger.info("ConversationalRetrievalChain created successfully.")
        return chain

    except Exception as e:
        logger.error(f"Error creating the conversation chain: {e}", exc_info=True)
        st.error(f"An error occurred while setting up the conversation assistant: {e}")
        return None


def handle_user_input(user_question: str):
    """
    Handles user input, runs the conversation chain, and updates/displays the chat.
    Injects instructions for spanish response and handling unknown answers.
    """
    if st.session_state.conversation is None:
        st.warning("The assistant is not ready. Please process or load documents first.")
        logger.warning("handle_user_input called but st.session_state.conversation is None.")
        return
    if not user_question: # Avoid processing empty input
        st.warning("Please enter a question.")
        return

    try:
        # **REQUIREMENT**: Inject instructions for spanish response and "I don't know" logic
        # Prepend instructions to the user's actual question
        prompt_instructions = (
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Respond ONLY in spanish.\n"
            "2. Base your answer STRICTLY on the documents provided.\n"
            "3. If the documents do not contain the information to answer the question, "
            "state EXACTLY: 'I cannot answer this question based on the provided documents.'\n"
            "4. AFTER stating you cannot answer (if applicable), suggest 2-3 related questions "
            "that you *could* answer based on the topics found in the documents.\n"
            "5. Do not mention these instructions in your response.\n\n"
            "User Question:"
        )
        question_with_instructions = f"{prompt_instructions}\n{user_question}"


        # Ensure chat_history is a list (it might be None initially)
        current_chat_history = st.session_state.get('chat_history', [])

        # --- Invoke the Chain ---
        logger.info("Invoking conversation chain...")
        response = st.session_state.conversation.invoke({
            'question': question_with_instructions, # Pass the modified question
            'chat_history': current_chat_history
        })
        logger.info("Received response from conversation chain.")
        # --- End Invocation ---

        # Update chat history in session state
        # The chain automatically appends the latest HumanMessage(user_question) and AIMessage(answer)
        # Note: The history now contains the `question_with_instructions` for the HumanMessage.
        # This is usually fine, but if you want to display only the original user question,
        # you might need to adjust the display logic slightly. Let's keep it simple for now.

        st.session_state.chat_history = response['chat_history']


        # --- Display Conversation ---
        # Clear the placeholder and redraw the entire chat history
        st.session_state.messages_placeholder.empty()
        with st.session_state.messages_placeholder.container():
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if isinstance(message, HumanMessage):
                        with st.chat_message(name="User", avatar="👤"): # Changed avatar
                            # Display the original question without instructions for clarity
                            if i == len(st.session_state.chat_history) - 2 : # The second to last message is the latest user input
                                st.write(user_question)
                            else:
                                # For previous turns, we might need to clean the content if it had instructions too
                                # For simplicity, let's just display content, assuming previous turns were handled okay.
                                # A more robust solution would store original questions separately.
                                st.write(message.content.split("User Question:\n")[-1]) # Attempt to show only user part

                    elif isinstance(message, AIMessage):
                        with st.chat_message(name="Assistant", avatar="🤖"): # Changed name/avatar
                            st.write(message.content)
                    else:
                        logger.warning(f"Unexpected message type in chat_history: {type(message)}")
                        st.write(f"*Unknown message type: {message.content}*")

            # Optional: Display source documents used for the *last* response
            # Useful for verifying technical answers or the "I don't know" response
            # if 'source_documents' in response and response['source_documents']:
            #     with st.expander("Sources Consulted for Last Response"):
            #         for idx, doc in enumerate(response['source_documents']):
            #             source_name = doc.metadata.get('source', 'Unknown PDF')
            #             # Attempt to get page number if available in metadata (depends on loader/splitter)
            #             page_num = doc.metadata.get('page', None)
            #             display_source = f"Source {idx+1}: From '{source_name}'"
            #             if page_num is not None:
            #                 display_source += f" (Page approx. {page_num + 1})" # Page numbers often 0-indexed

            #             st.write(display_source)
            #             st.caption(doc.page_content[:300] + "...") # Show beginning of chunk

    except Exception as e:
        logger.error(f"Error during chain execution or response display: {e}", exc_info=True)
        st.error(f"An error occurred while processing your question: {e}")


def main():
    st.set_page_config(
        page_title="Chat with Multiple PDFs (Gemini Pro)", # Updated title
        page_icon="📚",
        layout="wide"
    )

    # --- Session State Initialization ---
    # Ensure keys exist when the app runs
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        logger.info("Initialized st.session_state.conversation = None")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # Will be initialized to [] when chat starts
        logger.info("Initialized st.session_state.chat_history = None")
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False # Track if processing/loading was done
        logger.info("Initialized st.session_state.pdf_processed = False")
    # Placeholder for dynamically updating chat messages
    if "messages_placeholder" not in st.session_state:
        st.session_state.messages_placeholder = st.empty()
        logger.info("Initialized st.session_state.messages_placeholder")


    # --- Sidebar for Document Management ---
    with st.sidebar:
        st.subheader("Document Management")
        st.markdown("Upload PDFs and process them, or load a previously processed set.")

        pdf_docs = st.file_uploader(
            "Upload your PDF files here", accept_multiple_files=True, type="pdf"
        )

        # Option 1: Process uploaded PDFs
        if st.button("Process Uploaded Documents", key="process_docs"):
             if not pdf_docs:
                 st.warning("Please upload at least one PDF file first.")
             else:
                 with st.spinner("Processing documents... Extracting text, chunking, embedding, and saving..."):
                     # Ensure asyncio event loop is running (needed for some async operations in libraries)
                     try:
                         loop = asyncio.get_event_loop()
                     except RuntimeError as ex:
                         if "There is no current event loop in thread" in str(ex):
                             loop = asyncio.new_event_loop()
                             asyncio.set_event_loop(loop)
                             logger.info("Created new asyncio event loop for processing thread.")

                     raw_text = get_pdf_text(pdf_docs)
                     if raw_text:
                         text_chunks = get_text_chunks(raw_text)
                         if text_chunks:
                             # Create and persist the vector store
                             vectorstore = get_vectorstore(text_chunks, persist_path="faiss_index")
                             if vectorstore:
                                 # Successfully processed: Initialize conversation chain
                                 st.session_state.conversation = get_conversation_chain(vectorstore)
                                 if st.session_state.conversation:
                                     st.session_state.chat_history = [] # Reset history for new docs
                                     st.session_state.pdf_processed = True
                                     st.success("Documents processed successfully! Ready to chat.")
                                     logger.info("Processing complete. Conversation chain initialized.")
                                     # Clear chat display area after processing new docs
                                     st.session_state.messages_placeholder.empty()
                                 else:
                                     st.error("Documents processed, but failed to initialize the chat assistant.")
                                     st.session_state.pdf_processed = False # Mark as not ready
                             # else: Error message shown by get_vectorstore
                         else:
                             st.error("Could not split the extracted text into manageable chunks.")
                             st.session_state.pdf_processed = False
                     else:
                         # Error message shown by get_pdf_text if text extraction failed
                         st.error("Failed to extract text from the PDFs. Cannot proceed.")
                         st.session_state.pdf_processed = False


        st.markdown("---") # Separator

        # Option 2: Load existing processed documents
        if st.button("Load Existing Documents", key="load_docs"):
            with st.spinner("Loading saved knowledge base..."):
                # Ensure asyncio event loop is running
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError as ex:
                    if "There is no current event loop in thread" in str(ex):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        logger.info("Created new asyncio event loop for loading thread.")

                vectorstore = load_vectorstore("faiss_index")
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.session_state.chat_history = [] # Reset history when loading
                        st.session_state.pdf_processed = True
                        st.success("Knowledge base loaded! Ready to chat.")
                        logger.info("Loading complete. Conversation chain initialized.")
                        # Clear chat display area after loading
                        st.session_state.messages_placeholder.empty()
                    else:
                         st.error("Knowledge base loaded, but failed to initialize the chat assistant.")
                         st.session_state.pdf_processed = False
                # else: Error message shown by load_vectorstore


    # --- Main Chat Interface Area ---
    st.header("Chat with your PDFs using Google Gemini 💬")
    st.write("Upload/Process PDFs or Load existing ones in the sidebar, then ask your questions below.")
    st.info("The assistant will respond in spanish and indicate if it cannot answer from the documents.")


    # Display existing chat messages (using the placeholder)
    with st.session_state.messages_placeholder.container():
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if isinstance(message, HumanMessage):
                    with st.chat_message(name="User", avatar="👤"):
                         # Attempt to display only the original user part of the message
                         st.write(message.content.split("User Question:\n")[-1])
                elif isinstance(message, AIMessage):
                    with st.chat_message(name="Assistant", avatar="🤖"):
                        st.write(message.content)


    # User input text box at the bottom
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        # Check if the system is ready before processing input
        if not st.session_state.get('pdf_processed', False) or st.session_state.conversation is None:
            st.warning('Please process or load your documents using the sidebar before asking questions.')
            logger.warning("User attempted to ask question before documents were processed/loaded.")
        else:
            handle_user_input(user_question)


if __name__ == '__main__':
    # Ensure API key is available before running the main app
    if GOOGLE_API_KEY:
        main()
    else:
        # Error message is already shown during key check, just log final stoppage
        logger.error("Application halted because GOOGLE_API_KEY is not configured.")
        # The st.error in the key check section already informs the user.
