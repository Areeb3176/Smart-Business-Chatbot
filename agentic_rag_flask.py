# agentic_rag_flask.py
import logging
import os
import re
import sys
import warnings
from typing import List, Union

import requests
from bs4 import BeautifulSoup
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_community.document_loaders import (UnstructuredMarkdownLoader, WebBaseLoader)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from tavily import TavilyClient
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# Global dictionary to store RAG components
rag_components = {}

# Set up environment variables from .env or system environment
os.environ["USER_AGENT"] = os.environ.get("USER_AGENT", "AgenticRAG/1.0_Flask_Default")

# API Key Checks (logger warnings, clients will error if missing)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

logger = logging.getLogger(__name__)

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found in environment variables. Web search will fail.")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables. LLM calls will fail.")

# Resolve or suppress warnings
#logger.basicConfig(level=logger.ERROR, force=True) # Default to ERROR
# logger.getLogger("sagemaker").setLevel(logger.CRITICAL) # sagemaker not used
# logger.getLogger("sagemaker.config").setLevel(logger.CRITICAL) # sagemaker not used
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module='torch.indexed_ops') # Common sentence-transformer warning
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')
#logger.basicConfig(level=logger.INFO) # Keep INFO for debugging graph steps and general info
 # Use this logger instance throughout this file, e.g. logger.info(...)
# Define paths and parameters
DATA_FOLDER = 'data'
# Corrected typo as per original provided code's correction
persist_directory_huggingface = '/app/persistent_db'
collection_name = 'rag'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

AUTHORITATIVE_PAKISTAN_BUSINESS_DOMAINS = [
    "secp.gov.pk", "fbr.gov.pk", "invest.gov.pk", "boi.gov.pk",
    "sbp.org.pk", "ipo.gov.pk", "psw.gov.pk", "smeda.org",
    "fpcci.org.pk", "kcci.com.pk", "lcci.com.pk",
]

def remove_tags(soup):
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        element.decompose()
    content = ""
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.get_text(strip=True)
        if element.name.startswith('h'):
            level = int(element.name[1])
            content += '#' * level + ' ' + text + '\n\n'
        elif element.name == 'p':
            content += text + '\n\n'
        elif element.name == 'li':
            content += '- ' + text + '\n'
    return content

def get_info(URLs):
    combined_info = ""
    for url in URLs:
        try:
            response = requests.get(url, headers={"User-Agent": os.environ["USER_AGENT"]})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                combined_info += "URL: " + url + ": " + remove_tags(soup) + "\n\n"
            else:
                combined_info += f"Failed to retrieve information from {url}\n\n"
        except Exception as e:
            combined_info += f"Error fetching URL {url}: {e}\n\n"
    return combined_info

def staticChunker(folder_path):
    docs = []
    logger.info(f"Creating chunks. CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    if not os.path.exists(folder_path):
        logger.warning(f"Data folder '{folder_path}' not found. Cannot create chunks.")
        return []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(folder_path, file_name)
            logger.info(f"Processing file: {file_path}")
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source_file"] = file_name
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunked_docs = text_splitter.split_documents(documents)
                docs.extend(chunked_docs)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
    return docs

def load_or_create_vs(persist_directory, embedding_model_instance):
    if not embedding_model_instance:
         logger.error("Embedding model not initialized before vector store access!")
         raise ValueError("Embedding model instance is required for vector store.")

    if os.path.exists(persist_directory):
        logger.info(f"Loading existing vector store from {persist_directory}...")
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model_instance,
                collection_name=collection_name
            )
            logger.info("Vector store loaded.")
        except Exception as e:
             logger.error(f"Error loading vector store from {persist_directory}: {e}")
             logger.warning("Attempting to create a new vector store.")
             vectorstore = create_new_vectorstore(persist_directory, embedding_model_instance)
    else:
        logger.info(f"Vector store not found at {persist_directory}. Creating a new one...")
        vectorstore = create_new_vectorstore(persist_directory, embedding_model_instance)
    return vectorstore

def create_new_vectorstore(persist_directory, embedding_model_instance):
    docs = staticChunker(DATA_FOLDER)
    if not docs:
        logger.error(f"No documents found or processed in '{DATA_FOLDER}'. Cannot create vector store.")
        raise ValueError("No documents to create vector store.")

    logger.info("Computing embeddings for new vector store...")
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model_instance,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        logger.info('Vector store created and persisted successfully!')
        return vectorstore
    except Exception as e:
        logger.error(f"Fatal Error: Failed to create vector store: {e}")
        raise

def initialize_llm(model_name, streaming=False): # answer_style removed, not used by ChatGroq init
    temperature = 0.0
    # `reasoning_format` seems specific to some models, check Groq docs if needed for all
    # For now, assume it's fine or ignored by models not supporting it.
    # if "deepseek-" in model_name:
    #     return ChatGroq(model=model_name, temperature=temperature, streaming=streaming, model_kwargs={"reasoning_format": "hidden"})
    return ChatGroq(model=model_name, temperature=temperature, streaming=streaming)


def initialize_embedding_model(selected_embedding_model):
    logger.info(f"Initializing embedding model: {selected_embedding_model}")
    try:
        return HuggingFaceEmbeddings(model_name=selected_embedding_model)
    except Exception as e:
        logger.error(f"Error initializing embedding model {selected_embedding_model}: {e}")
        logger.warning("Falling back to default sentence-transformers/all-MiniLM-L6-v2")
        try:
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as fallback_e:
            logger.error(f"CRITICAL: Failed to initialize even the fallback embedding model: {fallback_e}")
            raise

def initialize_grader_chain(grader_llm_instance, selected_grading_model_name):
    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    if not grader_llm_instance:
        logger.error("Grader LLM not initialized before creating grader chain!")
        raise ValueError("Grader LLM instance required.")

    try:
        structured_llm_grader = grader_llm_instance.with_structured_output(GradeDocuments)
    except Exception as e:
        logger.error(f"Error creating structured output for grader LLM ({selected_grading_model_name}): {e}")
        logger.warning("Using standard LLM call for grading might be less reliable. This path is not fully implemented here.")
        # If this is critical, we should raise an exception.
        # For now, let's assume it's a severe issue if structured output fails.
        raise ValueError(f"Failed to set up structured output for grader LLM: {e}")


    SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question about business in Pakistan.
    Follow these instructions for grading:
    - If the document contains keyword(s) or semantic meaning directly addressing or closely related to the user's question, considering the context of business, investment, or entrepreneurship in Pakistan, grade it as relevant.
    - Your grade should be **only** 'yes' or 'no' (lowercase) to indicate whether the document is relevant to the question or not."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", "Retrieved document:\n\n{documents}\n\nUser question:\n{question}"),
    ])
    return grade_prompt | structured_llm_grader

def load_document_summaries(file_path: str) -> str:
    """Loads document summaries from a text file."""
    logger.info(f"Attempting to load document summaries from: {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"Summary file not found at '{file_path}'. Routing will rely on general tool descriptions only.")
        return "No document summaries are available."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            summaries = f.read()
            logger.info("Successfully loaded document summaries.")
            return summaries
    except Exception as e:
        logger.error(f"Error reading summary file '{file_path}': {e}")
        return "Failed to load document summaries."

# --- Main Application Initialization ---
def initialize_rag_components(model_name, selected_embedding_model, selected_routing_model, selected_grading_model):
    """Initializes and stores all RAG components in the global `rag_components` dict."""
    global rag_components # Ensure we are modifying the global dict
    logger.info("Initializing RAG components...")
    try:
        # Load summaries first, as the router will need them
        rag_components['doc_summaries'] = load_document_summaries('summaries.txt')

        rag_components['embed_model'] = initialize_embedding_model(selected_embedding_model)
        persist_dir = persist_directory_huggingface # Using the corrected name
        rag_components['vectorstore'] = load_or_create_vs(persist_dir, rag_components['embed_model'])
        rag_components['retriever'] = rag_components['vectorstore'].as_retriever(search_kwargs={"k": 5}) # Default k=5

        # Initialize LLMs
        # Answer style is dynamic per request, not for LLM init. Streaming=True for main LLM for potential future use.
        rag_components['llm'] = initialize_llm(model_name, streaming=True)
        rag_components['router_llm'] = initialize_llm(selected_routing_model, streaming=False)
        rag_components['grader_llm'] = initialize_llm(selected_grading_model, streaming=False)
        
        rag_components['doc_grader'] = initialize_grader_chain(rag_components['grader_llm'], selected_grading_model)

        # Initialize Tavily Client
        if TAVILY_API_KEY:
            rag_components['tavily_client'] = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            rag_components['tavily_client'] = None
            logger.warning("Tavily client not initialized due to missing API key.")

        # Compile workflow (defined later in this file)
        if 'workflow' not in globals():
             logger.error("Workflow graph definition is missing!")
             raise NameError("Workflow graph definition (workflow) not found.")
        
        logger.info("Attempting workflow compilation...")
        rag_components['compiled_workflow'] = workflow.compile()
        logger.info(f"Workflow compiled. Type: {type(rag_components['compiled_workflow'])}")
        logger.info("RAG components initialization successful.")

    except Exception as e:
        logger.error(f"Error during RAG components initialization: {e}", exc_info=True)
        raise # Re-raise to signal failure to the Flask app


# --- Model List (Groq only) ---
model_list = [
    "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192",
    "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b"
]

# --- RAG Prompt (Same as original) ---
rag_prompt = PromptTemplate(
    template=r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                "You are a helpful, highly accurate and trustworthy assistant specialized in answering questions related to doing business, investing, and entrepreneurship specifically within Pakistan. Your goal is to provide actionable, clear, and contextually relevant information for users navigating Pakistan's business landscape."
                Your responses must strictly adhere to the provided context, answer style, question's language, and conversation history using the follow rules:

                1. **Question and answer language**:
                - Detect the question's primary language (e.g., English, Urdu, etc.). Your entire response must be in this primary language. If the question is a mix, use the language that constitutes the majority of the query or the language of the core question being asked." (Urdu script handling by LLMs can be hit-or-miss, but this is a better instruction).

                2. If the context documents contain 'Internet search results:' in 'page_content' field, always consider them in your response. Documents starting with 'Smart guide results:' are from internal documents.

                3. **Context-Only Answers with a given answer style**:
                - Always base your answers on the provided context and answer style.
                - If the context explicitly states 'I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship,' output this context verbatim. This might appear as a document starting with "Apology:".

                4. **Response style**:
                - Address the query directly without unnecessary or speculative information.
                - Do not draw from your knowledge base; strictly use the given context. However, for 'Moderate' or 'Explanatory' answer styles, you may synthesize the provided context more deeply, offering clearer explanations, illustrative examples derived from the context, and structuring the information as an experienced business advisor would, to enhance clarity. Always ground these elaborations firmly in the provided contextual documents.
                5. **Answer style**
                - if the context contains documents with two different prefixes 'Smart guide results:' and 'Internet search results:', strictly follow the format for answer generation specified in rule 9: "hybrid context handling". In that case, create two distinct sections even for 'concise' answer style.
                - If answer style = "Concise", generate a concise answer. But create the two sections as mentioned before if there are two different prefixes.
                - If answer style = "Moderate", use a moderate approach to generate answer where you can provide a little bit more explanation and elaborate the answer to improve clarity, integrating your own experience.
                - If answer style = "Explanatory", provide a detailed and elaborated answer in the question' language by providing more explanations with examples and illustrations to improve clarity in best possible way, integrating your own experience. However, the explanations, examples and illustrations should be strictly based on the context.
                6. **Conversational tone**
                 - Maintain a conversational and helping style. Anticipate potential follow-up questions or related areas of interest based on the user's query and the provided context. Where appropriate, you can subtly offer to provide more details on a sub-topic or point to related concepts if they appear in the context, e.g., 'This covers [topic X]. Would you like to know more about [related topic Y also mentioned in the documents]?
                 - Use simple language. Explain difficult concepts or terms wherever needed. Present the information in the best readable form.

                7. **Formatting Guidelines**:
                - Use bullet points for lists.
                - Include line breaks between sections for clarity.
                - Highlight important numbers, dates, and terms using **bold** formatting.
                - Create tables wherever appropriate to present data clearly.
                - If there are discrepancies in the context, clearly explain them.

                8. **Citation Rules**:
                - **very important**: Include citations in the answer at all relevant places if they are present in the context. Under no circumstances ignore them.
                - For responses based on the context documents starting with 'Smart guide results:', cite the document source (e.g., source_file metadata) with each piece of information in the format: [source_file]. Add page number if available [source_file, page xx].
                - For responses based on the documents starting with 'Internet search results:', include all the URLs (from metadata.source) in hyperlink form returned by the websearch. **very important**: The URLs should be labelled with the website title (metadata.title). Example: [Website Title](URL).
                - Do not invent any citation or URL. Only use the citation or URL in the context.

                9. **Hybrid Context Handling**:
                - If and only if the context contains documents starting with both 'Smart guide results:' and 'Internet search results:', structure your response in corresponding sections with the following headings:
                    - **Smart guide results**: Include data from documents starting with 'Smart guide results:' and its citations. If these documents do not contain any information relevant to the question, output 'No relevant information found in smart guides.' in this section.
                    - **Internet search results**: Include data from documents starting with 'Internet search results:' and its citations (URLs). If these documents do not contain any relevant information, output 'No relevant information found from internet search.' in this section.
                    - Ensure that you create two separate sections if and only if the context contain documents starting with both prefixes.
                    - Do not create two different sections or mention 'Smart guide results:' or 'Internet search results:' in your response if the context only contains documents with one prefix (or only the apology message).
                    - If answer style = "Explanatory", both the sections should be detailed and should contain all the points relevant to the question found in their respective contexts.
                10. **Integrity and Trustworthiness**:
                - Ensure every part of your response complies with these rules.

                --- CONVERSATION HISTORY ---
                {chat_history}
                --- END CONVERSATION HISTORY ---

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question}
                Answer style: {answer_style}
                Context:
                {context}

                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "answer_style", "chat_history"]
)

# --- Graph State ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]
    answer_style: str
    chat_history: List[Union[HumanMessage, AIMessage]]


# --- Graph Nodes (modified to use rag_components) ---

def retrieve(state: GraphState) -> GraphState:
    logger.info("---NODE: RETRIEVE---")
    question = state["question"]
    retriever = rag_components.get('retriever')
    if not retriever:
         logger.error("Retriever not initialized!")
         return {**state, "documents": []} # Keep state, return empty docs
    logger.info(f"Retrieving documents for: {question}")
    try:
        documents = retriever.invoke(question)
        logger.info(f"Retrieved {len(documents)} documents.")
        formatted_docs = []
        for doc in documents: # Prefixing here
            formatted_doc = Document(page_content=f"Smart guide results: {doc.page_content}", metadata=doc.metadata)
            formatted_docs.append(formatted_doc)
        return {**state, "documents": formatted_docs}
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return {**state, "documents": []}


def grade_documents(state: GraphState) -> GraphState:
    logger.info("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state.get("documents", [])
    web_search_needed = "No"
    filtered_docs = []

    if not documents:
        logger.info("No documents retrieved. Web search needed.")
        web_search_needed = "Yes"
        return {**state, "documents": [], "web_search_needed": web_search_needed}

    doc_grader = rag_components.get('doc_grader')
    grader_llm = rag_components.get('grader_llm') # For logger model name
    if not doc_grader or not grader_llm:
         logger.error("Document grader chain or grader LLM not initialized!")
         web_search_needed = "Yes" # Assume web search needed if grader fails
         return {**state, "documents": [], "web_search_needed": web_search_needed}

    logger.info(f"Grading {len(documents)} retrieved documents using {grader_llm.model_name if hasattr(grader_llm, 'model_name') else 'grader LLM'}...")
    relevant_found = False
    for count, doc in enumerate(documents):
        try:
            # Grader expects original content
            original_content = doc.page_content.replace("Smart guide results: ", "", 1)
            score = doc_grader.invoke({"documents": [Document(page_content=original_content)], "question": question})
            grade = score.binary_score.strip().lower()
            logger.info(f"Document {count} relevance: {grade}")
            if grade == "yes":
                filtered_docs.append(doc) # Keep the prefixed doc
                relevant_found = True
        except Exception as e:
            logger.error(f"Error grading document {count}: {e}. Assuming irrelevant.")

    if not relevant_found:
        logger.info("No relevant documents found after grading. Web search needed.")
        web_search_needed = "Yes"
        filtered_docs = [] # Clear any previously added docs if no relevant ones found
    else:
        logger.info(f"Found {len(filtered_docs)} relevant documents after grading. No web search needed.")
        web_search_needed = "No"

    return {**state, "documents": filtered_docs, "web_search_needed": web_search_needed}


def web_search(state: GraphState) -> GraphState:
    logger.info("---NODE: WEB SEARCH---")
    question = state["question"]
    existing_documents = state.get("documents", []) 

    tavily_client = rag_components.get('tavily_client')
    if not tavily_client:
        logger.error("Tavily Client not initialized. Web search cannot be performed.")
        failure_doc = Document(page_content="Internet search results: Web search failed: Tavily client not configured.")
        all_docs = existing_documents + [failure_doc]
        return {**state, "documents": all_docs}

    # Clean question from prefixes for search
    search_query = re.sub(r'\b(Smart guide results:|Internet search results:)\b', '', question, flags=re.IGNORECASE).strip()
    search_query = search_query + " in Pakistan" # Add context
    logger.info(f"Performing web search for: '{search_query}'")

    web_results_docs = []
    try:
        # Using Tavily Client directly
        search_result = tavily_client.search(
            query=search_query,
            search_depth="basic", # or "advanced" if needed
            max_results=5,
            include_answer=False, # We just want content
            include_raw_content=False, # Processed content is usually better
            include_images=False,
            # site_filter=AUTHORITATIVE_PAKISTAN_BUSINESS_DOMAINS # This could be too restrictive; Tavily does relevance.
        )

        if search_result and "results" in search_result:
             for res in search_result["results"]:
                 content = res.get("content", "")
                 url = res.get("url", "")
                 title = res.get("title", url) # Use URL as title if title is missing
                 if content and url: # Ensure both content and URL are present
                     metadata = {"source": url, "title": title}
                     # Prefix web search results
                     web_doc = Document(page_content=f"Internet search results: {content}", metadata=metadata)
                     web_results_docs.append(web_doc)

        if not web_results_docs:
            logger.info("Web search returned no results.")
            # Provide a specific document indicating no results
            no_results_doc = Document(page_content="Internet search results: Web search did not find relevant results for this query.")
            web_results_docs.append(no_results_doc)

    except Exception as e:
        logger.error(f"Error during web search: {e}")
        error_doc = Document(page_content=f"Internet search results: Web search failed: {e}")
        web_results_docs.append(error_doc)

    # Combine with any existing (filtered) documents from retrieval
    combined_docs = existing_documents + web_results_docs
    logger.info(f"Total documents after web search: {len(combined_docs)}")

    return {**state, "documents": combined_docs}

def generate(state: GraphState) -> GraphState:
    logger.info("---NODE: GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    answer_style = state.get("answer_style", "Concise") # Default to Concise if not provided
    chat_history = state.get("chat_history", [])

    # Format chat history for the prompt
    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_history.append(f"Assistant: {msg.content}")
    history_str = "\n".join(formatted_history)
    
    logger.info(f"Chat History for generate node (first 300 chars):\n{history_str[:300]}...")

    llm = rag_components.get('llm')
    if not llm:
         logger.error("Answering LLM not initialized!")
         return {**state, "generation": "Error: LLM not ready."}

    if not documents: # No documents from retrieve/web search (or unrelated path leading here)
        logger.info("No documents available for generation.")
        # If history exists, LLM might still answer based on it, or apology doc is present.
        # If no docs AND no history, prompt might be weak.
        # The prompt has instructions for context-only answers.
        # If apology_doc is the only doc, it will be used as context.

    context_str = "\n\n".join([doc.page_content for doc in documents])
    logger.info(f"Generating {answer_style} response using {llm.model_name if hasattr(llm, 'model_name') else 'LLM'}...")

    rag_chain = rag_prompt | llm | StrOutputParser()

    try:
        # LLM streaming is True, but invoke() gets the full response here.
        # If token-wise streaming to client is desired, generate node must yield tokens.
        # For now, this node produces the full generation string.
        generation_output = rag_chain.invoke({
            "context": context_str,
            "question": question,
            "answer_style": answer_style,
            "chat_history": history_str
        })
        logger.info("Generation complete.")
        return {**state, "generation": generation_output}

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error during generation with {llm.model_name if hasattr(llm, 'model_name') else 'LLM'}: {error_message}")
        return {**state, "generation": f"Sorry, an error occurred during answer generation: {error_message}"}

def handle_unrelated(state: GraphState) -> GraphState:
    logger.info("---NODE: HANDLE UNRELATED---")
    response = "I apologize, but I'm designed to answer questions specifically related to business and entrepreneurship in Pakistan. Could you please rephrase your question to focus on these topics?"
    apology_doc = Document(page_content=f"Apology: {response}")
    # This node prepares the document. 'generate' node will produce the final response based on this.
    return {**state, "documents": [apology_doc], "generation": ""} # Clear any prior generation

def handle_chit_chat(state: GraphState) -> GraphState:
    logger.info("---NODE: HANDLE CHIT CHAT---")
    question = state["question"].lower().strip()
    user_name = "there" # Could be personalized if user info is available

    greetings_responses = {
        "hello": f"Hello {user_name}! How can I help you with your business questions about Pakistan today?",
        "hi": f"Hi {user_name}! What business-related questions about Pakistan do you have?",
        "hey": f"Hey {user_name}! I'm here to assist with your business queries regarding Pakistan. What's on your mind?",
        "good morning": f"Good morning {user_name}! Ready to talk about business in Pakistan?",
        "good afternoon": f"Good afternoon {user_name}! How can I assist with your Pakistan business inquiries?",
        "good evening": f"Good evening {user_name}! What can I help you with regarding business in Pakistan?",
    }
    acknowledgements_thanks = {
        "thanks": "You're welcome! Is there anything else I can help you with regarding business in Pakistan?",
        "thank you": "You're most welcome! Do you have any other questions about business in Pakistan?",
        "ok": "Okay! Let me know if you have any questions about business in Pakistan.",
        "okay": "Okay! Feel free to ask if anything comes up.",
        "got it": "Great! Let me know if you need more information on business in Pakistan.",
    }
    farewells = {
        "bye": "Goodbye! Feel free to ask if you have more questions later.",
        "goodbye": "Goodbye! Have a great day!",
        "see you": "See you! Take care."
    }
    chit_chat_map = {**greetings_responses, **acknowledgements_thanks, **farewells}
    question_cleaned = re.sub(r'[^\w\s]', '', question)
    response = chit_chat_map.get(question_cleaned)

    if not response:
        if len(question_cleaned.split()) <= 2: # Fallback for very short inputs
            response = f"Hello! I'm ready to help with your questions about business in Pakistan. What can I do for you?"
        else: # Safeguard if router sent a non-chitchat query here
            response = "I'm not sure how to respond to that in this context. Could you ask a business-related question about Pakistan?"
    
    # This node directly sets the generation.
    return {**state, "documents": [], "generation": response, "web_search_needed": "No"}


# --- Routing Functions ---
def route_question(state: GraphState) -> str:
    logger.info("---NODE: ROUTE QUESTION---")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    # Fetch the pre-loaded summaries
    doc_summaries = rag_components.get('doc_summaries', "No summaries available.")

    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage): formatted_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage): formatted_history.append(f"Assistant: {msg.content}")
    history_str = "\n".join(formatted_history)
    logger.info(f"Chat History for route_question node (first 300 chars):\n{history_str[:300]}...")
    
    # The tool descriptions are now context-aware based on the summaries
    tool_selection = {
        "retrieve": ("Choose this if the User Question can be answered by the content described in the 'Knowledge Base Summaries'. Cross-reference the question's topic with the summaries to find a match. This is for questions about business planning, laws, registration, etc., that are covered in the internal documents."),
        "websearch": ("Choose this if the User Question asks for highly current information (e.g., today's tax rates, recent news), or if the topic is clearly NOT mentioned in the 'Knowledge Base Summaries'. Use this as a fallback if the internal documents are not relevant."),
        "chit_chat": ("Simple greetings (e.g., 'hello', 'hi'), farewells (e.g., 'bye'), thanks (e.g., 'thank you'), or short acknowledgements (e.g., 'ok', 'got it')."),
        "unrelated": ("Substantive questions that are not related to business in Pakistan and are also not covered in the 'Knowledge Base Summaries'. Questions about other countries or completely off-topic subjects fit here.")
    }
    SYS_PROMPT = """Act as a specialized query router for a Pakistan business assistant. Your primary task is to decide the best first step to answer a user's question by analyzing it against a set of available knowledge base summaries.

Instructions:
1.  Read the User Question and the Conversation History.
2.  Carefully read the provided "Knowledge Base Summaries".
3.  Compare the User Question to the content of the summaries.
4.  Based on this comparison, select a tool using the 'Tool Descriptions'.
    - If the question seems answerable by the summaries, choose 'retrieve'.
    - If the summaries do not cover the question's topic, or if it requires real-time data, choose 'websearch'.
5.  Output *only* the single chosen tool name ('websearch', 'retrieve', 'chit_chat', or 'unrelated') in lowercase.

--- KNOWLEDGE BASE SUMMARIES ---
{doc_summaries}
--- END KNOWLEDGE BASE SUMMARIES ---
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", "Conversation History:\n{chat_history}\n\nUser Question: \"{question}\"\n\nTool Descriptions:\n{tool_selection}\n\nSelected Tool:"),
    ])
    # Note the new 'doc_summaries' input variable
    inputs = {"question": question, "tool_selection": tool_selection, "chat_history": history_str, "doc_summaries": doc_summaries}

    router_llm = rag_components.get('router_llm')
    if not router_llm:
         logger.error("Router LLM not initialized! Defaulting to 'retrieve'.")
         return "retrieve"

    try:
        logger.info(f"Routing question using {router_llm.model_name if hasattr(router_llm, 'model_name') else 'router LLM'}...")
        router_chain = prompt | router_llm | StrOutputParser()
        tool_raw = router_chain.invoke(inputs)
        tool = tool_raw.strip().lower()
        tool = ''.join(filter(str.isalpha, tool)) # Clean potential markdown or extra chars

        logger.info(f"Router LLM raw output: '{tool_raw}', Cleaned: '{tool}'")

        if tool in ["websearch", "retrieve", "unrelated", "chit_chat"]:
             logger.info(f"Routing decision: {tool}")
             return tool
        else:
             logger.warning(f"LLM Router output '{tool}' is invalid.")
             # Fallback logic remains the same
             cleaned_question_for_fallback = re.sub(r'[^\w\s]', '', question.lower().strip())
             if len(cleaned_question_for_fallback.split()) <= 2 and not any(kw in cleaned_question_for_fallback for kw in ["what", "how", "why", "when", "where", "list", "explain", "tell me about"]):
                 logger.info("Short query and invalid router output, defaulting to 'chit_chat'.")
                 return "chit_chat"
             logger.info("Defaulting to 'retrieve' due to invalid router output on a longer query.")
             return "retrieve"
    except Exception as e:
        logger.error(f"Error during LLM routing: {e}. Defaulting to 'retrieve'.")
        return "retrieve"

def decide_after_grading(state: GraphState) -> str:
    logger.info("---CONDITIONAL EDGE: DECIDE AFTER GRADING---")
    web_search_needed = state.get("web_search_needed", "No") # Default to No if somehow missing
    documents = state.get("documents", [])

    logger.info(f"Decision from grading node: web_search_needed = {web_search_needed}")
    logger.info(f"Number of documents after grading: {len(documents)}")

    if web_search_needed == "Yes":
        logger.info("Routing to: websearch")
        return "websearch"
    else: # web_search_needed == "No"
        if not documents: # Grader said no web search, but documents are empty (e.g. all irrelevant)
             logger.warning("No documents after grading, but web_search_needed is 'No'. Forcing websearch to be safe.")
             return "websearch" # Or, could go to generate with empty context if that's desired.
        else:
             logger.info("Routing to: generate")
             return "generate"

# --- Workflow Definition ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("handle_unrelated", handle_unrelated)
workflow.add_node("handle_chit_chat", handle_chit_chat)

workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
        "unrelated": "handle_unrelated",
        "chit_chat": "handle_chit_chat",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grading,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("handle_unrelated", "generate") # Unrelated goes to generate for styled apology
workflow.add_edge("generate", END)
workflow.add_edge("handle_chit_chat", END) # Chit-chat provides response and ends

# --- Follow-up question generation (moved from Streamlit app.py, uses a default LLM) ---
DEFAULT_LLM_FOR_FOLLOWUP = "llama-3.1-8b-instant" # Can be configured

def get_followup_questions(last_user_query: str, last_assistant_response: str) -> List[str]:
    """Generates follow-up questions based on the last interaction."""
    prompt = f"""Based on the conversation below:
User: {last_user_query}
Assistant: {last_assistant_response}
Generate three concise follow-up questions (max 15 words each) that a user might ask next about business or entrepreneurship in Pakistan, related to the conversation.
Each question should be on a separate line, starting with a number (e.g., 1. ). Do not include your own introductory text.
Follow-up Questions:"""
    try:
        # Ensure GROQ_API_KEY is set in environment for this to work
        followup_llm = ChatGroq(model=DEFAULT_LLM_FOR_FOLLOWUP, temperature=0.6)
        response = followup_llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        
        # Try to find numbered list items first
        questions = re.findall(r"^\s*\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)", text, re.MULTILINE | re.DOTALL)
        questions = [q.strip() for q in questions if q.strip()]

        if not questions: # Fallback to splitting by newlines if no numbered list found
             questions = [q.strip("-* ").strip() for q in text.split('\n') if q.strip("-* ").strip()]
        
        return [q[:100] for q in questions[:3]] # Limit length and number of questions
    except Exception as e:
        logger.error(f"Failed to generate follow-up questions: {e}")
        return []