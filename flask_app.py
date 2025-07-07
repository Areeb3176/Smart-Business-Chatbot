# flask_app.py
import os
import json
import time
import logging
import re
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Custom RAG logic
import agentic_rag_flask 

# Load .env file at the very beginning
load_dotenv()

# It's often a C++ extension issue with sentence-transformers on some setups.
# If you don't need it, you can remove it.
try:
    import torch
    torch.classes.__path__ = []
    logging.info("torch.classes.__path__ cleared.")
except ImportError:
    logging.warning("PyTorch not found, skipping torch.classes.__path__ clear.")


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Configure logging for Flask app
# Ensure this is the primary basicConfig call for logging
if not logging.getLogger().hasHandlers(): # Avoid reconfiguring if already set by another module (unlikely here)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Application Configuration ---
# These will be the defaults. UI can send updates to /config
# which will then call setup_rag_application again.
current_rag_config = {
    "selected_model": "llama-3.1-8b-instant",
    "selected_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "selected_routing_model": "llama-3.1-8b-instant",
    "selected_grading_model": "llama-3.1-8b-instant",
}

# --- RAG Application Setup ---
def setup_rag_application(config_params):
    """Initializes or re-initializes the RAG components."""
    logging.info(f"Setting up RAG application with config: {config_params}")
    try:
        agentic_rag_flask.initialize_rag_components(
            model_name=config_params["selected_model"],
            selected_embedding_model=config_params["selected_embedding_model"],
            selected_routing_model=config_params["selected_routing_model"],
            selected_grading_model=config_params["selected_grading_model"]
        )
        logging.info("RAG application setup/re-initialization complete.")
        # Store the successfully used config
        global current_rag_config
        current_rag_config = config_params.copy()
    except Exception as e:
        logging.error(f"FATAL: Error Initializing RAG Application: {e}", exc_info=True)
        # This is a critical failure; the app might not function correctly.
        # Depending on deployment, you might want to exit or have specific error handling.
        raise  # Re-raise to make it visible during startup or config change

# Initial setup on Flask app start
try:
    setup_rag_application(current_rag_config)
except Exception as e:
    logging.critical(f"Flask app startup failed due to RAG initialization error: {e}")
    # Depending on your needs, you might sys.exit(1) here or let Flask start with RAG disabled.
    # For now, we let it start and endpoints will fail if RAG is not ready.

# --- Helper for Chat History Preparation ---
def prepare_langchain_history_for_graph(session_messages, n_turns_memory=3):
    # `session_messages` includes the current user query.
    # History for the graph should be turns *before* the current query.
    history_for_conversion = session_messages[:-1] if session_messages else []
    
    temp_graph_history = []
    assistant_message_buffer = None
    pairs_collected = 0

    for i in range(len(history_for_conversion) - 1, -1, -1):
        message = history_for_conversion[i]
        if message["role"] == "assistant":
            # Prefer 'raw_content' for history, as 'content' might have HTML (timer)
            content = message.get("raw_content", message.get("content", "")) 
            if content and content != "...": # Avoid placeholders
                # If raw_content contained timer, strip it for pure AI message history
                # For now, assume raw_content is PURE AI response as per current logic.
                # This is consistent with how it's saved below.
                assistant_message_buffer = AIMessage(content=content)
        elif message["role"] == "user":
            user_content = message.get("content", "")
            if user_content and assistant_message_buffer: 
                if pairs_collected < n_turns_memory:
                    # Standard order for Langchain history: [AI_old, Human_old, AI_new, Human_new]
                    # insert at 0 means: Human first, then AI for that pair
                    temp_graph_history.insert(0, HumanMessage(content=user_content))
                    temp_graph_history.insert(0, assistant_message_buffer)
                    assistant_message_buffer = None 
                    pairs_collected += 1
                else:
                    break 
    
    return temp_graph_history

# --- Flask Routes ---
@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = []
    if 'show_timer' not in session:
        session['show_timer'] = True # Default to showing timer

    # Pass data needed for the UI (models, current settings)
    return render_template('index.html',
                           messages=session['messages'],
                           model_list=agentic_rag_flask.model_list,
                           embed_list=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"],
                           current_config=current_rag_config, # Global config reflecting current RAG setup
                           default_answer_style=session.get('answer_style', "Explanatory"),
                           show_timer=session['show_timer'])

# flask_app.py
# ... (other imports and code) ...

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    answer_style = data.get('answer_style', 'Explanatory') 
    session['answer_style'] = answer_style 
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if 'messages' not in session:
        session['messages'] = []
    
    session['messages'].append({"role": "user", "content": question})
    session.modified = True

    langchain_history = prepare_langchain_history_for_graph(session['messages'])
    logging.info(f"Prepared LangGraph history with {len(langchain_history)} messages for {len(session['messages'])-1} prior turns.")

    # Define compiled_workflow and inputs_for_graph in the chat function's scope
    compiled_workflow = agentic_rag_flask.rag_components.get('compiled_workflow')
    if not compiled_workflow:
        logging.error("RAG compiled_workflow not found in rag_components.")
        error_msg = "Error: The RAG system is not initialized. Please contact support."
        session['messages'].append({"role": "assistant", "content": error_msg, "raw_content": error_msg})
        session.modified = True
        return jsonify({"error": "RAG system not initialized. Cannot process chat."}), 503

    inputs_for_graph = {
        "question": question,
        "answer_style": answer_style,
        "chat_history": langchain_history
    }

    # Pass compiled_workflow and inputs_for_graph to generate_sse_stream
    def generate_sse_stream(workflow_to_run, graph_inputs): # Added parameters
        start_time = time.time()
        accumulated_response_content = ""
        final_graph_output_details = None

        try:
            logging.info(f"Invoking RAG graph stream with input: question='{graph_inputs['question'][:50]}...', style='{graph_inputs['answer_style']}', history_len={len(graph_inputs['chat_history'])}")
            
            # Use the passed-in parameters
            for chunk in workflow_to_run.stream(graph_inputs, {"recursion_limit": 25}):
                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]
                logging.info(f"SSE Stream: Graph node '{node_name}' completed. Output keys: {list(node_output.keys()) if isinstance(node_output, dict) else 'Not a dict'}") # My addition from before
                
                if node_name in ["generate", "handle_chit_chat"] and isinstance(node_output, dict) and "generation" in node_output:
                    final_graph_output_details = node_output
                    current_generation_chunk = node_output["generation"]
                    if current_generation_chunk:
                        accumulated_response_content = current_generation_chunk 
                        yield f"data: {json.dumps({'type': 'message_chunk', 'content': accumulated_response_content})}\n\n"
                        logging.info(f"SSE Stream: Sent message chunk from {node_name}: '{accumulated_response_content[:100]}...'")
            
            if not accumulated_response_content:
                if final_graph_output_details and isinstance(final_graph_output_details, dict) and "generation" in final_graph_output_details:
                    accumulated_response_content = final_graph_output_details["generation"]
                    if accumulated_response_content:
                        logging.info("SSE Stream: Recovered response from final_graph_output_details.")
                        yield f"data: {json.dumps({'type': 'message_chunk', 'content': accumulated_response_content})}\n\n"
                    else:
                        accumulated_response_content = "The process completed, but I did not generate a specific response."
                        logging.warning(f"SSE Stream: Using fallback for empty generation. Final graph output: {final_graph_output_details}")
                        yield f"data: {json.dumps({'type': 'message_chunk', 'content': accumulated_response_content})}\n\n"
                else:
                    accumulated_response_content = "I'm not sure how to respond to that. Please try rephrasing or ask a business question about Pakistan."
                    logging.warning(f"SSE Stream: Using fallback response (no generation found). Final graph output: {final_graph_output_details}")
                    yield f"data: {json.dumps({'type': 'message_chunk', 'content': accumulated_response_content})}\n\n"

        except Exception as e:
            # Log the raw exception first with full traceback
            logging.error(f"SSE Stream: Exception caught in RAG graph execution: {type(e).__name__} - {str(e)}", exc_info=True)
            
            client_error_content = "Sorry, an unexpected error occurred on the server. Please try again." # Default
            try:
                specific_detail = str(e)
                if specific_detail and specific_detail.strip():
                    client_error_content = f"An error occurred: {specific_detail[:150]}" 
                else:
                    client_error_content = "An unspecified error occurred during processing."
            except Exception as e_str_conv:
                logging.error(f"SSE Stream: Could not convert exception to string for client: {e_str_conv}")
            
            logging.info(f"SSE Stream: Yielding error to client with content: '{client_error_content}'")
            # Ensure the accumulated_response_content that gets saved to session reflects this error too.
            accumulated_response_content = client_error_content # Set this so session reflects the error
            yield f"data: {json.dumps({'type': 'error', 'content': client_error_content})}\n\n" # Send error to client
        
        end_time = time.time()
        generation_time = end_time - start_time
        logging.info(f"SSE Stream: Processing (success or error path) finished in {generation_time:.2f}s.")

        timer_html_for_display = ""
        # Check if an error type message was sent. If so, accumulated_response_content is already the error message.
        # We might not want a timer for an error.
        is_actual_error_response = "An error occurred:" in accumulated_response_content or "Sorry, an unexpected error occurred" in accumulated_response_content

        if not is_actual_error_response and session.get('show_timer', True) and generation_time > 0.1:
            timer_html_for_display = f"<br><small><i>Response generated in: {generation_time:.2f} seconds</i></small>"
            yield f"data: {json.dumps({'type': 'timer', 'html_content': timer_html_for_display})}\n\n"

        content_for_session_display = accumulated_response_content + timer_html_for_display
        
        assistant_message_for_session = {
            "role": "assistant",
            "content": content_for_session_display,
            "raw_content": accumulated_response_content # If error, this is the error message
        }
        session['messages'].append(assistant_message_for_session)
        session.modified = True

        # Use graph_inputs here as well for consistency for last_user_q
        last_user_q = graph_inputs['question'] 
        last_assistant_raw_resp = accumulated_response_content # This is the raw response or error string
        
        # Update the check for is_error_or_unrelated to be more robust
        is_error_or_unrelated_for_followup = (
            not last_assistant_raw_resp or
            is_actual_error_response or # Use the flag we determined above
            "Error: The RAG system is not initialized" in last_assistant_raw_resp or # Specific init error
            "I apologize, but I'm designed to answer questions" in last_assistant_raw_resp or # Unrelated apology
            "I'm not sure how to respond" in last_assistant_raw_resp # Generic fallback
        )

        followup_questions_list = []
        if not is_error_or_unrelated_for_followup:
            try:
                followup_questions_list = agentic_rag_flask.get_followup_questions(last_user_q, last_assistant_raw_resp)
                logging.info(f"SSE Stream: Generated follow-up questions: {followup_questions_list}")
            except Exception as e:
                logging.error(f"SSE Stream: Error generating follow-up questions: {e}", exc_info=True) # Add exc_info
        
        yield f"data: {json.dumps({'type': 'followup_questions', 'questions': followup_questions_list})}\n\n"
        
        yield f"event: end_stream\ndata: Stream ended\n\n"

    # Call generate_sse_stream with the necessary arguments
    return Response(stream_with_context(generate_sse_stream(compiled_workflow, inputs_for_graph)), mimetype='text/event-stream')

# ... (rest of flask_app.py)


@app.route('/reset', methods=['POST'])
def reset_chat():
    session.pop('messages', None)
    # session.pop('answer_style', None) # Keep UI preferences or reset them? For now, keep.
    logging.info("Chat session reset.")
    return jsonify({"status": "success", "message": "Conversation reset"})

@app.route('/config', methods=['POST'])
def update_config():
    data = request.json
    logging.info(f"Received config update request: {data}")
    
    new_config = current_rag_config.copy() # Start with current global config
    needs_reinitialization = False

    # Update model selections
    for key in ["selected_model", "selected_embedding_model", "selected_routing_model", "selected_grading_model"]:
        if key in data and new_config.get(key) != data[key]:
            new_config[key] = data[key]
            needs_reinitialization = True
            logging.info(f"Config change: {key} from {current_rag_config.get(key)} to {data[key]}")
    
    # Update UI preferences stored in session
    if 'show_timer' in data:
        session['show_timer'] = data['show_timer']
        logging.info(f"Session 'show_timer' updated to: {session['show_timer']}")
    
    # Persist answer style from config to session if sent
    if 'answer_style' in data:
        session['answer_style'] = data['answer_style']
        logging.info(f"Session 'answer_style' updated to: {session['answer_style']}")


    if needs_reinitialization:
        try:
            logging.info("Re-initializing RAG application due to config change...")
            setup_rag_application(new_config) # This will update global current_rag_config on success
            return jsonify({"status": "success", "message": "Configuration updated and RAG system re-initialized."})
        except Exception as e:
            logging.error(f"Error re-initializing RAG application: {e}", exc_info=True)
            # Revert to old config display on client? For now, client needs to handle UI state.
            return jsonify({"status": "error", "message": f"Failed to re-initialize RAG system: {str(e)}"}), 500
    else:
        return jsonify({"status": "success", "message": "Configuration updated (no RAG re-initialization needed)."})

if __name__ == '__main__':
    # Make sure DATA_FOLDER exists, or create it if appropriate for your app
    if not os.path.exists(agentic_rag_flask.DATA_FOLDER):
        logging.warning(f"Data folder '{agentic_rag_flask.DATA_FOLDER}' does not exist. Vector store creation might fail if it's new.")
        # os.makedirs(agentic_rag_flask.DATA_FOLDER, exist_ok=True) # Uncomment to create if needed
    
    app.run(debug=True, host='0.0.0.0', port=5001) # Use a different port if 5000 is common