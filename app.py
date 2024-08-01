import streamlit as st
import PyPDF2
import json
import torch
import ollama
import os
import re
from openai import OpenAI

# ANSI escape codes for colors (for terminal output, not used in Streamlit)
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='your_api_key_here'
)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ''
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return clean_text(text)

# Function to extract text from a JSON file
def extract_text_from_json(file):
    data = json.load(file)
    text = json.dumps(data, ensure_ascii=False)
    return clean_text(text)

# Function to clean and chunk text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < 1000:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return "\n".join(chunks)

# Function to append text to the vault
def append_to_vault(text):
    with open("vault.txt", "a", encoding="utf-8") as vault_file:
        vault_file.write(text + "\n")

# Function to generate embeddings for the vault content
def generate_vault_embeddings(vault_content):
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    return torch.tensor(vault_embeddings)

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to rewrite the query based on conversation history
def rewrite_query(user_input_json, conversation_history):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model='llama3',
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, conversation_history):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, top_k=3)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        st.info("Context Pulled from Documents: \n\n" + context_str)
    else:
        st.info("No relevant context found.")
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model='llama3',
        messages=messages
    )
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Return the content of the response from the model
    return response.choices[0].message.content

# Define Streamlit app
def main():
    st.title("RAG-Based Interactive Chat")

    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, ask me a question. My knowledge is always up to date!"}
        ]

    # Load the vault content
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    # Generate embeddings for the vault content
    vault_embeddings_tensor = generate_vault_embeddings(vault_content)

    # Display the conversation history
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        else:
            with st.chat_message("human"):
                st.markdown(message["content"])

    # Draw the chat input box
    if question := st.chat_input("What's up?"):
        
        # Draw the user's question
        with st.chat_message('human'):
            st.markdown(question)

        # Generate the answer
        answer = ollama_chat(question, "You are a helpful assistant that is an expert at extracting the most useful information from a given text.", vault_embeddings_tensor, vault_content, st.session_state.messages)

        # Draw the bot's answer
        with st.chat_message('assistant'):
            st.markdown(answer)
        
        # Save conversation
        st.session_state.messages.append({"role": "human", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # File upload section
    st.subheader("Upload Files")
    uploaded_file = st.file_uploader("Choose a PDF, TXT, or JSON file", type=["pdf", "txt", "json"])

    if uploaded_file is not None:
        # Process the uploaded file
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'txt':
            text = uploaded_file.read().decode("utf-8")
        elif file_type == 'json':
            text = extract_text_from_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # Append text to vault
        append_to_vault(text)
        st.success("File content appended to vault.txt")

if __name__ == "__main__":
    main()
