import openai
import faiss
import re
import chainlit as cl
import warnings
import phoenix as px
from phoenix.trace import using_project
from phoenix.otel import register
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Phoenix session and OpenTelemetry tracer
with using_project('rag_chatbot'):
    session = px.launch_app()
tracer_provider = register(project_name="rag_chatbot", endpoint="http://localhost:6006/v1/traces")

# Toggling between models
huggingfaceopenmodel = True  # True for Hugging Face, False for GPT-4o

# Gpt4o API key details
openai.api_key = ${{secrets.APP_KEY}}
openai.api_base = "https://llmops-classroom-openai.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-07-01-preview"
gpt_deployment_name = "llmops_CT_GPT4o"

# FAISS index for Vector DB
dimension = 768  # BERT embedding size
index = faiss.IndexFlatIP(dimension)
document_chunks = []  # To store document chunks
chunk_embeddings = []  # To store embeddings for retrieval

# Hugging Face BERT model based tokenizer for embeddings
embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('bert-base-uncased')

# Hugging Face QA model
if huggingfaceopenmodel:
    hf_qa_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    hf_qa_model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    hf_qa_pipeline = pipeline("question-answering", model=hf_qa_model, tokenizer=hf_qa_tokenizer)

# Extract text from a URL shared
def extract_text_from_url(url):
    try:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted tags
        for unwanted_tag in soup(["script", "style", "footer", "nav", "header"]):
            unwanted_tag.decompose()

        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text(separator=" ", strip=True) for para in paragraphs if len(para.get_text(strip=True)) > 50])
        return text
    except Exception as e:
        print(f"Error extracting content from URL: {e}")
        return ""

# Sliding window method to create overlapping chunks (applies to both models)
def rolling_window_chunking(text, window_size=250, step_size=100):
    words = text.split()
    chunks = []
    for start in range(0, len(words), step_size):
        chunk = words[start:start + window_size]
        if len(chunk) > 0:
            chunks.append(' '.join(chunk))
    return chunks

# Generate embeddings using Hugging Face BERT model
def get_huggingface_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding / np.linalg.norm(embedding)  # Normalize embedding

# Generate answers using Hugging Face QA model (If True)
def generate_answer_huggingface(question, context):
    answer = hf_qa_pipeline(question=question, context=context)
    return answer['answer'], answer.get('score')  # Return answer and confidence score from QA model

# Generate answers using Azure GPT-4o
def generate_answer_gpt4o(question, context=None):
    try:
        response = openai.ChatCompletion.create(
            engine=gpt_deployment_name,  # Use your deployment name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": question}
            ],
            max_tokens=300,  # Tokens limited for GPT-4o usage
            temperature=0.7, 
            stop=None
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer, None
    except Exception as e:
        print(f"Error generating answer using GPT-4o: {e}")
        return None, None

# Rank chunks using cosine similarity
def rank_chunks_by_similarity(question_embedding, chunk_embeddings):
    scores = [cosine_similarity([question_embedding], [chunk_emb])[0][0] for chunk_emb in chunk_embeddings]
    return np.array(scores)

# Dynamically build context using the sliding window approach
def build_context_from_chunks(ranked_indices, max_chunks=3):
    selected_chunks = [document_chunks[i] for i in ranked_indices[:max_chunks]]
    return ' '.join(selected_chunks)

# Chainlit message handler
@cl.on_message
async def handle_message(message):
    text = message.content.strip()

    # Initialize Phoenix tracing
    with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("handle_message") as span:
        url_pattern = re.compile(r'(https?://\S+)')
        match = re.search(url_pattern, text)

        if match:
            url = match.group(0)
            # Text extraction from URL
            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("extract_text_from_url") as extract_span:
                extracted_text = extract_text_from_url(url)
                extract_span.set_attribute("document_length", len(extracted_text))
            if not extracted_text:
                await cl.Message(content="Failed to extract content from the URL. Please try a different URL.").send()
                return

            # Create rolling window chunks of text
            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("rolling_window_chunking") as chunk_span:
                chunks = rolling_window_chunking(extracted_text)
                chunk_span.set_attribute("num_chunks", len(chunks))

            # Generate embeddings and store in FAISS
            for chunk in chunks:
                with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("generate_huggingface_embeddings") as embed_span:
                    embedding = get_huggingface_embedding(chunk)
                    embed_span.set_attribute("embedding_model", "bert-base-uncased")
                    embed_span.set_attribute("chunk_size", len(chunk))  # Chunk size                 
                chunk_embeddings.append(embedding)
                index.add(np.array([embedding]))
                document_chunks.append(chunk)

            await cl.Message(content="The context is now set from the URL provided. You can now ask questions related to it.").send()

        else:
            if len(document_chunks) == 0:
                await cl.Message(content="No context is set yet. Please provide a URL to set up the context.").send()
                return

            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("question_embedding") as question_embed_span:
                question_embedding = get_huggingface_embedding(text)
                question_embed_span.set_attribute("embedding_model", "bert-base-uncased")

            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("faiss_search") as faiss_search_span:
                _, chunk_indices = index.search(np.array([question_embedding]), 5)
                faiss_search_span.set_attribute("num_chunks_in_search", len(chunk_indices[0]))

            retrieved_chunks = [document_chunks[i] for i in chunk_indices[0]]

            # Build context using sliding window
            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("build_context") as context_span:
                full_context = build_context_from_chunks(chunk_indices[0], max_chunks=3)

            # Perform cosine similarity
            with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("cosine_similarity") as cosine_span:
                chunk_similarities = rank_chunks_by_similarity(question_embedding, chunk_embeddings)

            top_similarity = np.max(chunk_similarities)

            # Generate answer
            if huggingfaceopenmodel:
                with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("generate_answer_huggingface") as qa_span:
                    best_answer, confidence = generate_answer_huggingface(text, full_context)
                    qa_span.set_attribute("confidence_score", confidence)
            else:
                with tracer_provider.get_tracer("rag_chatbot").start_as_current_span("generate_answer_gpt4o") as gpt_span:
                    best_answer, confidence = generate_answer_gpt4o(text, full_context)
                    gpt_span.set_attribute("gpt_version", "gpt-4o")

            if best_answer is None:
                response = "Could not generate a confident enough answer."
            else:
                response = f"Answer: {best_answer}\n\nCosine Similarity: {top_similarity:.2f}"

                if huggingfaceopenmodel and confidence is not None:
                    response += f"\nConfidence: {confidence:.2f}"

            await cl.Message(content=response).send()

# Print Phoenix URL to open in browser
print(f"Phoenix UI is available at: {session.url}")