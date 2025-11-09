import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load and Extract Text from JSON
def extract_text_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    documents = []
    for entry in data:
        document_text = f"Station Code: {entry['CODE']}, Station Name: {entry['STATION NAME']}, Railway Zone: {entry['RAILWAY ZONE']}"
        documents.append(document_text)
    
    return documents

# Load the JSON file (replace with your file path)
json_file_path = 'konkanRailwayCodeNameZone.json'
documents = extract_text_from_json(json_file_path)
print("Documents extracted:", documents[:5])  # Print the first few documents

# Step 2: Convert Documents to Embeddings
# Using sentence-transformers for generating document embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for the extracted documents
embeddings = model.encode(documents)
print("Embeddings shape:", embeddings.shape)

# Step 3: Store Embeddings in FAISS Vector Store
dimension = embeddings.shape[1]  # Number of dimensions in the embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(np.array(embeddings))
print("Embeddings added to FAISS index.")

# Step 4: User Query and Embedding Retrieval
def retrieve_documents(query, k=5):
    # Convert the query into an embedding
    query_embedding = model.encode([query])
    print("Query embedding:", query_embedding)
    
    # Search for top-k similar documents
    D, I = index.search(np.array(query_embedding), k)  # Retrieves top-k documents
    print("Distances:", D)
    print("Indices:", I)
    
    # Retrieve and return the top-k documents
    retrieved_docs = [documents[i] for i in I[0]]
    return retrieved_docs

# Step 5: Hugging Face LLM for Response Generation
def generate_response(query, retrieved_docs):
    # Load Hugging Face's GPT-2 model and tokenizer
    model_name = 'EleutherAI/gpt-neo-2.7B'  # Replace with 'EleutherAI/gpt-j-6B' if you want to use that model
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    gpt_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Combine query and retrieved documents for input
    input_text = f"User query: {query}\nRelevant documents: {' '.join(retrieved_docs)}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate response from the model
    outputs = gpt_model.generate(**inputs, max_length=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Step 6: Example Run (Querying and Generating a Response)
def run_rag_pipeline(query):
    # Retrieve relevant documents based on query
    retrieved_docs = retrieve_documents(query, k=10)
    print("Retrieved documents:", retrieved_docs)

    # Generate a response based on the query and retrieved documents
    response = generate_response(query, retrieved_docs)
    
    # Make the response more elaborate and human-like
    if not retrieved_docs:
        return "I'm sorry, but I couldn't find any relevant stations based on your query. Please try a different query."

    return (f"Based on your query '{query}', here are some stations that might be of interest:\n\n" +
            "\n".join([f"- {doc}" for doc in retrieved_docs]) + 
            "\n\nI hope this helps! Let me know if you need more information.")

# Take user query as input
user_query = input("Enter your query: ")
response = run_rag_pipeline(user_query)
print("Generated Response:\n", response)
