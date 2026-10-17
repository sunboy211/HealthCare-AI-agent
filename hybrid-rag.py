import os
from flashrank import Ranker
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate

# Initialize our local Ollama LLM
llm = Ollama(model="phi3", base_url="http://localhost:11434")

# Load and Split Internal Nordic Media/Care Data
loader = TextLoader("./internal_data/protocol.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Setup Retriever A: Dense Vector Search (Captures Semantics)
# (Note: Using an embedding model to vectorize text)
embeddings = OpenAIEmbeddings()  # Enforces semantic vectors, can be swapped to local BGE
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup Retriever B: Sparse Keyword Search (Captures Room IDs, Names, Dates)
# BM25 is the exact industry-standard keyword algorithm requested by Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# FUSE THEM: Hybrid Retrieval via EnsembleRetriever
# We assign weights (e.g., 50% Vector, 50% Keyword) to score and combine results
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# ADD RERANKING: Squeeze out the noise using a Cross-Encoder
# Flashrank is an ultra-fast, lightweight local rerank engine perfect for on-prem setup
compressor = FlashrankRerank(top_n=2)  # Only allow the absolute Top 2 dense chunks to pass
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

# Construct the Final LangChain AI Workflow (QA Chain)
system_prompt = (
    "You are a professional AI intelligence officer. Answer the user question "
    "strictly using the provided retrieved context below. If you do not know, say you don't know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(rerank_retriever, question_answer_chain)

# Trigger the Pipeline
if __name__ == "__main__":
    # create the file './internal_data/protocol.txt' before running!
    print("Executing Fused Hybrid RAG Pipeline...")
    response = rag_chain.invoke({"input": "What is the specific crisis protocol for Room 302?"})

    print("\n--- Final Guardrailed Response ---")
    print(response["answer"])
