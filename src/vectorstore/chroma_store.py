"""ChromaDB vector store integration (following NOVA pattern)"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.config import settings


class ChromaStore:
    """ChromaDB vector store for financial data embeddings"""

    def __init__(
        self,
        collection_name: str = "cosmo_financial_data",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(settings.chroma_persist_dir)

        # Initialize embeddings (OpenAI for now, as per settings)
        self.embeddings = self._initialize_embeddings()

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            )
        )

        # Initialize LangChain Chroma vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def _initialize_embeddings(self):
        """Initialize embedding model based on settings"""
        if settings.embedding_provider == "openai":
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        else:
            # For GLM embeddings, we would use GLM's embedding endpoint
            # For now, default to OpenAI
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store

        Args:
            documents: List of LangChain Document objects
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        return self.vectorstore.add_documents(documents=documents, ids=ids)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to the vector store

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text

        Returns:
            List of document IDs
        """
        return self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Query string
            k: Number of results to return (defaults to settings.retrieval_top_k)
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        k = k or settings.retrieval_top_k
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        k = k or settings.retrieval_top_k
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        collection = self.client.get_collection(name=self.collection_name)
        return collection.count()

    def as_retriever(self, **kwargs):
        """Get a LangChain retriever interface"""
        return self.vectorstore.as_retriever(**kwargs)


def create_chroma_store(
    collection_name: str = "cosmo_financial_data",
    persist_directory: Optional[str] = None,
) -> ChromaStore:
    """Factory function to create ChromaStore instance"""
    return ChromaStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
