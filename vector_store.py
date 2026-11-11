import uuid

from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import MultiVectorRetriever


class VectorStoreManager:
    def __init__(
        self,
        collection_name="multi_modal_rag",
        persist_directory="./chroma_db",
        embedding_model="BAAI/bge-m3",
        device="cuda",
    ):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Persistent vector database
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
            #persist_directory=persist_directory,
        )

        # Parent document storage
        self.store = InMemoryStore()
        self.id_key = "doc_id"

        # Multi-vector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

    def _add_items(self, items, summaries):
        """Index summaries and associate them with parent documents."""

        if len(items) != len(summaries):
            raise ValueError(
                f"Items ({len(items)}) and summaries "
                f"({len(summaries)}) must have equal lengths."
            )

        if not items:
            return

        ids = [str(uuid.uuid4()) for _ in items]

        summary_docs = [
            Document(
                page_content=summary,
                metadata={self.id_key: ids[i]},
            )
            for i, summary in enumerate(summaries)
        ]

        self.vectorstore.add_documents(summary_docs)
        self.store.mset(list(zip(ids, items)))

    def create_db(
        self,
        texts,
        text_summaries,
        tables,
        table_summaries,
        images,
        image_summaries,
    ):
        """Add text, table, and image parent documents."""

        self._add_items(texts, text_summaries)
        self._add_items(tables, table_summaries)
        self._add_items(images, image_summaries)

    def retrieve(self, query, **kwargs):
        """Retrieve parent documents relevant to a query."""
        return self.retriever.invoke(query, **kwargs)
