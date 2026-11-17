
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings.embedding_model import get_embedding_model
from app.ingestion.pdf_loader import load_all_papers


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "researchiq_papers_image_aware"


def clean_metadata(metadata: dict) -> dict:
    """Remove nested metadata that ChromaDB cannot store."""

    cleaned = {}

    for key, value in metadata.items():

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value

        elif isinstance(value, list):
            if all(
                isinstance(item, (str, int, float, bool))
                for item in value
            ):
                cleaned[key] = value

    # Extract page numbers from Docling metadata
    dl_meta = metadata.get("dl_meta", {})

    if isinstance(dl_meta, dict):

        doc_items = dl_meta.get("doc_items", [])
        page_numbers = set()

        for item in doc_items:

            if not isinstance(item, dict):
                continue

            for provenance in item.get("prov", []):

                if isinstance(provenance, dict):
                    page_no = provenance.get("page_no")

                    if page_no is not None:
                        page_numbers.add(page_no)

        if page_numbers:
            sorted_pages = sorted(page_numbers)

            cleaned["page_numbers"] = sorted_pages
            cleaned["page_start"] = sorted_pages[0]
            cleaned["page_end"] = sorted_pages[-1]

        headings = dl_meta.get("headings", [])

        if headings:
            cleaned["headings"] = headings

    return cleaned


def generate_document_ids(documents):
    """Generate deterministic IDs for documents."""

    document_ids = []

    for document in documents:

        source = document.metadata.get("source", "unknown")

        content = f"{source}:{document.page_content}"

        document_id = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        document_ids.append(document_id)

    return document_ids


def create_vector_store(documents):

    if not documents:
        raise ValueError("No documents available for indexing.")

    print("\n========== CLEANING METADATA ==========")

    for document in documents:
        document.metadata = clean_metadata(document.metadata)

    print(f"Documents received: {len(documents)}")

    embedding_model = get_embedding_model()

    print("\n========== CREATING CHROMA DATABASE ==========")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    document_ids = generate_document_ids(documents)

    print("\n========== INDEXING DOCUMENTS ==========")

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    total_records = vector_store._collection.count()

    print("\n========== VECTOR STORE RESULT ==========")
    print(f"Total chunks indexed: {len(documents)}")
    print(f"Total records in Chroma: {total_records}")
    print(f"Database location: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    return vector_store


def get_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


if __name__ == "__main__":

    print("\n========== LOADING RESEARCH PAPERS ==========")

    documents = load_all_papers()

    print("\n========== DOCUMENT SUMMARY ==========")
    print(f"Total documents: {len(documents)}")

    vector_store = create_vector_store(documents)

    print("\nVector database created successfully.")



def get_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store


if __name__ == "__main__":

    documents = load_all_papers()

    vector_store = create_vector_store(documents)

    print("\nVector database created successfully.")

    print(
        "Total records in Chroma:",
        vector_store._collection.count()
    )
