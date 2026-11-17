
from app.vectorstore.vector_store import get_vector_store


def get_retriever(k: int = 4):
    """
    Return a retriever connected to the image-aware ChromaDB collection.
    """
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_documents(question: str, k: int = 4):
    """
    Retrieve relevant text and image-description chunks.
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(question)


def format_documents(documents):
    """
    Format retrieved documents with source and page citations.
    """
    formatted_documents = []

    for document in documents:
        content = document.page_content
        metadata = document.metadata

        source = metadata.get("source", "Unknown source")

        page_numbers = metadata.get("page_numbers", [])

        if page_numbers:
            pages = ", ".join(str(page) for page in page_numbers)
        else:
            pages = metadata.get("page_start", "Unknown")

        formatted_documents.append(
            f"{content}\n[Source: {source}, Page: {pages}]"
        )

    return "\n\n".join(formatted_documents)
