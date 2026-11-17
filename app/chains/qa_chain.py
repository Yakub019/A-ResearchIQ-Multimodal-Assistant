
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.retrieval.retriever import (
    retrieve_documents,
    format_documents,
)
from app.prompts.qa_prompts import QA_PROMPT


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2,
    )


def get_qa_chain():
    """Create the LangChain question-answering chain."""
    return QA_PROMPT | get_llm() | StrOutputParser()


def answer_question(question: str):
    """
    Answer questions using retrieved text and image descriptions.
    """

    # Step 1: Retrieve relevant chunks
    source_documents = retrieve_documents(question, k=4)

    if not source_documents:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the research papers.",
            "source_documents": [],
        }

    # Step 2: Format retrieved context
    context = format_documents(source_documents)

    # Step 3: Generate grounded answer
    answer = get_qa_chain().invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "question": question,
        "answer": answer,
        "source_documents": source_documents,
    }


if __name__ == "__main__":
    question = "Explain the diagram showing how human plans are inferred."

    result = answer_question(question)

    print("\n========== RESEARCHIQ ANSWER ==========")
    print(result["answer"])

    print("\n========== SOURCES ==========")

    for document in result["source_documents"]:
        print(document.metadata.get("source"))
        print(document.metadata.get("page_numbers"))
