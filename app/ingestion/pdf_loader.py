
import os
from pathlib import Path

from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


def create_document_converter():

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file."
        )

    pipeline_options = PdfPipelineOptions()

    # Enable picture descriptions
    pipeline_options.do_picture_description = True

    # Enable remote services
    pipeline_options.enable_remote_services = True

    # Configure Groq vision API
    picture_options = PictureDescriptionApiOptions(
        url="https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {groq_api_key}"
        },
        params={
            "model": "qwen/qwen3.8-27b",
            "max_completion_tokens": 200,
        },
        prompt=(
            "Describe this image accurately. "
            "If it contains a chart, explain its axes, values, "
            "trends, labels, and key findings. "
            "If it contains a diagram, explain its components "
            "and relationships."
        ),
        timeout=120,
        concurrency=1,
        scale=1.0,
        batch_size=1,
    )

    pipeline_options.picture_description_options = picture_options

    # Verify configuration
    print("\n========== VISION CONFIGURATION ==========")
    print(
        "Picture description enabled:",
        pipeline_options.do_picture_description
    )
    print(
        "Backend:",
        type(pipeline_options.picture_description_options).__name__
    )
    print(
        "API endpoint:",
        pipeline_options.picture_description_options.url
    )
    print(
        "Vision model:",
        pipeline_options.picture_description_options.params["model"]
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def load_pdf(pdf_path: Path):

    print(f"\nProcessing: {pdf_path.name}", flush=True)

    converter = create_document_converter()

    loader = DoclingLoader(
        file_path=str(pdf_path),
        converter=converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(),
    )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = pdf_path.name

    print(
        f"\nSuccessfully processed: {pdf_path.name}",
        flush=True
    )
    print(f"Total chunks: {len(documents)}", flush=True)

    return documents


def load_all_papers():

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDFs found in data/papers/"
        )

    all_documents = []

    for pdf_path in pdf_files:

        try:
            documents = load_pdf(pdf_path)
            all_documents.extend(documents)

        except Exception as error:
            print(
                f"Failed: {pdf_path.name}: {error}",
                flush=True
            )

    print("\n========== INGESTION SUMMARY ==========")
    print(f"PDFs found: {len(pdf_files)}")
    print(f"Total chunks: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":

    test_pdf = PAPERS_DIR / "01_sample.pdf"

    documents = load_pdf(test_pdf)

    print("\n========== IMAGE-AWARE CHUNK TEST ==========")

    image_chunks = []

    for index, document in enumerate(documents, start=1):

        content = document.page_content

        if (
            "This image" in content
            or "Main Components" in content
            or "depicting" in content
        ):
            image_chunks.append(document)

        if index <= 10:
            print(f"\n--- CHUNK {index} ---")
            print(content[:1500])

    print("\n========== IMAGE DESCRIPTION SUMMARY ==========")
    print(f"Total chunks: {len(documents)}")
    print(f"Chunks containing image descriptions: {len(image_chunks)}")
