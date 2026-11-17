
from app.ingestion.pdf_loader import PAPERS_DIR, load_pdf


PDF_PATH = PAPERS_DIR / "01_sample.pdf"


def test_pdf_loader():

    print("\n========== PDF LOADER TEST ==========")

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    documents = load_pdf(PDF_PATH)

    assert documents, "No documents extracted from PDF."

    print(f"\nTotal chunks extracted: {len(documents)}")

    for index, document in enumerate(documents[:5]):

        print(f"\n========== CHUNK {index + 1} ==========")

        print("\nCONTENT:")
        print(document.page_content[:1000])

        print("\nMETADATA:")
        print(document.metadata)

        assert document.page_content.strip(), "Empty chunk found."

    print("\nPDF LOADER TEST PASSED!")


if __name__ == "__main__":
    test_pdf_loader()
