from pathlib import Path

from app.ingestion.loaders import FileParser
from app.ingestion.cleaner import clean_document


def test_clean_documents():
    documents_path = Path("data/documents")

    parser = FileParser()

    documents = parser.load_directory(documents_path)

    print(f"Loaded documents: {len(documents)}")
    print("*" * 80)

    for document in documents:
        print(f"\nSource ID: {document.source_id}")
        print(f"Title: {document.title}")
        print(f"Media type: {document.media_type}")

        print("\nOriginal text:")
        print(document.text[:500])

        cleaned_document = clean_document(document)

        print("\nCleaned text:")
        print(cleaned_document.text[:500])

        print("\n" + "*" * 80)

        
        assert cleaned_document.source_id == document.source_id
        assert cleaned_document.title == document.title
        assert cleaned_document.media_type == document.media_type

        assert "   " not in cleaned_document.text
        assert "\n\n\n" not in cleaned_document.text

    print("\nCleaning test passed!")


if __name__ == "__main__":
    test_clean_documents()