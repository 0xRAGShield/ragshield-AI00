from pathlib import Path

from app.ingestion.loaders import FileParser
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import (
    ParentChildChunker,
)


def main():

    documents_path = Path("data/documents")

    parser = FileParser()

    documents = parser.load_directory(
        documents_path
    )

    if not documents:
        print("No documents were loaded.")
        return

    print("PARENT-CHILD CHUNKING TEST")
    print("*" * 80)

    print(f"Total documents: {len(documents)}")

    chunker = ParentChildChunker(
        parent_chunk_size=1200,
        child_chunk_size=400,
    )

    total_chunks = 0

    for document in documents:

        print(f"DOCUMENT: {document.title}")
        print(f"SOURCE ID: {document.source_id}")
        print("*" * 80)

        original_length = len(document.text)

        # Cleaning BEFORE chunking
        document = clean_document(document)

        cleaned_length = len(document.text)

        print(f"Original characters: {original_length}")
        print(f"Cleaned characters: {cleaned_length}")

        chunks = chunker.chunk_document(document)

        total_chunks += len(chunks)

        print(f"Parent-Child chunks: {len(chunks)}")

        for chunk in chunks:

            print("\n" + "*" * 80)

            print(f"Chunk ID: {chunk.chunk_id}")

            print(f"Source ID: {chunk.source_id}")

            print(f"Text: {chunk.text}")

            print(f"Length: {len(chunk.text)}")

            print(f"Metadata: {chunk.metadata}")

    print("FINAL SUMMARY")
    print("*" * 80)

    print(f"Total documents: {len(documents)}")

    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()

