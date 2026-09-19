from app.ingestion.loaders import FileParser
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import ParentChildChunker
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.models.evidence_models import SourceDocument


def main():
    documents_path = "data/documents"

    parser = FileParser()
    loaded_files = parser.load_directory(
        documents_path
    )

    documents = [
        file
        for file in loaded_files
        if isinstance(file, SourceDocument)
    ]

    print(f"Documents loaded: {len(documents)}")

    if not documents:
        print("No text documents found.")
        return

    cleaned_documents = [
        clean_document(document)
        for document in documents
    ]

    chunker = ParentChildChunker()

    chunks = chunker.chunk_documents(
        cleaned_documents
    )

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks created.")
        return

    embedder = BGE_M3_Embedding()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    vectors = embedder.embed(texts)

    print(f"Texts embedded: {len(texts)}")
    print(f"Vectors created: {len(vectors)}")

    if vectors:
        print(
            f"Embedding size: {len(vectors[0])}"
        )

        print(
            "First vector:",
            vectors[0][:10]
        )


if __name__ == "__main__":
    main()