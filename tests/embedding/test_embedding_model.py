from app.ingestion.loaders import FileParser
from app.embeddings.embedding_model import BGE_M3_Embedding


def main():
    documents_path = "data/documents"

    parser = FileParser()
    documents = parser.load_directory(documents_path)

    print(f"Documents loaded: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    embedder = BGE_M3_Embedding()

    texts = []
    for document in documents:
        if document.text.strip():
            texts.append(document.text)

    vectors = embedder.embed(texts)

    print(f"Texts embedded: {len(texts)}")
    print(f"Vectors created: {len(vectors)}")

    if vectors:
        print(f"Embedding size: {len(vectors[0])}")
        print("First vector:", vectors[0][:10])


if __name__ == "__main__":
    main()