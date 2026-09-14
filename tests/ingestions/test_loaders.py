
print("FILE PARSER TEST IS RUNNING")
print("=" * 60)

from pathlib import Path
from PIL import Image

from app.ingestion.loaders import FileParser


parser = FileParser()

# TEST DOCUMENTS
print("\nDOCUMENTS")
print("*" * 60)

documents = parser.load_directory(
    "data/documents" 
)

print(f"Loaded documents: {len(documents)}")


for document in documents:

    print("\nSource ID:", document.source_id)
    print("Title:", document.title)
    print("Media type:", document.media_type)

    print("Text length:", len(document.text))

    print("\nText preview:")
    print(document.text[:500])

    print("-" * 60)

# TEST IMAGES
print("\nIMAGES")
print("*" * 60)

image_directory = Path(
    "data/images"
)

images = []

for image_path in image_directory.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in parser.IMAGE_EXTENSIONS:
        continue

    try:

        image = parser.load_file(
            str(image_path)
        )

        images.append(image)

    except ValueError as error:

        print(
            f"Skipping {image_path.name}: {error}"
        )


print(f"Loaded images: {len(images)}")


for image in images:

    print("\nSource:", image["metadata"]["source"])
    print("File type:", image["metadata"]["file_type"])
    print("Extension:", image["metadata"]["extension"])
    print("Image path:", image["image_path"])

    # Check that the image can actually be opened
    img = Image.open(
        image["image_path"]
    )

    print("Image size:", img.size)
    print("Image mode:", img.mode)

    print("-" * 60)


print("\nFILE PARSER TEST FINISHED")

