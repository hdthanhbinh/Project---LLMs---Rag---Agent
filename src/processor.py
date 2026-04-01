from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path):
	"""Load a document file and return a list of LangChain Document objects."""
	path = Path(file_path)
	suffix = path.suffix.lower()

	if suffix == ".pdf":
		loader = PDFPlumberLoader(str(path))
	elif suffix == ".docx":
		loader = Docx2txtLoader(str(path))
	elif suffix == ".txt":
		loader = TextLoader(str(path), encoding="utf-8")
	else:
		raise ValueError(f"Unsupported file format: {suffix}")

	return loader.load()


def split_text(documents, chunk_size=1000, chunk_overlap=100):
	"""Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	return text_splitter.split_documents(documents)


def get_embedding_model():
	"""Create and return a configured HuggingFace embedding model."""
	return HuggingFaceEmbeddings(
		model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
		model_kwargs={"device": "cpu"},
		encode_kwargs={"normalize_embeddings": True},
	)


def process_pipeline(file_path, chunk_size=1000, chunk_overlap=100):
	"""Run the full processing pipeline and return chunks with embedding model."""
	docs = load_document(file_path)
	chunks = split_text(docs, chunk_size, chunk_overlap)
	return {
		"chunks": [chunk.page_content for chunk in chunks],
		"embedding_model": get_embedding_model(),
	}


if __name__ == "__main__":
	test_file = Path(__file__).with_name("test.txt")
	sample_text = (
		"LangChain pipeline test content. " * 80
		+ "This block is intentionally long so text splitting can produce multiple chunks. " * 20
	)
	test_file.write_text(sample_text, encoding="utf-8")
	success = False

	try:
		result = process_pipeline(str(test_file), chunk_size=500, chunk_overlap=50)
		chunk_list = result["chunks"]

		print(f"Tong so chunks duoc tao: {len(chunk_list)}")
		if chunk_list:
			print("Noi dung chunk dau tien:")
			print(chunk_list[0])
		else:
			print("Khong co chunk nao duoc tao.")
		success = True
	finally:
		if success and test_file.exists():
			test_file.unlink()
			print("Da xoa file test.txt sau khi test thanh cong.")
