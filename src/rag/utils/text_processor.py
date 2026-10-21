from langchain_text_splitters import RecursiveCharacterTextSplitter

import hashlib

def split_document(text: str, document_id: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        seprators=["\n\n", "\n", "。", "!", "，", ".", ",", " ", ""]
    )

    chunks = splitter.split_text(text)
    result = []

    for index, chunk_text in enumerate(chunks):
        hash_input = f"{document_id}_{chunk_text}".encode("utf-8")
        chunk_hash = hashlib.md5(hash_input).hexdigest()[:8]
        fin_chunk_id = f"{document_id}_{chunk_hash}"

        result.append({
            "document_id": document_id,
            "chunk_id": fin_chunk_id,
            "text": chunk_text,
        })