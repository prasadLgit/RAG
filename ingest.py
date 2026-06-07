# ingest.py
#pip install pymupdf tiktoken supabase openai tqdm python-dotenv
import time
import os, uuid, re
import fitz  # PyMuPDF
import tiktoken
from supabase import create_client, Client
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# ---- Load environment
load_dotenv(find_dotenv(usecwd=True))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY_1 = os.environ["GEMINI_API_KEY_1"]
GEMINI_API_KEY_2 = os.environ["GEMINI_API_KEY_2"]

# ---- Config
PDF_PATH = "human-nutrition-text.pdf"
DOC_ID = "nutrition-v1"               # keep this STABLE to avoid duplicates
EMBED_MODEL = "gemini-embedding-001"  # 1536 dims -> matches your table
BATCH_EMBED = 50
BATCH_INSERT = 200

# Sentence chunking params
SENTS_PER_CHUNK = 20
SENT_OVERLAP = 2
MAX_TOKENS = 1300     # safety cap (trim if 10 sentences are too long)
MIN_TOKENS = 50      # skip very tiny fragments

enc = tiktoken.get_encoding("cl100k_base")  # matches OpenAI embeddings tokenizer

def clean_text(t: str) -> str:
    # normalize whitespace and fix hyphenation across line breaks
    t = t.replace("\r", " ")
    t = re.sub(r"-\s*\n\s*", "", t)     # join "nutri-\n tion" => "nutrition"
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("\n", " ").strip()
    return t

def split_sentences(text: str):
    # simple sentence splitter (good for prose)
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_page_by_sentences(text: str,
                            sents_per_chunk: int = SENTS_PER_CHUNK,
                            overlap: int = SENT_OVERLAP,
                            max_tokens: int = MAX_TOKENS,
                            min_tokens: int = MIN_TOKENS):
    sents = split_sentences(text)
    i = 0
    step = max(1, sents_per_chunk - overlap)
    while i < len(sents):
        piece = sents[i:i + sents_per_chunk]
        if not piece:
            break
        chunk = " ".join(piece)

        # enforce token ceiling
        ids = enc.encode(chunk)
        while max_tokens and len(ids) > max_tokens and len(piece) > 1:
            piece = piece[:-1]
            chunk = " ".join(piece)
            ids = enc.encode(chunk)

        if len(ids) >= min_tokens:
            yield chunk
        i += step

def pdf_pages(path: str):
    """Yield (page_number_1based, cleaned_text)."""
    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            txt = doc[i].get_text("text") or ""
            yield (i + 1, clean_text(txt))
    finally:
        doc.close()

def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    client1 = genai.Client(api_key=GEMINI_API_KEY_1)
    client2 = genai.Client(api_key=GEMINI_API_KEY_2)
    # Optional: keep the table clean for this document
    sb.table("chunks").delete().eq("doc_id", DOC_ID).execute()

    print("Reading PDF by pages...")
    pages = list(pdf_pages(PDF_PATH))

    # Build chunks with page metadata
    inputs, metas = [], []
    print("Chunking (10 sentences per chunk, 2 overlap)...")
    for page, text in pages:
        if not text:
            continue
        for chunk in chunk_page_by_sentences(text):
            inputs.append(chunk)
            metas.append({"page": page, "source": PDF_PATH})

    print(f"✅ Built {len(inputs)} chunks from {PDF_PATH}")

    # Generate embeddings
    vectors = []
    print("Generating embeddings...")
    for i in tqdm(range(0, len(inputs), BATCH_EMBED), desc="Embedding"):
        batch = inputs[i:i + BATCH_EMBED]
        while True:
          try:
            if (i//BATCH_EMBED)%2 == 1:
              resp = client1.models.embed_content(
              model=EMBED_MODEL,
              contents=batch,
              config={
                       "output_dimensionality": 1536
                     }
            )
            else:
              resp = client2.models.embed_content(
              model=EMBED_MODEL,
              contents=batch,
              config={
                     "output_dimensionality": 1536
                    }
            )

            vectors.extend([e.values for e in resp.embeddings])
            break

          except Exception as e:
            if "429" in str(e):
               print("Rate limit hit. Waiting 30 seconds...")
               time.sleep(30)
            else:
               raise
    
    # Prepare rows
    rows = []
    for idx, (content, emb, meta) in enumerate(zip(inputs, vectors, metas)):
        rows.append({
            "doc_id": DOC_ID,
            "chunk_index": idx,
            "content": content,
            "metadata": meta,      # contains {source, page}
            "embedding": emb
        })

    print("Uploading to Supabase...")
    for j in tqdm(range(0, len(rows), BATCH_INSERT), desc="Uploading"):
        sb.table("chunks").insert(rows[j:j+BATCH_INSERT]).execute()

    print(f"🎉 Done! Inserted {len(rows)} chunks for doc_id={DOC_ID}")

if __name__ == "__main__":
    main()
