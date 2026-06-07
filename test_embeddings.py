# probe_matches.py
import os, textwrap
from supabase import create_client, Client
from google import genai
from dotenv import load_dotenv, find_dotenv

# ---- Load env
load_dotenv(find_dotenv(usecwd=True))
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]  # using key 1, same as ingest.py

EMBED_MODEL = "gemini-embedding-001"  # 1536-dims; matches your table
PDF_PATH = "human-nutrition-text.pdf"  # used as a filter in metadata
TOP_K = 3

queries = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
    "What are micronutrients?"
]

def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    client = genai.Client(api_key=GEMINI_API_KEY)

    for q in queries:
        # embed query
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=q,
            config={"output_dimensionality": 1536}
        )
        e = resp.embeddings[0].values

        # call your RPC with a metadata filter to this PDF
        resp = sb.rpc("match_documents", {
            "query_embedding": e,
            "match_count": TOP_K,
            "filter": {"source": PDF_PATH}
        }).execute()

        rows = resp.data or []
        print("\n" + "="*90)
        print(f"QUERY: {q}")
        if not rows:
            print("  (no matches)")
            continue

        for rank, r in enumerate(rows, start=1):
            page = (r.get("metadata") or {}).get("page", "?")
            sim  = r.get("similarity", None)
            sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "?"
            preview = textwrap.shorten(r.get("content","").replace("\n"," "), width=160)
            print(f"  [{rank}] page {page}  sim={sim_str}  chunk_index={r.get('chunk_index')}")
            print(f"      {preview}")

if __name__ == "__main__":
    main()
