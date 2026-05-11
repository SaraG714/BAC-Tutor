import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
print("API key cargada:", key[:10] + "..." if key else "NO ENCONTRADA")

client = genai.Client(api_key=key)
result = client.models.embed_content(model="gemini-embedding-001", contents="prueba")
print("Conexion OK, embedding size:", len(result.embeddings[0].values))
