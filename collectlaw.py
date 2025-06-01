import joblib
import re
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

file_path = r"C:\Users\Eier\OneDrive\Dokument\DataScience\DeepLearning\chatbot\Inkomstskattelagen.txt"
with open(file_path, "r", encoding="utf-8") as file:
    Inkomstskattelagen = file.read()

# Skapa chunks
chunks = re.split(r'\d+ §', Inkomstskattelagen)
chunks = ["§" + chunk.strip() for chunk in chunks if chunk.strip()]
print(f"Antal chunks: {len(chunks)}")

batch_size = 16  # Justera efter din dators kapacitet
all_embeddings = []

print("Startar batch-loop")

for batch_start in range(0, len(chunks), batch_size):
    batch = chunks[batch_start:batch_start + batch_size]
    print(f"Processar batch {batch_start + 1} till {batch_start + len(batch)}")
    inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    all_embeddings.extend(embeddings.tolist())

print(f"Sparar embeddings, antal: {len(all_embeddings)}")
joblib.dump(all_embeddings, "embedded.pkl")
print("Klar! embedded.pkl skapad.")

emb_array = np.array(all_embeddings).astype("float32")
index = faiss.IndexFlatL2(emb_array.shape[1])
index.add(emb_array)
faiss.write_index(index, "faiss.index")
print("Klar! embedded.pkl och faiss.index skapade.")

index = faiss.read_index("faiss.index")       