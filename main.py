from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]


embedder = SentenceTransformer("all-MiniLM-L6-v2")

docVectors = embedder.encode(docs)


data = [ {"id": i, "vector": docVectors[i], "text": docs[i], "subject": "history"} for i in range(len(embeddings)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

# This will exclude any text in "history" subject despite close to the query vector.

searchVector = embedder.encode('Who researched in Artificial intelligence')

res = client.search(
    collection_name="demo_collection",
    data=[searchVector],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)
print(res)

retrieved_texts = [hit.entity.get("text") for hit in res[0]]

# a query that retrieves all entities matching filter expressions.
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
##print(res)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

context = "\n\n".join(retrieved_texts)
prompt = f"Answer the question using the context:\n\n{context}"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = llm_model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# delete
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
##print(res)