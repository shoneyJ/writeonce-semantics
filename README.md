# writeonce-semantics

Search the research documents for semantic similarities, return a prompt responce using a open source LLM.

1. embed the documents in json and md files to vector database [https://milvus.io/](https://milvus.io/)

2. Use opensource LLM Mistral to load the embeddings and repond with Retrieval Augmented Generation to repond to chatbot user.

# Development Setup

## Milvus Lite

```
pip install -r requirements.txt
```
[https://sbert.net/](https://sbert.net/)

sentence-transformer for embedding the documents


Mistral for LLM