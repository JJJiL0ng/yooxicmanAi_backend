# scripts/init_pinecone.py
import pinecone
from yaml import safe_load

def init_pinecone():
    with open("config/config.yaml") as f:
        config = safe_load(f)
    
    pinecone.init(
        api_key=config['vector_store']['pinecone']['api_key'],
        environment=config['vector_store']['pinecone']['environment']
    )
    
    if config['vector_store']['pinecone']['index_name'] not in pinecone.list_indexes():
        pinecone.create_index(
            name=config['vector_store']['pinecone']['index_name'],
            dimension=config['vector_store']['pinecone']['dimension']
        )

if __name__ == "__main__":
    init_pinecone()