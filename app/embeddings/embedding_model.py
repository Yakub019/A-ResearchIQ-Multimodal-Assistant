
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


if __name__ == "__main__":

    embeddings = get_embedding_model()

    text = "Retrieval Augmented Generation improves LLM responses."

    vector = embeddings.embed_query(text)

    print("\n========== EMBEDDING TEST ==========")
    print("Model:", MODEL_NAME)
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])
    print("Embedding generated successfully!")
