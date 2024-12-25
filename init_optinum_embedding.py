from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
OptimumEmbedding.create_and_save_optimum_model(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", './sentence-transformers'
)