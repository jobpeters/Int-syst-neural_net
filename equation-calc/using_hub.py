import tensorflow as tf
import tensorflow_hub as hub



if __name__ == "__main__":
    module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    embed = hub.KerasLayer(module_url)
    embeddings = embed(["A long sentence.", "single-word",
                  "http://example.com"])
    print(embeddings.shape)  #(3,128)