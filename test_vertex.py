import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
import os
import numpy as np


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/Users/ashmac/.config/gcloud/application_default_credentials.json"
)

# TODO(developer): Update & uncomment line below
PROJECT_ID = "gemini-copilot-testing"
vertexai.init(project=PROJECT_ID, location="us-central1")


model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
image = Image.load_from_file(
    "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)

embeddings = model.get_embeddings(
    image=image,
    contextual_text="",
    dimension=1408,
)
print(f"Image Embedding: {embeddings.image_embedding[:10]}")
print(f"Text Embedding: {embeddings.text_embedding[:10]}")

text_embedding = embeddings.text_embedding
image_embedding = embeddings.image_embedding

# dot product between image and text embeddings


# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     magnitude_vec1 = np.linalg.norm(vec1)
#     magnitude_vec2 = np.linalg.norm(vec2)
#     similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
#     return similarity


# similarity = cosine_similarity(image_embedding, text_embedding)
# print(f"Similarity: {similarity}")

"""
Image Embedding: [-0.012314274, 0.072718516, 0.000201684801, 0.0308900643, -0.00855009258, -0.0374744199, 0.00490623713, -0.00600622687, 0.0266232714, 0.00892610382]
Text Embedding: [-0.00434634695, 0.0272456184, -0.00995295309, 0.0097399326, -0.016157493, 0.00231144414, -0.00243385951, 0.0101483958, -0.00551879546, 0.00378345908]
"""
