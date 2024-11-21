import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors 
    of an inner product space that measures the cosine of the angle between them.

    Parameters:
    vec1 (array-like): First vector.
    vec2 (array-like): Second vector.

    Returns:
    float: Cosine similarity between vec1 and vec2. The value is between -1 and 1, 
           where 1 indicates that the vectors are identical, 0 indicates orthogonality, 
           and -1 indicates that the vectors are diametrically opposed.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)