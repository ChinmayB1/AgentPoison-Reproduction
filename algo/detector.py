import numpy as np
import torch

class GeometricDetector:
    def __init__(self, threshold_similarity=0.8, threshold_centroid=0.5, threshold_repetitive=0.6):
        self.threshold_similarity = threshold_similarity
        self.threshold_centroid = threshold_centroid
        self.threshold_repetitive = threshold_repetitive

    def compute_pairwise_similarities(self, embeddings):
        """
        embeddings: numpy array of shape (K, D) where K is number of retrieved documents
        """
        if len(embeddings) <= 1:
            return 0.0
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = embeddings / (norms + 1e-8)
        # Cosine similarity matrix
        sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        # Get upper triangle elements excluding diagonal
        indices = np.triu_indices(len(embeddings), k=1)
        pairwise_sims = sim_matrix[indices]
        return np.mean(pairwise_sims)

    def compute_centroid_concentration(self, embeddings):
        """
        embeddings: numpy array of shape (K, D)
        """
        if len(embeddings) == 0:
            return 0.0
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return np.mean(distances)

    def compute_text_repetitiveness(self, texts):
        """
        texts: list of strings of the retrieved documents
        """
        if len(texts) <= 1:
            return 0.0
        # Compute pairwise Jaccard similarities of word sets
        word_sets = [set(str(t).lower().split()) for t in texts]
        jaccard_sims = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                set_i, set_j = word_sets[i], word_sets[j]
                if not set_i or not set_j:
                    jaccard_sims.append(0.0)
                else:
                    jaccard_sims.append(len(set_i & set_j) / len(set_i | set_j))
        return np.mean(jaccard_sims)

    def detect(self, embeddings, texts):
        """
        Returns a dict of metrics and a boolean flag indicating if it is suspicious
        """
        # Convert embeddings to numpy if they are torch tensors
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().detach().numpy()
        
        pairwise_sim = self.compute_pairwise_similarities(embeddings)
        centroid_dist = self.compute_centroid_concentration(embeddings)
        text_rep = self.compute_text_repetitiveness(texts)
        
        # A simple detection rule: if similarity is high or text repetitiveness is high, it is suspicious
        is_suspicious = (pairwise_sim > self.threshold_similarity) or (text_rep > self.threshold_repetitive)
        
        return {
            "pairwise_similarity": float(pairwise_sim),
            "centroid_concentration": float(centroid_dist),
            "text_repetitiveness": float(text_rep),
            "is_suspicious": bool(is_suspicious)
        }
