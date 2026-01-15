from sentence_transformers import SentenceTransformer
from collections import Counter
import faiss
import numpy as np

class IntentEmbedder:
    def __init__(self, intent_examples, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.intent_labels = []
        self.examples = []
        self.embeddings = None
        self.index = None
        
        self._embed_all(intent_examples)
    
    def _embed_all(self, intent_examples):
        for intent, examples in intent_examples.items():
            self.examples.extend(examples)
            self.intent_labels.extend([intent] * len(examples))
        
        self.embeddings = self.model.encode(
            self.examples,
            normalize_embeddings=True,  
            convert_to_numpy=True
        )
        
        self._build_index()
    def _build_index(self):
        """Build FAISS index"""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
    def predict(self, query, k=3, threshold=0.6):
        """Predict intent for a query"""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        if distances[0][0] < threshold:
            return "unknown", distances[0][0], []
        
        matches = [
            {
                'intent': self.intent_labels[idx],
                'example': self.examples[idx],
                'score': float(dist)
            }
            for dist, idx in zip(distances[0], indices[0])
        ]
        
        intent_votes = [self.intent_labels[idx] for idx in indices[0]]
        predicted_intent = Counter(intent_votes).most_common(1)[0][0]
        
        return predicted_intent, distances[0][0], matches