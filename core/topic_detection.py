from bertopic import BERTopic
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np

from core.log import logger

class TopicModeling:
    def __init__(self, method="bertopic", num_topics=None):
        """
        Initialize the topic modeling method.
        
        Args:
            method (str): "bertopic", "lda", or "nmf".
            num_topics (int): The number of topics
        """
        self.method = method
        if method == "bertopic" and num_topics is None:
            num_topics = 18
        if method == "lda" and num_topics is None:
            num_topics = 60
        elif method == "nmf" and num_topics is None:
            num_topics = 90
        self.num_topics = num_topics
        self.model = None
        self.fitted = False

    def fit_transform(self, sentences):
        """
        Train the topic model and transform sentences into topic representations.

        Args:
            sentences (list): List of textual sentences.

        Returns:
            np.ndarray: One-hot encoded topic vectors.
        """
        if self.method == "bertopic":
            if not self.fitted:
                # Use GPU-based embedding model for BERTopic
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

                # Initialize BERTopic with a fixed number of topics
                self.model = BERTopic(
                    nr_topics=self.num_topics,
                    embedding_model=embedding_model,
                    verbose=True,
                    top_n_words=20
                
                )
                topics, _ = self.model.fit_transform(sentences)
                self.fitted = True

                # Ensure topics are within a valid range
                self.num_topics = max(topics) + 1  # Update num_topics dynamically

                # Free GPU memory
                del embedding_model  # Delete the embedding model
                torch.cuda.empty_cache()  # Clear unused GPU memory
                gc.collect()  # Run garbage collector

                logger.debug(f"Topic indices shape: {np.array(topics).shape}")
                logger.debug(f"Max topic index: {max(topics)}")
                logger.debug(f"Expected num_topics: {self.num_topics}")

                return self.get_topic_vectors(topics)
            else:
                # If we call fit_transform again by mistake, either raise an error or just transform
                raise RuntimeError("This model is already fitted. Call .transform() for new data.")

        # Use CountVectorizer for LDA, TfidfVectorizer for NMF
        vectorizer = CountVectorizer() if self.method == "lda" else TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # Train LDA or NMF
        if self.method == "lda":
            self.model = LDA(n_components=self.num_topics, random_state=42)
        elif self.method == "nmf":
            self.model = NMF(n_components=self.num_topics, random_state=42)

        # Get topic distributions (probabilities)
        topic_probs = self.model.fit_transform(X)

        # Convert probabilities to topic indices (argmax)
        topic_indices = np.argmax(topic_probs, axis=1)

        return self.get_topic_vectors(topic_indices)

    def transform(self, sentences):
        if self.method == "bertopic":
            topics, _ = self.model.transform(sentences)
            return self.get_topic_vectors(topics)

    def get_topic_vectors(self, topics):
        """
        Convert topic indices into one-hot encoded vectors.

        Args:
            topics (list or np.ndarray): Topic indices.

        Returns:
            np.ndarray: One-hot encoded topic representation.
        """
        num_sentences = len(topics)

        topic_vectors = np.zeros((num_sentences, self.num_topics))

        for i, topic in enumerate(topics):
            if 0 <= topic < self.num_topics:  # Ensure topic index is within bounds
                topic_vectors[i, topic] = 1  # One-hot encode the topic assignment
            else:
                continue  # Ignore invalid topics

        return topic_vectors