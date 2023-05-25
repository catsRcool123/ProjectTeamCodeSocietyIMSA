from collections import Counter
from itertools import combinations
from typing import List
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import math

class TextSummarizer:
    def __init__(self):
        self.stop_words = stopwords.words("english")

    def read(self, file_path: str) -> List[str]:
        with open(file_path, "r") as f:
            text = f.read()
        sentences = text.split(". ")
        return sentences

    def similarity(self, sent1: List[str], sent2: List[str]) -> float:
        sent1_counter = Counter(w.lower() for w in sent1 if w.lower() not in self.stop_words)
        sent2_counter = Counter(w.lower() for w in sent2 if w.lower() not in self.stop_words)
        all_words = set(sent1_counter.keys()) | set(sent2_counter.keys())
        vec1 = [sent1_counter.get(w, 0) for w in all_words]
        vec2 = [sent2_counter.get(w, 0) for w in all_words]
        return 1 - cosine_distance(vec1, vec2)

    def matrix(self, sentences: List[List[str]]) -> List[List[float]]:
        n = len(sentences)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        for i, j in combinations(range(n), 2):
            similarity_matrix[i][j] = similarity_matrix[j][i] = self.similarity(
                sentences[i], sentences[j]
            )
        return similarity_matrix

    def summary(self, shortness, file_path: str) -> str:
        sentences = self.read(file_path)
        if shortness == 1:
          short = 0.10
        elif shortness == 2:
          short = 0.25
        else:
          short = 0.50
        num_sentences = max(1, math.ceil(len(sentences) * short))
        sentence_vectors = self.matrix([s.split() for s in sentences])
        scores = {i: sum(sentence_vectors[i]) for i in range(len(sentences))}
        ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked_indices[:num_sentences])
        info = ". ".join([sentences[i] for i in top_indices])
        return info, num_sentences