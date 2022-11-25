"""
Semantic search through local DB by embeddings similarity
"""
from dataclasses import dataclass
import json
import os
import pickle
from typing import List, Union, Tuple
import nltk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .nn_config import NNConfig
from .search_interface import SearchInterface
from .utils import get_sentence_transformer, is_local_request


@dataclass
class Paragraph:
    """
    Document paragraphs
    """
    document_name: str
    text: str
    sentence_embeddings: np.ndarray


__DEFAULT_KNN_PARAMS__ = {
    "weights": "distance"
}


class SearchLocalDatabaseSemantic(SearchInterface):
    """
    Semantic search through local DB by embeddings similarity
    """
    def __init__(self, directory: str, model: str, top_n: int, embedder_params: NNConfig,
                 knn_params: Union[dict, None] = None) -> None:
        super().__init__()
        self.embedder_params = embedder_params
        self.model = get_sentence_transformer(model, device=self.embedder_params.device)
        self.encoder_dim = self.model.encode([""]).shape[-1]
        self.directory = directory
        self.top_n = top_n
        if knn_params is None:
            knn_params = {}
        self.knn_params = dict(__DEFAULT_KNN_PARAMS__, **knn_params)
        self.paragraphs, self.knn = self._scan(self.directory, False, self.top_n, self.knn_params)

    def _embeddings_to_string(self, embeddings: np.ndarray) -> str:
        embeddings_bytes = embeddings.astype(np.float32).tobytes()
        embeddings_bytes_ascii = []
        top_bit = b"\x80"[0]
        top_bit_others = b"\x7f"[0]
        for embeddings_byte in embeddings_bytes:
            embeddings_byte_part1 = (embeddings_byte & top_bit) >> 7
            embeddings_byte_part2 = embeddings_byte & top_bit_others
            embeddings_bytes_ascii.append(embeddings_byte_part1)
            embeddings_bytes_ascii.append(embeddings_byte_part2)
        return bytes(embeddings_bytes_ascii).decode("ascii")

    def _embeddings_from_string(self, embeddings_data: str) -> np.ndarray:
        embeddings_bytes = embeddings_data.encode("ascii")
        embeddings_bytes_parsed_values = []
        for i in range(len(embeddings_bytes) // 2):
            embeddings_byte_part1 = embeddings_bytes[i * 2]
            embeddings_byte_part2 = embeddings_bytes[i * 2 + 1]
            embeddings_byte = (embeddings_byte_part1 << 7) | embeddings_byte_part2
            embeddings_bytes_parsed_values.append(embeddings_byte)
        embeddings_bytes_parsed = bytes(embeddings_bytes_parsed_values)
        return np.frombuffer(bytes(embeddings_bytes_parsed), dtype=np.float32).reshape([-1, self.encoder_dim])

    def _scan(self, directory: str, rebuild_knn: bool, top_n: int, knn_params: dict) \
        -> Tuple[List[Paragraph], Union[KNeighborsClassifier, None]]:
        paragraphs = []
        for document_name in os.listdir(directory):
            document_name_full = os.path.join(directory, document_name)
            if os.path.isdir(document_name_full):
                continue
            if not document_name_full.lower().endswith(".json"):
                continue
            with open(document_name_full, "r", encoding="utf-8") as src:
                document_json = json.load(src)
                document_name_original = document_json["name"]
                for paragraph in document_json["paragraphs"]:
                    paragraphs.append(Paragraph(
                        document_name=document_name_original,
                        text=paragraph["text"],
                        sentence_embeddings=self._embeddings_from_string(paragraph["sentence_embeddings"]),
                    ))
        knn_fname = os.path.join(directory, "knn.pkl")
        if (not os.path.exists(knn_fname)) or rebuild_knn:
            knn = self._build_knn(paragraphs, top_n, knn_params)
            with open(knn_fname, "wb") as target:
                pickle.dump(knn, target)
        else:
            with open(knn_fname, "rb") as src:
                knn = pickle.load(src)
        return paragraphs, knn

    def _build_knn(self, paragraphs: List[Paragraph], top_n: int, knn_params: dict) \
        -> Union[KNeighborsClassifier, None]:
        if len(paragraphs) == 0:
            return None
        knn_x = []
        knn_y = []
        for i, paragraph in enumerate(paragraphs):
            for j in range(paragraph.sentence_embeddings.shape[0]):
                knn_x.append(paragraph.sentence_embeddings[j])
                knn_y.append(i)
        knn = KNeighborsClassifier(n_neighbors=top_n, **knn_params)
        knn.fit(np.array(knn_x), np.array(knn_y))
        return knn

    def _process_document_name(self, document: str) -> str:
        return "".join([
            c
            for c in document.lower()
            if c.isalnum()
        ])

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.embedder_params.batch_size,
            normalize_embeddings=True
        )

    def update(self, document: str, amendment: str) -> None:
        document_fname = self._process_document_name(document) + ".json"
        document_fname = os.path.join(self.directory, document_fname)
        if os.path.exists(document_fname):
            with open(document_fname, "r", encoding="utf-8") as src:
                document_data = json.load(src)
        else:
            document_data = {
                "name": document,
                "paragraphs": [],
            }

        new_paragraphs = amendment.split("\n")
        new_paragraphs = [item.strip() for item in new_paragraphs]
        new_paragraphs = [item for item in new_paragraphs if item != ""]
        for new_paragraph in new_paragraphs:
            sentences = nltk.sent_tokenize(new_paragraph)
            embeddings = self._encode(sentences)
            document_data["paragraphs"].append({
                "text": new_paragraph,
                "sentence_embeddings": self._embeddings_to_string(embeddings),
            })

        with open(document_fname, "w", encoding="utf-8") as target:
            json.dump(document_data, target)

        self.paragraphs, self.knn = self._scan(self.directory, True, self.top_n, self.knn_params)

    def search(self, query: str) -> List[str]:
        _, query = is_local_request(query)
        if self.knn is None:
            return [""]
        query_embeddings = self._encode([query])
        paragraph_indices_proba = self.knn.predict_proba(query_embeddings)[0]
        top_indices = (-paragraph_indices_proba).argsort()[:self.top_n]
        top_paragraphs_indices = [self.knn.classes_[idx] for idx in top_indices]
        result = []
        for idx in top_paragraphs_indices:
            paragraph = self.paragraphs[idx]
            result.append(f"{paragraph.document_name} : {paragraph.text}")
        return result
