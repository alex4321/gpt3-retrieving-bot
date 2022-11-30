from typing import Tuple, Union, List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from .search_interface import SearchInterface
from sentence_transformers import SentenceTransformer
import os


_KNN_FNAME = "knn.pkl"
_EMBEDDINGS_FNAME = "embeddings.pkl"
_TEXTS_FNAME = "texts.pkl"


class SearchLocalDatabase(SearchInterface):
    def __init__(self, model: SentenceTransformer, data_directory: str, top_n: int, encoder_batch_size: int) -> None:
        self.model = model
        self.top_n = top_n
        self.data_directory = data_directory
        self.encoder_batch_size = encoder_batch_size
        self._knn, self._embeddings, self._texts = self._load_data()

    def _load_pickle(self, fname: str) -> object:
        with open(fname, "rb") as src:
            return pickle.load(src)

    def _save_pickle(self, fname: str, data: object) -> None:
        with open(fname, "wb") as target:
            pickle.dump(data, target)

    def _load_data(self) -> Tuple[\
        Union[KNeighborsClassifier, None], \
        Union[np.ndarray, None], \
        Union[np.ndarray, None] \
        ]:
        os.makedirs(self.data_directory, exist_ok=True)
        knn_path = os.path.join(self.data_directory, _KNN_FNAME)
        embeddings_path = os.path.join(self.data_directory, _EMBEDDINGS_FNAME)
        texts_path = os.path.join(self.data_directory, _TEXTS_FNAME)
        paths = [self.data_directory, knn_path, embeddings_path, texts_path]
        for path in paths:
            if not os.path.exists(path):
                return None, None, None
        knn = self._load_pickle(os.path.join(self.data_directory, _KNN_FNAME))
        embeddings = self._load_pickle(os.path.join(self.data_directory, _EMBEDDINGS_FNAME))
        texts = self._load_pickle(os.path.join(self.data_directory, _TEXTS_FNAME))
        return knn, embeddings, texts

    def add_documents(self, documents: List[str]) -> None:
        documents = [document.strip() for document in documents]
        document_embeddings = self.model.encode(documents, batch_size=self.encoder_batch_size)
        if self._embeddings is not None:
            new_embeddings = np.vstack([self._embeddings, document_embeddings])
        else:
            new_embeddings = document_embeddings
        if self._texts is not None:
            new_texts = np.vstack([self._texts, documents])
        else:
            new_texts = documents
        knn = KNeighborsClassifier(n_neighbors=min([self.top_n, len(new_embeddings)]))
        knn.fit(new_embeddings, np.arange(len(new_embeddings)))
        self._save_pickle(os.path.join(self.data_directory, _KNN_FNAME), knn)
        self._save_pickle(os.path.join(self.data_directory, _EMBEDDINGS_FNAME), new_embeddings)
        self._save_pickle(os.path.join(self.data_directory, _TEXTS_FNAME), new_texts)
        self._knn, self._embeddings, self._texts = self._load_data()

    def search(self, query: str) -> List[str]:
        if not self._knn:
            return []
        embeddings = self.model.encode([query], batch_size=self.encoder_batch_size)
        scores = self._knn.predict_proba(embeddings)[0]
        top_indices = (-scores).argsort()[:self.top_n]
        return [
            self._texts[index]
            for index in top_indices
        ]
