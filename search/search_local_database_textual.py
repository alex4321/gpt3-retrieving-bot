from typing import List, Union, Tuple
import os
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import fuzzysearch
import json
from .search_interface import SearchInterface
from dataclasses import dataclass


@dataclass
class Paragraph:
    text: str
    processed_text: str
    document_name: str


__DEFAULT_SEARCH_PARAMS__ = {
    "max_l_dist": 3,
}


class SearchLocalDatabaseTextual(SearchInterface):
    def __init__(self, directory: str, language: str, top_results: int, search_params: Union[dict, None] = None) -> None:
        self.directory = directory
        self.language = language
        self.paragraphs = self.scan(self.directory)
        self.stopwords = stopwords.words(language)
        self.top_results = top_results
        if search_params is None:
            search_params = {}
        self.search_params = dict(__DEFAULT_SEARCH_PARAMS__, **search_params)
    
    def preprocess(self, text: str) -> str:
        tokens = wordpunct_tokenize(text.lower())
        tokens = filter(str.isalnum, tokens)
        tokens = filter(lambda token: token not in self.stopwords, tokens)
        return " ".join(tokens)

    def scan_directory(self, directory: str) -> List[Tuple[str, str]]:
        paragraph_files = []
        for document_path in os.listdir(directory):
            document_path_full = os.path.join(directory, document_path)
            if not os.path.isdir(document_path_full):
                continue
            for paragraph_path in os.listdir(document_path_full):
                paragraph_path_full = os.path.join(document_path_full, paragraph_path)
                if os.path.isdir(paragraph_path_full):
                    continue
                if not paragraph_path_full.lower().endswith(".txt"):
                    continue
                paragraph_files.append((document_path, paragraph_path_full))
        return paragraph_files

    def scan(self, directory: str) -> List[Paragraph]:
        paragraphs = []
        for document, filename in self.scan_directory(directory):
            with open(filename, "r", encoding="utf-8") as src:
                paragraph_json = json.load(src)
                paragraphs.append(Paragraph(
                    text=paragraph_json["text"],
                    processed_text=paragraph_json["processed_text"],
                    document_name=document,
                ))
        return paragraphs
    
    def update(self, document: str, amendment: str) -> None:
        document_full_path = os.path.join(self.directory, self.preprocess(document))
        if not os.path.exists(document_full_path):
            os.makedirs(document_full_path)
        
        paragraph_count = len(self.scan_directory(self.directory))

        new_paragraphs_text = amendment.split("\n")
        new_paragraphs_text = [item.strip() for item in new_paragraphs_text]
        new_paragraphs_processed = [self.preprocess(item) for item in new_paragraphs_text]
        new_paragraphs = [
            Paragraph(text=text, processed_text=processed_text, document_name=document)
            for text, processed_text in zip(new_paragraphs_text, new_paragraphs_processed)
            if processed_text != ""
        ]

        for i, paragraph in enumerate(new_paragraphs):
            new_paragraph_fname = f"PARAGRAPH_{i + 1 + paragraph_count}.TXT"
            new_paragraph_fname_full = os.path.join(document_full_path, new_paragraph_fname)
            with open(new_paragraph_fname_full, "w", encoding="utf-8") as target:
                json.dump(
                    {
                        "text": paragraph.text,
                        "processed_text": paragraph.processed_text,
                    },
                    target
                )
        
        self.paragraphs = self.scan(self.directory)
    
    def search(self, query: str) -> List[str]:
        query = self.preprocess(query)
        paragraph_distances = []
        for paragraph in self.paragraphs:
            paragraph_matches = fuzzysearch.find_near_matches(query, paragraph.processed_text, **self.search_params)
            if len(paragraph_matches) > 0:
                distances = map(lambda match: match.dist, paragraph_matches)
                min_distance = min(distances)
                paragraph_distances.append((min_distance, paragraph))
        paragraph_distances = sorted(paragraph_distances, key=lambda pair: pair[0])
        paragraphs = [
            paragraph
            for _, paragraph in paragraph_distances
        ]
        paragraphs = paragraphs[:self.top_results]
        return [
            paragraph.document_name + " : " + paragraph.text
            for paragraph in paragraphs
        ]
