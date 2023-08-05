import os
from collections import defaultdict

from gensim.parsing.preprocessing import preprocess_string


class TextDocument:
    def __init__(self, document_location, doc_label, doc_description=None):
        self.doc_path = document_location
        self.doc_label = doc_label
        self.doc_description = doc_description
        self.token_count = defaultdict(int)
        if doc_description is None:
            self.doc_description = os.path.basename(self.doc_path)

    @property
    def doc_path(self):
        return self._doc_path

    @doc_path.setter
    def doc_path(self, path):
        if os.path.isfile(path):
            self._doc_path = path
        else:
            raise ValueError(
                f"The document location is wrong, file {path} does not exist!"
            )

    @property
    def vocabulary(self):
        return self.token_count.keys()

    @property
    def tokens_count(self):
        return len(self.vocabulary)

    def read_document(self, gensim_custom_tokenizer=None):
        self.token_count = defaultdict(int)
        with open(self.doc_path, "r") as f:
            doc_body = f.read()

            if gensim_custom_tokenizer is not None:
                tokens = preprocess_string(doc_body, gensim_custom_tokenizer)
                for token in tokens:
                    self.token_count[token] += 1

                return tokens

        return doc_body

    def read_document_gen(self, batch_size=128, gensim_custom_tokenizer=None):
        self.token_count = defaultdict(int)
        with open(self.doc_path, "r") as f:
            while True:
                lines = ""
                for i in range(batch_size):
                    lines += f.readline()

                if gensim_custom_tokenizer is not None:
                    tokens = preprocess_string(lines, gensim_custom_tokenizer)
                    for token in tokens:
                        self.token_count[token] += 1
                    yield tokens
                else:
                    yield lines

                if lines == "":
                    break

    def get_n_top_tokens(self, n):
        return [
            item[0]
            for item in sorted(
                self.token_count.items(), key=lambda item: item[1], reverse=True
            )[:n]
        ]

    def get_n_bottom_tokens(self, n):
        return [
            item[0]
            for item in sorted(
                self.token_count.items(), key=lambda item: item[1], reverse=False
            )[:n]
        ]

    def get_frequent_tokens(self, min_freq):
        return [k for k, v in self.token_count.items() if v >= min_freq]

    def get_rare_tokens(self, max_freq):
        return [k for k, v in self.token_count.items() if v <= max_freq]

    def get_token_frequency(self, token):
        return self.token_count[token]
