from collections import defaultdict

from gensim.parsing.preprocessing import (
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
)

from kst.data_sets.TextDocument import TextDocument


class TextCorpus:
    def __init__(
        self, corpus_description=None, gensim_custom_tokenizer=None, stream=False
    ):
        self.text_documents_list = []
        self.corpus_description = corpus_description
        self.gensim_custom_tokenizer = gensim_custom_tokenizer
        self.token_absolute_count = defaultdict(int)
        self.token_doc_count = defaultdict(int)
        self.corpus = []
        self.labels = []

    @property
    def gensim_custom_tokenizer(self):
        return self._gensim_custom_tokenizer

    @gensim_custom_tokenizer.setter
    def gensim_custom_tokenizer(self, tokenizer):
        if tokenizer is None:
            self._gensim_custom_tokenizer = [
                lambda x: x.lower(),
                strip_tags,
                strip_punctuation,
                strip_multiple_whitespaces,
            ]
        else:
            self._gensim_custom_tokenizer = tokenizer

    @property
    def vocabulary(self):
        return self.token_absolute_count.keys()

    @property
    def docs_count(self):
        return len(self.text_documents_list)

    def reset(self):
        self.text_documents_list = []
        self.token_absolute_count = defaultdict(int)
        self.token_doc_count = defaultdict(int)
        self.corpus = []
        self.labels = []

    def get_frequent_tokens(self, min_absolute_count, min_doc_count):
        return [
            k
            for k in self.vocabulary
            if (self.token_absolute_count[k] >= min_absolute_count)
            and (self.token_doc_count[k] >= min_doc_count)
        ]

    def get_rare_tokens(self, max_absolute_count, max_doc_count):
        return [
            k
            for k in self.vocabulary
            if (self.token_absolute_count[k] <= max_absolute_count)
            and (self.token_doc_count[k] <= max_doc_count)
        ]

    def add_document(
        self, text_document: TextDocument, batch_size=None, stream=False
    ):
        self.text_documents_list.append(text_document)

        if batch_size is not None:
            dock = []
            for batch in text_document.read_document_gen(
                    batch_size=batch_size,
                    gensim_custom_tokenizer=self.gensim_custom_tokenizer,
            ):
                dock += batch
        else:
            dock = text_document.read_document(
                gensim_custom_tokenizer=self.gensim_custom_tokenizer
            )

        for k, v in text_document.token_count.items():
            self.token_absolute_count[k] += v
            self.token_doc_count[k] += 1

        if stream:
            return dock, text_document.doc_label

        self.corpus += [dock]
        self.labels += [text_document.doc_label]

    def get_corpus(self, tokens=None):
        if tokens is None:
            return self.corpus
        result = []
        for dock in self.corpus:
            result += [[word for word in dock if word in tokens]]
        return result
