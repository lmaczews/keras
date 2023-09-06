from kst.data_sets.TextSequences import TextSequences


class TextSequencesCorpus:
    def __init__(self, corpus_description = None, gensim_custom_tokenizer=None):
        self.collection_of_sequences = []
        self.corpus_description = corpus_description
        self.gensim_custom_tokenizer = gensim_custom_tokenizer
        self.vocabulary = set([])
        self._token2index = {}
        self._index2token = {}

    def add_sequences(self, sequences: TextSequences):
        self.collection_of_sequences += [sequences]
        self.vocabulary.update(sequences.vocabulary)

    def prepare_token_to_index_mappings(self):
        for index, token in enumerate(self.vocabulary):
            self._token2index[token] = index + 1
            self._index2token[index + 1] = token

    def get_index(self, token):
        return self._token2index.get(token)

    def get_token(self, index):
        return self._index2token.get(index)