import re
from collections import defaultdict

from gensim.parsing.preprocessing import preprocess_string


class TextSequences():

    def __init__(self, begin_of_sequence='<BOS>', end_of_sequence='<EOS>'):
        self.txt_sequence = []
        self.token_count = defaultdict(int)
        self.begin_of_sequence = begin_of_sequence
        self.end_of_sequence = end_of_sequence
        self.token_count[self.begin_of_sequence] += 1
        self.token_count[self.end_of_sequence] += 1

    @property
    def vocabulary(self):
        return self.token_count.keys()

    @property
    def tokens_count(self):
        return len(self.vocabulary)

    def text_preprocessing(self, text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''

        text = text.lower()

        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

        return text

    def add_sequence(self, text, gensim_custom_tokenizer=None, initial_preprocessing=True):
        if initial_preprocessing:
            text = self.text_preprocessing(text)

        if gensim_custom_tokenizer is not None:
            tokens = preprocess_string(text, gensim_custom_tokenizer)
        else:
            tokens = text.split(' ')

        for token in tokens:
            self.token_count[token] += 1

        self.txt_sequence += [tokens]

    def get_two_consecutive_sequences(self):
        i = 0
        while len(self.txt_sequence) > i + 1:
            yield self.txt_sequence[i], [self.begin_of_sequence] + self.txt_sequence[i + 1] + [self.end_of_sequence]
            i += 1
