import os

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_short,
    strip_numeric,
)


class TextDataSets:
    def __init__(self, ds_path, description, custom_preprocessing=None):
        self.ds_path = ds_path
        self.description = description
        self.custom_preprocessing = custom_preprocessing

    def read_data(self):
        pass


class SingleClassDS(TextDataSets):
    def __init__(
        self,
        ds_path,
        description,
        class_label,
        custom_preprocessing=[
            lambda x: x.lower(),
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            remove_stopwords,
            lambda x: strip_short(x, minsize=2),
            strip_numeric,
        ],
    ):
        super(SingleClassDS, self).__init__(ds_path, description, custom_preprocessing)
        self.class_label = class_label

    def read_txt_data(self, read_n_lines=None):
        features = []
        targets = []
        n_lines_count = 0

        if os.path.isfile(self.ds_path):
            with open(self.ds_path, "r") as f:
                for line in f.readlines():
                    if self.custom_preprocessing is not None:
                        features.append(
                            preprocess_string(line, self.custom_preprocessing)
                        )
                    else:
                        features.append(line)
                    targets.append(self.class_label)
                    if read_n_lines is not None:
                        n_lines_count += 1
                        if n_lines_count > read_n_lines:
                            break
        elif os.path.isdir(self.ds_path):
            for filename in os.listdir(self.ds_path):
                file_path = os.path.join(self.ds_path, filename)
                if os.path.isfile(file_path) and file_path.endswith(".txt"):
                    with open(file_path) as f:
                        for line in f.readlines():
                            if self.custom_preprocessing is not None:
                                features.append(
                                    preprocess_string(line, self.custom_preprocessing)
                                )
                            else:
                                features.append(line)
                            targets.append(self.class_label)
                            if read_n_lines is not None:
                                n_lines_count += 1
                                if n_lines_count > read_n_lines:
                                    break

        return features, targets

    def read_txt_gen(self, batch_size):
        if os.path.isfile(self.ds_path):
            data_file = open(self.ds_path, "r")

            while True:
                features = []
                targets = []

                for i in range(batch_size):
                    line = data_file.readline()
                    if line == "":
                        break

                    if self.custom_preprocessing is not None:
                        features.append(
                            preprocess_string(line, self.custom_preprocessing)
                        )
                    else:
                        features.append(line)
                    targets.append(self.class_label)

                yield features, targets

                if line == "":
                    break

        elif os.path.isdir(self.ds_path):
            file_path_list = (
                os.path.join(self.ds_path, filename)
                for filename in os.listdir(self.ds_path)
                if filename.endswith(".txt")
            )

            while True:
                features = []
                targets = []

                for i in range(batch_size):
                    try:
                        file_path = next(file_path_list)
                    except:
                        file_path = ""
                        break
                    with open(file_path) as f:
                        for line in f.readlines():
                            if self.custom_preprocessing is not None:
                                features.append(
                                    preprocess_string(line, self.custom_preprocessing)
                                )
                            else:
                                features.append(line)
                            targets.append(self.class_label)

                if file_path == "":
                    break

                yield features, targets


if __name__ == "__main__":
    ds = SingleClassDS(
        ds_path="/Users/lukaszmaczewski/Documents/Learning/DeepNeuralNetworks/keras/data/aclImdb/test/neg/0_2.txt",
        description="test_negative",
        class_label=0,
    )

    print(ds.read_txt_data())

    ds = SingleClassDS(
        ds_path="/Users/lukaszmaczewski/Documents/Learning/DeepNeuralNetworks/keras/data/aclImdb/test/neg",
        description="test_negative",
        class_label=0,
    )

    for i in ds.read_txt_gen(batch_size=2):
        print(i)
