import spacy

spacy_eng = spacy.load("en_core_web_trf")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.index_to_string = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.string_to_index = {v: k for k, v in self.index_to_string.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.index_to_string)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        index = self.__len__() + 1

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        for word, frequency in frequencies.items():
            if frequency >= self.freq_threshold:
                self.index_to_string[index] = word
                self.string_to_index[word] = index
                index += 1

    def text_to_numbers(self, text):
        return [
            self.string_to_index[token]
            if token in self.string_to_index
            else self.string_to_index["<UNK>"]
            for token in self.tokenizer_eng(text)
        ]

