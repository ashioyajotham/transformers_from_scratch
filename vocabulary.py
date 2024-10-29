import re
import unittest

<<<<<<< HEAD
from typing import List, Optional

=======
from typing import List, Dict, Tuple, Optional

# We are creating a Vocabulary class that will be used to store the mapping between tokens and their corresponding indices.
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da

class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]):
<<<<<<< HEAD
        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
=======
        self.token_to_index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index_to_token = {v : k for k, v in self.token_to_index.items()}

>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
        if not list_of_sentences:
            return
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

<<<<<<< HEAD
    def add_tokens(self, tokens: List[str]) -> None:
        """
        Adds tokens the vocabulary

        :param tokens:
        :return: None
        """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Split on all tokens and punctuation. Optionally adds BOS and EOS tokens.

        :param sentence:
        :param add_special_tokens:
        :return: List of string tokens
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts a string to a list of token indices given the vocabulary

        :param sentence: a string representing a sentence
        :param add_special_tokens: whether or not to add a bos and eos token
        :return: list of token indices
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token2index[token] for token in tokens]

    def batch_encode(
        self, sentences: List[str], padding=True, add_special_tokens: bool = False
    ) -> List[List[int]]:
        """
        Convert a list of string sentences to nested list of token indices. Optionally adds padding & bos+eos tokens

        :param sentences: A list of sentences to be encoded into a batch
        :param padding: Boolean that allows for padding up to the longest sequence in the batch
        :param add_special_tokens: Boolean that allows for adding a BOS and EOS token to each sentence in the batch
        :return: nested list of tokenized sequences
        """
        tokenized_sentences = [
            self.encode(sentence, add_special_tokens) for sentence in sentences
        ]
        if padding:
            max_length = max([len(tokens) for tokens in tokenized_sentences])
            tokenized_sentences = [
                s + ((max_length - len(s)) * [self.token2index[self.PAD]])
                for s in tokenized_sentences
            ]
        return tokenized_sentences


=======

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add tokens to the vocabulary
        :param tokens: list of tokens
        :return: None
        """
        for token in tokens:
            if token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token

    def tokenize(self, sentence: str, add_special_tokens:bool = True) -> List[int]:
        """
        Tokenize a sentence
        :param sentence: input sentence
        :param add_special_tokens: whether to add special tokens or not
        :return: list of tokens
        """
        return re.findall(r'\w+|[^\s\w]+', sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens
    
    def encode(self, sentence: str, add_special_tokens:bool = True) -> List[int]:
        """
        Encode a sentence ie convert a sentence to a list of indices
        :param sentence: input sentence
        :param add_special_tokens: whether to add special tokens or not
        :return: list of indices
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token_to_index[token] for token in tokens]
    
    def batch_encode(self, sentences: List[str], padding = True, add_special_tokens:bool = False) -> List[List[int]]:
        """
        Convert a list of sentences to a list of list (nested) of token indices. Optionally pad the sequences and add special tokens.
        :param sentences: list of input sentences
        :param add_special_tokens: whether to add special tokens or not
        :return: list of list of indices
        """
        tokenized_sentences = [self.encode(sentence, add_special_tokens) for sentence in sentences]
        if padding:
            max_length = max(len(sentence) for sentence in tokenized_sentences)
            tokenized_sentences = [sentence + ((max_length - len(sentence)) * [self.token_to_index[self.PAD]]) for sentence in tokenized_sentences]
        return tokenized_sentences
    
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
class TestVocabulary(unittest.TestCase):
    maxDiff = None

    def test_tokenize(self):
<<<<<<< HEAD
        input_sequence = "Hello my name is Joris and I was born with the name Joris."
        output = Vocabulary([]).tokenize(input_sequence)
        self.assertEqual(
            [
                "BOS",
                "Hello",
                "my",
                "name",
                "is",
                "Joris",
                "and",
                "I",
                "was",
                "born",
                "with",
                "the",
                "name",
                "Joris",
                ".",
                "EOS",
            ],
            output,
        )

    def test_init_vocab(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
=======
        vocab = Vocabulary(None)
        input_sentence = "Hello, my name is Jotham and I am a machine learning researcher."
        output = Vocabulary([]).tokenize(input_sentence)
        self.assertEqual(
            [
                "BOS", "Hello", ",", "my", "name", "is", "Jotham", "and", "I", "am", "a", "machine", "learning", "researcher", ".", "EOS"
            ]
        )

    def test_init_vocab(self):
        input_sentences = ["Hello, my name is Jotham and I am a machine learning researcher."]
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
        vocab = Vocabulary(input_sentences)
        expected = {
            "BOS": 0,
            "EOS": 1,
            "PAD": 2,
            "Hello": 3,
            "my": 4,
            "name": 5,
            "is": 6,
<<<<<<< HEAD
            "Joris": 7,
            "and": 8,
            "I": 9,
            "was": 10,
            "born": 11,
            "with": 12,
            "the": 13,
            ".": 14,
        }
        self.assertEqual(vocab.token2index, expected)

    def test_encode(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0])
        self.assertEqual(output, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14, 1])

    def test_encode_no_special_tokens(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0], add_special_tokens=False)
        self.assertEqual(output, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14])
=======
            "Jotham": 7,
            "and": 8,
            "I": 9,
            "am": 10,
            "a": 11,
            "machine": 12,
            "learning": 13,
            "researcher": 14
        }

        self.assertEqual(vocab.token_to_index, expected)

    def test_encode(self):
        input_sentences = ["Hello, my name is Jotham and I am a machine learning researcher."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0])
        self.assertEqual(output, [0,3,4,5,6,7,8,9,10,11,12,13,5,7,14,1])

    def test_encode_no_special_tokens(self):
        input_sentences = ["Hello, my name is Jotham and I am a machine learning researcher."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0], add_special_tokens=False)
        self.assertEqual(output, [3,4,5,6,7,8,9,10,11,12,13,5,7,14])
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da

    def test_batch_encode(self):
        input_sentences = [
            "This is one sentence",
            "This is another, much longer sentence",
            "Short sentence",
        ]
        vocab = Vocabulary(input_sentences)
        output = vocab.batch_encode(input_sentences, add_special_tokens=False)
        self.assertEqual(
            output,
<<<<<<< HEAD
            [[3, 4, 5, 6, 2, 2, 2], [3, 4, 7, 8, 9, 10, 6], [11, 6, 2, 2, 2, 2, 2]],
        )


if __name__ == "__main__":
=======
            [[3,4,5,6,2,2,2], [3,4,7,8,9,10,6], [11,6,2,2,2,2,2]],
        )
        
if __name__ == '__main__':
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
    unittest.main()