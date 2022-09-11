import re
import numpy as np
import pickle
import argparse


class NGramModel(object):

    def __init__(self, n: int = 2):
        self.n = n
        self.model = dict()

    def __clear_sample(self, sample: str) -> list:
        sample = sample.lower()
        sample = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', sample)
        return sample.split('\n')

    def __create_vocab(self, ngram: list = None) -> dict:
        vocab = dict()
        for object in ngram:
            if object not in vocab:
                rep = 0
                for i in np.arange(len(ngram)):
                    if object == ngram[i]:
                        rep += 1
            vocab[object] = rep
        return vocab

    def __create_ngrams(self, tokens: list, n: int) -> list:
        ngrams = zip(*[tokens[i:] for i in np.arange(n)])
        return list(ngrams)

    def __load_train_data(self, input_dir: str = None) -> list:
        sample = ''
        if input_dir is not None:
            with open(input_dir, 'r', encoding='utf-8') as train_file:
                sample = train_file.read()
        else:
            sample = str(input())

        raw_tokenized_sample = self.__clear_sample(sample)
        tokens = list()
        for elem in raw_tokenized_sample:
            tokens += elem.split(' ')
        return list(filter(lambda x: x != '', tokens))

    def __smooth(self) -> dict:
        vocab_size = len(self.vocab)

        n_grams = self.__create_ngrams(self.tokens, self.n)
        n_vocab = self.__create_vocab(n_grams)

        m_grams = self.__create_ngrams(self.tokens, self.n - 1)
        m_vocab = self.__create_vocab(m_grams)

        def sm_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return n_count / (m_count + 0.001 * vocab_size)

        return {n_gram: sm_count(n_gram, count) for n_gram, count in n_vocab.items()}

    def __best_choice(self, prev, i):

        candidates = ((ngram[-1], prob) for ngram, prob in self.model.items() if ngram[:-1] == prev)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return '', 1
        else:
            return candidates[0 if prev != () else i]

    def fit(self, input_dir: str = None):
        self.tokens = self.__load_train_data(input_dir)
        self.vocab = self.__create_vocab(self.tokens)
        if self.n == 1:
            self.model = {(unigram,): count / len(self.tokens) for unigram, count in self.vocab.items()}
        else:
            self.model = self.__smooth()

    def generate(self, length: int, prefix: str=None) -> str:
        if prefix is None:
            prefix = self.tokens[np.random.randint(0, high=len(self.tokens))]

        result = prefix.split(' ')
        for i in np.arange(length):
            prev = () if self.n == 1 else tuple(result[-(self.n - 1):])
            next_token, _ = self.__best_choice(prev, i)
            result.append(next_token)
        return ' '.join(result)

    def save_model(self, path: str = './model.pkl'):
        with open(path, 'wb') as model:
            pickle.dump((self.model, self.tokens), model)

    def load_model(self, path: str = './model.pkl'):
        with open(path, 'rb') as model:
            self.model, self.tokens = pickle.load(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help='путь к директории, в которой лежит коллекция документов. '
                                                      'Если данный аргумент не задан, считать, что тексты вводятся из '
                                                      'stdin.')
    parser.add_argument('--model', type=str, help='путь к файлу, в который сохраняется модель.')

    args = parser.parse_args()

    model = NGramModel()

    if args.input_dir is not None:
        model.fit(args.input_dir)
    else:
        model.fit()

    model.save_model(args.model)
