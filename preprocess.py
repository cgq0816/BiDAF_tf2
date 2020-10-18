import re
import os
import nltk
import numpy as np
import data_io as pio
from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer

GLove_path = '{}\glove.6B\glove.6B.50d.txt'.format(os.path.dirname(os.path.abspath(__file__)))

class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        self.build_charset()
        self.word_list = []
        self.max_char_len = 0
        self.embeddings_index = {}
        self.embedding_matrix = []
        self.load_glove(GLove_path)
        self.build_words()

    # 对字符创建字典
    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']

        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)
        # return self.ch2id, self.id2ch, self.charset, np.array(charEmbedding)

    # 对获得的单词构建字典
    def build_words(self):
        idx = list(range(len(self.word_list)))
        self.w2id = dict(zip(self.word_list, idx))
        self.id2w = dict(zip(idx, self.word_list))

    # 加载glove
    def load_glove(self, GLove_path):
        with open(GLove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # 获取单个单词
                word = values[0]
                if len(word) > self.max_char_len:
                    self.max_char_len = len(word)
                coefs = np.asarray(values[1:], dtype='float32')
                # 将单词与数组合成字典
                self.embeddings_index[word] = coefs
                # 创建纯单词列表，创建纯coefs array列表
                self.embedding_matrix.append(coefs)
                self.word_list.append(word)

    # 将iter_cqa中读的字符串放入都charset
    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    # 将iter_cqa中读的context, question, answer字符串使用nltk进行分词
    def dataword_info(self, inn):
        contextlist = []
        questionlist = []
        answerlist = []
        wordslist = []
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            # 通过正则去除标点，除'外
            self.func_re(context)
            self.func_re(question)
            self.func_re(answer)
            # 使用nltk进行分词
            contextlist.append(WordPunctTokenizer().tokenize(context))
            questionlist.extend(WordPunctTokenizer().tokenize(question))
            answerlist.extend(WordPunctTokenizer().tokenize(answer))

            # 汇总到wordslist中
            wordslist.extend(contextlist)
            wordslist.extend(questionlist)
            wordslist.extend(answerlist)

        return contextlist, questionlist, answerlist, wordslist

    # 得到json数据集中对应key的value值，主要是context, question, text
    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    # 先将字母变成小写再分词
    def seg_text(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text)]
        return words

    def char_encode(self, context, question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)

        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def word_encode(self, context, question):
        q_seg_list = self.seg_text(question)
        c_seg_list = self.seg_text(context)

        question_encode = self.convert2id_word(sent=q_seg_list, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id_word(sent=c_seg_list, maxlen=left_length, end=True,)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id_char(self, word_list = [], max_char_len=None, maxlen=None, begin=False, end=False):
        char_list = []
        char_list = [[self.get_id_char('[CLS]')] + [self.get_id_char('[PAD]')] * (max_char_len - 1)]\
                    * begin + char_list
        # 将单词列表中逐个遍历出，并切成char
        for word in word_list:
            ch = [ch for ch in word]
            if max_char_len is not None:
                ch = ch[:max_char_len]
        # ch = ['[CLS]'] * begin + ch

        ids = list(map(self.get_id_char, ch))
        while len(ids) < max_char_len:
            ids.append(self.get_id_char('[PAD]'))
        char_list.append(np.array(ids))
        if maxlen is not None:
            char_list = char_list[:maxlen - 1 * end]
            # char_list += ['[SEP]'] * end
            # char_list += ['[PAD]'] * (maxlen - len(char_list))
            char_list += [[self.get_id_char('[PAD]')] * max_char_len] * (maxlen - len(char_list))
        else:
            char_list += ['[SEP]'] * end

        return char_list

    def convert2id_word(self, word_list = [], maxlen=None, begin=False, end=False):
        word = [word for word in word_list]
        word = ['[cls]'] * begin + word

        if maxlen is not None:
            word = word[:maxlen - 1 * end]
            word += ['[sep]'] * end
            word += ['[pad]'] * (maxlen - len(word))
        else:
            word += ['[sep]'] * end

        ids = list(map(self.get_id_word, word))

        return ids

    def get_id_char(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_id_word(self, ch):
        return self.w2id.get(ch, self.w2id['[unk]'])

    def get_dataset(self, ds_fp):
        cs, qs, be = [], [], []
        for _, c, q, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            be.append((b, e))
        return map(np.array, (cs, qs, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            c_seg_list = self.seg_text(context)
            q_seg_list = self.seg_text(question)
            c_char_ids = self.get_sent_ids_char(maxlen=self.max_clen, word_list=c_seg_list)
            q_char_ids = self.get_sent_ids_char(maxlen=self.max_qlen, begin=True, word_list=q_seg_list)
            c_word_ids = self.get_sent_ids_word(maxlen=self.max_clen, word_list=c_seg_list)
            q_word_ids = self.get_sent_ids_word(maxlen=self.max_qlen, begin=True, word_list=q_seg_list)
            # cids = self.get_sent_ids(context, self.max_clen)
            # qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            nb = -1
            ne = -1
            len_all_char = 0
            for i, w in enumerate(c_seg_list):
                if i == 0:
                    continue
                if b > len_all_char -1 and b <= len_all_char+len(w) -1:
                    b = i + 1
                if e > len_all_char -1 and e <= len_all_char+len(w) -1:
                    e = i + 1
                len_all_char += len(w)
            if ne == -1:
                b = e = 0
            # if e >= len(cids):
            #     b = e = 0
            yield qid, c_char_ids, q_char_ids, c_word_ids, q_word_ids, b, e

    def get_sent_ids_char(self, maxlen=0, begin=False, end=True, word_list=[]):
        return self.convert2id_char(max_char_len=self.max_char_len, maxlen=maxlen, word_list=self.word_list)

    def get_sent_ids_word(self, maxlen=0, begin=False, end=True, word_list=[]):
        return self.convert2id_word(maxlen=maxlen, word_list=self.word_list)

    # 通过正则去除标点，除'外
    def func_re(self, value):
        pat_letter = re.compile(r'[^a-zA-Z \']+')
        # 对一些缩略词进行转换处理
        # to find the 's following the pronouns. re.I is refers to ignore case
        pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
        # to find the 's following the letters
        pat_s = re.compile("(?<=[a-zA-Z])\'s")
        # to find the ' following the words ending by s
        pat_s2 = re.compile("(?<=s)\'s?")
        # to find the abbreviation of not
        pat_not = re.compile("(?<=[a-zA-Z])n\'t")
        # to find the abbreviation of would
        pat_would = re.compile("(?<=[a-zA-Z])\'d")
        # to find the abbreviation of will
        pat_will = re.compile("(?<=[a-zA-Z])\'ll")
        # to find the abbreviation of am
        pat_am = re.compile("(?<=[I|i])\'m")
        # to find the abbreviation of are
        pat_are = re.compile("(?<=[a-zA-Z])\'re")
        # to find the abbreviation of have
        pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

        value = pat_letter.sub(' ', value).strip().lower()
        value = pat_is.sub(r"\1 is", value)
        value = pat_s.sub("", value)
        value = pat_s2.sub("", value)
        value = pat_not.sub(" not", value)
        value = pat_would.sub(" would", value)
        value = pat_will.sub(" will", value)
        value = pat_am.sub(" am", value)
        value = pat_are.sub(" are", value)
        value = pat_ve.sub(" have", value)
        value = value.replace('\'', ' ')

if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    print(p.char_encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    print(p.word_encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
