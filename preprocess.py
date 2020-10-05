import re
import numpy as np
import data_io as pio
from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer


class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        self.build_charset()
        self.build_words()

    # 对字符获得排序完的字典
    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        charEmbedding = []
        charEmbedding.append(np.zeros(len(self.charset), dtype="float32"))
        for i, alpha in enumerate(self.charset):
            onehot = np.zeros(len(self.charset), dtype="float32")

            # 生成每个字符对应的向量
            onehot[i] = 1

            # 生成字符嵌入的向量矩阵
            charEmbedding.append(onehot)
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)
        return self.ch2id, self.id2ch, self.charset, np.array(charEmbedding)

    # 对获得的词排序
    def build_words(self):
        for fp in self.datasets_fp:
             self.words = self.dataword_info(fp)
        count_dict = defaultdict(lambda: 0)
        for item in self.words:
            count_dict[item] += 1
        return sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

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
        print(contextlist)

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

    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode)
        context_encode = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids

    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

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
            cids = self.get_sent_ids(context, self.max_clen)
            qids = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids):
                b = e = 0
            yield qid, cids, qids, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)

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
    print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
