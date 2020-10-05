import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding

GLove_path = '{}\glove.6B\glove.6B.50d.txt'.format(os.path.dirname(os.path.abspath(__file__)))


print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class BiDAF:

    def __init__(
            self, clen, qlen, emb_size,
            max_features=5000,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
    ):
        """
        双向注意流模型
        :param clen:context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
        """
        self.clen = clen
        self.qlen = qlen
        self.max_features = max_features
        self.emb_size = emb_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    def build_model(self, word_index, charEmbedding):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding 
        cinn = tf.keras.layers.Input(shape=(self.clen,), name='context_input')
        qinn = tf.keras.layers.Input(shape=(self.qlen,), name='question_input')

        word_num = 10000
        # GloVe的向量维度
        embedding_dim = 100

        # 调用glove
        embeddings_index = {}
        # embedding_weight = np.zeros([word_num, embedding_dim])
        embedding_weight = np.random.uniform(-0.05, 0.05, size=[word_num, embedding_dim])
        with open(GLove_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        # 匹配GloVe向量
        embedding_matrix = np.zeros((word_num, embedding_dim))
        for word, i in word_index.items():
            if i < word_num:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

        model = tf.keras.Sequential()
        model.add(Embedding(word_num, embedding_dim, weights=[embedding_weight]))

        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(128, activation=tf.nn.relu))
        # model.add(Dense(2, activation='softmax'))
        # model.summary()

        with tf.Graph().as_default():

            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
            config = session_conf
            sess = tf.Session(config=session_conf)

        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.isTraining = tf.placeholder(tf.bool, name="isTraining")

        # 字符嵌入
        with tf.name_scope("embedding"):

            # 利用one-hot的字符向量作为初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(charEmbedding, dtype=tf.float32, name="charEmbedding"), name="W")
            # 获得字符嵌入
            self.embededChars = tf.nn.embedding_lookup(self.W, self.inputX)
            # 添加一个通道维度
            self.embededCharsExpand = tf.expand_dims(self.embededChars, -1)


        embedding_layer = tf.keras.layers.Embedding(self.max_features,
                                                    self.emb_size,
                                                    embeddings_initializer='uniform',
                                                    )
        cemb = embedding_layer(cinn)
        qemb = embedding_layer(qinn)

        for i in range(self.num_highway_layers):
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question

        # 3.注意流层
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [cinn, qinn]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)


if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])

    # 获得glove需要的word字符集
    word_index = ds.build_words()
    # 获得char cnn需要的char字符集
    ch2id, id2ch, charset, charEmbedding = ds.build_charset()

    train_c, train_q, train_y = ds.get_dataset('./data/squad/train-v1.1.json')
    test_c, test_q, test_y = ds.get_dataset('./data/squad/dev-v1.1.json')

    print(train_c.shape, train_q.shape, train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=50,
        max_features=len(ds.charset)
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_c, train_q], train_y,
        batch_size=64,
        epochs=10,
        validation_data=([test_c, test_q], test_y)
    )
