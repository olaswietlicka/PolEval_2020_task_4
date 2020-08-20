from transformers import TFRobertaForTokenClassification, TFRobertaForSequenceClassification, TFBertForSequenceClassification, TFXLNetForSequenceClassification, TFBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM, concatenate, \
    MaxPooling2D, Lambda, ZeroPadding2D, GlobalMaxPool2D, GlobalAvgPool2D, Multiply, Dropout, Subtract, Add, \
    Bidirectional, TimeDistributed, Conv1D, Conv3D, MaxPooling1D, GRU
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

sgd = optimizers.SGD(lr=0.001)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
rmsprop = optimizers.RMSprop(lr = 0.001)

def NER_PolEval(seq_len, tag_index, embedding_dim, word_index, embedding_matrix):
    input = Input(shape=(seq_len,), dtype='int32')
    middle_layer = Embedding(len(word_index)+1, embedding_dim,
                             embeddings_initializer=initializers.Constant(embedding_matrix),
                             trainable=False)(input)
    middle_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(middle_layer)
    middle_layer = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(middle_layer)
    middle_layer = Dense(32, activation='relu')(middle_layer)
    output = Dense(len(tag_index) + 1, activation='softmax')(middle_layer)
    NER_PolEval_model = Model(input, output)
    NER_PolEval_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                              optimizer=rmsprop,
                              metrics=['sparse_categorical_accuracy'])
    NER_PolEval_model.summary()
    # zapisanie struktury sieci do pliku:
    # graph_file = 'NER_PolEval_seq2seq.png'
    # plot_model(NER_PolEval_model, show_shapes=True, show_layer_names=True, to_file=graph_file)
    return NER_PolEval_model

def NER_PolEval_middle(seq_len_middle, tag_index, embedding_dim, word_index, embedding_matrix):
    input = Input(shape=(seq_len_middle,), dtype='int32')
    middle_layer = Embedding(len(word_index) + 1, embedding_dim, input_length=seq_len_middle,
                             embeddings_initializer=initializers.Constant(embedding_matrix),
                             trainable=True)(input)
    middle_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(middle_layer)
    middle_layer = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(middle_layer)
    # middle_layer = Conv1D(128, 1, activation='sigmoid')(middle_layer)
    # middle_layer = MaxPooling1D()(middle_layer)
    middle_layer = Dense(64, activation='sigmoid')(middle_layer)
    middle_layer = Flatten()(middle_layer)
    # middle_layer = Dense(32, activation='relu')(middle_layer)
    output = Dense(len(tag_index)+1, activation='softmax')(middle_layer)
    NER_PolEval_model = Model(input, output)
    # sgd = optimizers.SGD(lr=0.1)
    NER_PolEval_model.compile(loss='mse',
                              optimizer=rmsprop,
                              metrics=['accuracy'])
    NER_PolEval_model.summary()
    # zapisanie struktury sieci do pliku:
    # graph_file = 'NER_PolEval_middle.png'
    # plot_model(NER_PolEval_model, show_shapes=False, show_layer_names=False, to_file=graph_file)
    return NER_PolEval_model

def NER_PolEval_middle_GRU(seq_len_middle, tag_index, embedding_dim, word_index, embedding_matrix):
    input = Input(shape=(seq_len_middle,), dtype='int32')
    middle_layer = Embedding(len(word_index) + 1, embedding_dim, input_length=seq_len_middle,
                             embeddings_initializer=initializers.Constant(embedding_matrix),
                             trainable=True)(input)
    middle_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(middle_layer)
    middle_layer = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(middle_layer)
    # middle_layer = Conv1D(128, 1, activation='sigmoid')(middle_layer)
    # middle_layer = MaxPooling1D()(middle_layer)
    # middle_layer = Dense(64, activation='sigmoid')(middle_layer)
    middle_layer = GRU(128, dropout=0.1, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid')(middle_layer)
    middle_layer = Flatten()(middle_layer)
    # middle_layer = Dense(32, activation='relu')(middle_layer)
    output = Dense(len(tag_index)+1, activation='softmax')(middle_layer)
    NER_PolEval_model = Model(input, output)
    # sgd = optimizers.SGD(lr=0.1)
    NER_PolEval_model.compile(loss='mse',
                              optimizer=rmsprop,
                              metrics=['accuracy'])
    NER_PolEval_model.summary()
    # zapisanie struktury sieci do pliku:
    # graph_file = 'NER_PolEval_middle.png'
    # plot_model(NER_PolEval_model, show_shapes=False, show_layer_names=False, to_file=graph_file)
    return NER_PolEval_model

def NER_PolEval_Bert(seq_len_middle, tag_index):
    input = Input(shape=(seq_len_middle,), dtype='int32')
    middle_layer = TFBertModel.from_pretrained('bert-base-multilingual-uncased', trainable=False, num_labels=2*(len(tag_index)+1))(input)  # TFRobertaForTokenClassification.from_pretrained('roberta-base')
    middle_layer = middle_layer[0]
    middle_layer = Flatten()(middle_layer)
    output = Dense(len(tag_index)+1, activation='sigmoid')(middle_layer)
    model = Model(input, output)
    # adam = optimizers.Adam(lr=0.00001)  # , beta_1=0.9, beta_2=0.999)
    # rmsprop = optimizers.RMSprop(lr=0.01)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    model.summary()
    # graph_file = 'NER_PolEval_Bert.png'
    # plot_model(model, show_shapes=True, show_layer_names=True, to_file=graph_file)
    return model

def NER_PolEval_RoBertA(seq_len_middle, tag_index):
    input = Input(shape=(seq_len_middle,), dtype='int32')
    middle_layer = TFRobertaForSequenceClassification.from_pretrained('roberta-base')(input)  # TFRobertaForTokenClassification.from_pretrained('roberta-base')
    middle_layer = middle_layer[0]
    middle_layer = Flatten()(middle_layer)
    output = Dense(len(tag_index)+1, activation='sigmoid')(middle_layer)
    model = Model(input, output)
    adam = optimizers.Adam(lr=0.00001)  # , beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    # graph_file = 'NER_PolEval_RoBertA.png'
    # plot_model(model, show_shapes=True, show_layer_names=True, to_file=graph_file)
    return model

def NER_PolEval_XLNet(seq_len_middle, tag_index):
    input = Input(shape=(seq_len_middle,), dtype='int32')
    middle_layer = TFXLNetForSequenceClassification.from_pretrained('xlnet-large-cased')(input)  # TFRobertaForTokenClassification.from_pretrained('roberta-base')
    middle_layer = middle_layer[0]
    middle_layer = Flatten()(middle_layer)
    output = Dense(len(tag_index)+1, activation='sigmoid')(middle_layer)
    model = Model(input, output)
    adam = optimizers.Adam(lr=0.00001)  # , beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    # graph_file = 'NER_PolEval_XLNet.png'
    # plot_model(model, show_shapes=True, show_layer_names=True, to_file=graph_file)
    return model

def model_akty(seq_len_middle, tag_index, embedding_dim, word_index, word_index_morf, embedding_matrix_1, embedding_matrix_2):
    ## main model
    input = Input (shape=(seq_len_middle,))
    model = Embedding (len(word_index) + 1, embedding_dim, embeddings_initializer=initializers.Constant(embedding_matrix_1), input_length=seq_len_middle, trainable=False) (input)
    # model = Bidirectional (LSTM (units=85, return_sequences=True, dropout=drop), merge_mode="mul") (model)
    # model = TimeDistributed (Dense (100, activation="relu")) (model)

    model = Bidirectional (LSTM (128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)) (model)
    model = LSTM (128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2) (model)
    model = GRU(64, dropout=0.1, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid')(model)

    model = Flatten () (model)

    input3 = Input (shape=(seq_len_middle,))
    model3 = Embedding (len(word_index_morf) + 1, embedding_dim, embeddings_initializer=initializers.Constant(embedding_matrix_2), input_length=seq_len_middle, trainable=True) (input3)
    model3 = Bidirectional (LSTM (128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)) (model3)
    model3 = LSTM (128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2) (model3)
    model3 = GRU(64, dropout=0.1, recurrent_dropout=0.5, activation='tanh', recurrent_activation='sigmoid')(model3)

    model3 = Flatten () (model3)

    merged = tf.keras.layers.concatenate([model, model3])
    model = Dense (64, activation='relu') (merged)
    # model = Dropout(0.20) (model)
    # model = Dense (10, activation='relu') (model)
    # out = (Dense (2, activation='softmax')) (model)
    out = (Dense (len(tag_index)+1, activation='softmax')) (model)

    # model = Model (input, out)
    model = Model(inputs=[input, input3], outputs=out)
    # model.compile (loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile (loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    model.compile (loss='mse', optimizer=rmsprop, metrics=['accuracy'])
    model.summary ()
    return model