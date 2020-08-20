import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import gensim
from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import pickle
from prepare_data import *
from model_to_file import load_model_from_files, save_model_to_file
from modele import *
from tensorflow.keras.utils import plot_model, to_categorical

import tensorflow as tf

# dane do ustawienia na początku
seq_len = 30
seq_len_middle = 15
middle_word = 7
step = 1
create_new_tokenizer = True
create_new_tokenizer_morfeusz = True
load_training_data_from_file = False
load_validation_data_from_file = False
load_nn_from_file = False
embedding_dim = 100

file_with_embedding_matrix = 'nkjp-forms-restricted-100-skipg-ns.txt'
folder_z_raportami_train = r'.\task4-train\reports'
folder_z_raportami_validate = r'.\task4-validate\reports'
folder_z_raportami_test = r'.\task4-test\reports'
plik_z_nazwiskami_kobiet = 'Wykaz_nazwisk_żeńskich_uwzgl_os__zmarłe_2020-01-22.csv'
plik_z_nazwiskami_mezczyzn = 'Wykaz_nazwisk_męskich_uwzgl_os__zmarłe_2020-01-22.csv'
plik_z_imionami_kobiet = 'lista_imion_żeńskich_os_żyjące_2020-01-21.csv'
plik_z_imionami_mezczyzn = 'lista_imion_męskich_os_żyjące_2020-01-21.csv'
ground_truth_train_file = r'.\task4-train\ground_truth-train.csv'
ground_truth_train = pd.read_csv(ground_truth_train_file, sep=';')

# te pliki wczytuję tylko po to, żeby wiedzieć ile wierszy mam do wypełnienia
ground_truth_validate_file = r'.\task4-validate\ground_truth-validate.csv'
ground_truth_validate = pd.read_csv(ground_truth_validate_file, sep=';')

stopwords_file = 'polish_stopwords.txt'

folder_morfeusz_train = r'.\train'
folder_morfeusz_validate = r'.\validate'
folder_morfeusz_test = r'.\test'

stopwords_file = open(stopwords_file, 'r', encoding='utf-8')
stopwords = stopwords_file.readlines()
stopwords_file.close()

if create_new_tokenizer:
    print('Teraz przygotuję słownik (Tokenizer)')
    # najpierw wczytam wszystkie pliki, żeby utworzyć słownik
    full_dictionary = []
    for folder in os.listdir(folder_z_raportami_train):
        folder_z_raportem = os.path.join(folder_z_raportami_train, folder)
        for i in os.listdir(folder_z_raportem):
            if i.endswith('.txt'):
                file = os.path.join(folder_z_raportami_train, folder_z_raportem, i)
                try:
                    file = open(file, 'r', encoding='utf-8')
                    nowy_raport = file.read()
                    # nowy_akt.decode("unicode_escape", "strict")
                    full_dictionary.append(nowy_raport)
                    file.close()
                except:
                    print('Tego pliku nie wczytałem: {}'.format(i))
    # print(full_dictionary)
    for folder in os.listdir(folder_z_raportami_validate):
        folder_z_raportem = os.path.join(folder_z_raportami_validate, folder)
        for i in os.listdir(folder_z_raportem):
            if i.endswith('.txt'):
                file = os.path.join(folder_z_raportami_validate, folder_z_raportem, i)
                try:
                    file = open(file, 'r', encoding='utf-8')
                    nowy_raport = file.read()
                    # nowy_akt.decode("unicode_escape", "strict")
                    full_dictionary.append(nowy_raport)
                    file.close()
                except:
                    print('Tego pliku nie wczytałem: {}'.format(i))

    # dokładam też wszystkie nazwiska kobiet i mężczyzn w formacie: NAZWISKO oraz Nazwisko
    male_surname = pd.read_csv(plik_z_nazwiskami_mezczyzn, sep=';')
    female_surname = pd.read_csv(plik_z_nazwiskami_kobiet, sep=';')
    male_name = pd.read_csv(plik_z_imionami_mezczyzn, sep=',')
    female_name = pd.read_csv(plik_z_imionami_kobiet, sep=',')
    for row in male_surname['Nazwisko aktualne']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem nazwiska mężczyzn')
    for row in female_surname['Nazwisko aktualne']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem nazwiska kobiet')
    for row in male_name['IMIĘ_PIERWSZE']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem imiona mężczyzn')
    for row in female_name['IMIĘ_PIERWSZE']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem imiona kobiet')

    for i in range(len(full_dictionary)):
        full_dictionary[i] = full_dictionary[i].split()
    full_dictionary_tmp = []
    for i in range(len(full_dictionary)):
        if full_dictionary[i] not in stopwords:
            full_dictionary_tmp.append(full_dictionary[i])
    # print(full_dictionary_tmp)
    full_dictionary = full_dictionary_tmp
    # print(full_dictionary)

    # print(full_dictionary)
    # początkowo miałam aby tokenizacja odbywała się na podstawie oryginalnych słów, ale nagle się okazuje, że nie zawsze układ wielkości liter jest identyczny w dokumencie jak w ground_truth_table
    t = Tokenizer(split=' ', lower=True, filters='§-!"”„#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n', oov_token='__UNKNOWN__')
    t.fit_on_texts(full_dictionary)
    word_index = t.word_index
    print('Number of Unique Tokens:', len(word_index))

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('Wczytam zapisany słownik (Tokenizer)')
    with open('tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)
    word_index = t.word_index
    print('Number of Unique Tokens:', len(word_index))

if create_new_tokenizer_morfeusz:
    print('Teraz przygotuję słownik (Tokenizer)')
    # najpierw wczytam wszystkie pliki, żeby utworzyć słownik
    full_dictionary = []
    for i in os.listdir(folder_morfeusz_train):
        if i.endswith('.csv'):
            file = os.path.join(folder_morfeusz_train, i)
            try:
                text = pd.read_csv(file, sep=',')
                # nowy_akt.decode("unicode_escape", "strict")
                for row in text['basic']:
                    full_dictionary.append(str(row))
                for row in text['token']:
                    full_dictionary.append(str(row))
            except:
                print('Tego pliku nie wczytałem: {}'.format(i))
    # print(full_dictionary)
    for i in os.listdir(folder_morfeusz_validate):
        if i.endswith('.csv'):
            file = os.path.join(folder_morfeusz_validate, i)
            try:
                text = pd.read_csv(file, sep=',')
                # nowy_akt.decode("unicode_escape", "strict")
                for row in text['basic']:
                    full_dictionary.append(str(row))
                for row in text['token']:
                    full_dictionary.append(str(row))
            except:
                print('Tego pliku nie wczytałem: {}'.format(i))

    # dokładam też wszystkie nazwiska kobiet i mężczyzn w formacie: NAZWISKO oraz Nazwisko
    male_surname = pd.read_csv(plik_z_nazwiskami_mezczyzn, sep=';')
    female_surname = pd.read_csv(plik_z_nazwiskami_kobiet, sep=';')
    male_name = pd.read_csv(plik_z_imionami_mezczyzn, sep=',')
    female_name = pd.read_csv(plik_z_imionami_kobiet, sep=',')
    for row in male_surname['Nazwisko aktualne']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem nazwiska mężczyzn')
    for row in female_surname['Nazwisko aktualne']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem nazwiska kobiet')
    for row in male_name['IMIĘ_PIERWSZE']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem imiona mężczyzn')
    for row in female_name['IMIĘ_PIERWSZE']:
        full_dictionary.append(str(row))
        full_dictionary.append(str(row).capitalize())
    print('Dołożyłem imiona kobiet')

    # print(full_dictionary)
    for i in range(len(full_dictionary)):
        full_dictionary[i] = full_dictionary[i].split()
    full_dictionary_tmp = []
    for i in range(len(full_dictionary)):
        if full_dictionary[i] not in stopwords:
            full_dictionary_tmp.append(full_dictionary[i])
    # print(full_dictionary_tmp)
    full_dictionary = full_dictionary_tmp

    # początkowo miałam aby tokenizacja odbywała się na podstawie oryginalnych słów, ale nagle się okazuje, że nie zawsze układ wielkości liter jest identyczny w dokumencie jak w ground_truth_table
    t_morf = Tokenizer(split=' ', lower=True, filters='♦•–§-!"”„#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                       oov_token='__UNKNOWN__')
    t_morf.fit_on_texts(full_dictionary)
    word_index_morf = t_morf.word_index
    print('Number of Unique Tokens:', len(word_index))

    with open('tokenizer_morfeusz.pickle', 'wb') as handle:
        pickle.dump(t_morf, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('Wczytam zapisany słownik (Tokenizer)')
    with open('tokenizer_morfeusz.pickle', 'rb') as handle:
        t_morf = pickle.load(handle)
    word_index_morf = t_morf.word_index
    print('Number of Unique Tokens:', len(word_index))

tags = ['street_start', 'street_continue', 'street_no_start', 'street_no_continue', 'company_start', 'company_continue',
        'drawing_date_day', 'drawing_date_month', 'drawing_date_year',
        'period_from_day', 'period_from_month', 'period_from_year', 'period_to_day', 'period_to_month',
        'period_to_year', 'postal_code_pre', 'postal_code_post', 'city_start', 'city_continue', 'city']
for k in range(12):
    employer = 'human_' + str(k) + '_start'
    tags.append(employer)
    employer = 'human_' + str(k) + '_continue'
    tags.append(employer)
    position = 'position_' + str(k) + '_continue'
    tags.append(position)
    position = 'position_' + str(k) + '_start'
    tags.append(position)
tags.append('o')
print(tags)
# print(len(tags))

tags_tokenizer = Tokenizer(filters='', lower=False)
tags_tokenizer.fit_on_texts(tags)
tag_index = tags_tokenizer.word_index
print('Unique tags:', len(tag_index))
# print(tag_index)

tags_less = ['street', 'street_no', 'company', 'drawing_date_day', 'drawing_date_month', 'drawing_date_year',
             'period_from_day', 'period_from_month', 'period_from_year', 'period_to_day',
             'period_to_month', 'period_to_year', 'postal_code_pre', 'postal_code_post', 'city', 'human', 'position',
             'o']
tags_less_tokenizer = Tokenizer(filters='', lower=False)
tags_less_tokenizer.fit_on_texts(tags_less)
tag_less_index = tags_less_tokenizer.word_index
print('Unique tags:', len(tag_less_index))

if load_training_data_from_file:
    X_train_15 = np.load('X_train_15.npy')
    y_train_15 = np.load('y_train_15.npy')
    X_validate_15 = np.load('X_validate_15.npy')
    y_validate_15 = np.load('y_validate_15.npy')

else:
    X_train_15, y_train_15 = prepare_data(ground_truth_train_file, folder_z_raportami_train, t, tags_tokenizer,
                                          seq_len_middle, step, 'train_15')
    X_validate_15, y_validate_15 = prepare_data(ground_truth_validate_file, folder_z_raportami_validate, t,
                                                tags_tokenizer, seq_len_middle, step, 'validate_15')


# dla pojedynczej ścieżki:
X_train_15 = np.vstack((X_train_15, X_validate_15))
y_train_15 = np.vstack((y_train_15, y_validate_15))

print(X_train_15.shape)
print(y_train_15.shape)

# weźmiemy tylko informację o środkowym slowie (ponieważ prepare_data działa dla modelu seq2seq czyli dla zadanej długości sekwencji daje tą samą długość odpowiedzi)
y_train_15 = y_train_15[:, middle_word]

print(set(y_train_15))

#indeksy w których są tylko nieinteresujące słowa:
y_train_15_o_idx = np.where(y_train_15 == len(tags))[0]
y_train_15_interesting = np.delete(y_train_15, y_train_15_o_idx, 0)
X_train_15_interesting = np.delete(X_train_15, y_train_15_o_idx, 0)
X_train_15_o = X_train_15[y_train_15_o_idx, :]
y_train_15_o = y_train_15[y_train_15_o_idx]
#
print(X_train_15_interesting.shape)
print(y_train_15_interesting.shape)
print(X_train_15_o.shape)
print(y_train_15_o.shape)
#
random_o = np.random.randint(low=X_train_15.shape[0], size=10*X_train_15_interesting.shape[0])
print(random_o)
#
X_train_15_o = X_train_15[random_o, :]
y_train_15_o = y_train_15[random_o]
#
X_train_15 = np.vstack((X_train_15_interesting, X_train_15_o))
y_train_15 = np.hstack((y_train_15_interesting, y_train_15_o))
print(X_train_15.shape)
print(y_train_15.shape)

# y_validate_15 = y_validate_15[:, middle_word]

y_train_15 = to_categorical(y_train_15)
# y_validate_15 = to_categorical(y_validate_15)

print('X_train_15 ma wymiar: {}'.format(X_train_15.shape))
print('y_train_15 ma wymiar: {}'.format(y_train_15.shape))
# print('X_validate_15 ma wymiar: {}'.format(X_validate_15.shape))
# print('y_validate_15 ma wymiar: {}'.format(y_validate_15.shape))

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train_15, X_test_15, y_train_15, y_test_15 = train_test_split(X_train_15, y_train_15, test_size=0.2, shuffle=True, random_state=42)

# tu albo wczytanie sieci neuronowej z pliku, albo trenowanie od zera:
if load_nn_from_file:
    print('Wczytuję model sieci neuronowej z pliku')

    print('Wczytuję model middle')
    model_middle = load_model_from_files('model_PolEval_middle_GRU_bigger.json', 'model_PolEval_middle_GRU_bigger.hdf5')
    model_middle.compile(loss='mse', metrics=['accuracy'], optimizer=rmsprop)
    print('Wczytałem model middle')
else:
    print('wczytuję macierz Embedding...')
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_with_embedding_matrix, binary=False)
    embedding_matrix = read_embedding(file_with_embedding_matrix, embedding_dim, t, stopwords)
    embedding_matrix_morf = read_embedding(file_with_embedding_matrix, embedding_dim, t_morf, stopwords)
    print('Shape of embedding matrix: ', embedding_matrix.shape)

    model_middle = NER_PolEval_middle_GRU(seq_len_middle, tag_index, embedding_dim, word_index, embedding_matrix)
    history_middle = model_middle.fit(X_train_15, y_train_15, batch_size=16384, epochs=25,
                                      validation_data=(X_test_15, y_test_15), verbose=1)#, callbacks=callbacks_list)
    save_model_to_file(model_middle, 'PolEval_middle_GRU_bigger')


full_table = pd.DataFrame(columns=ground_truth_train.columns)
full_table_middle = pd.DataFrame(columns=ground_truth_train.columns)
k = 0

#test
for folder in os.listdir(folder_z_raportami_test):
    k = k+1
    print('Obliczam dane: {}/{}'.format(k, len(os.listdir(folder_z_raportami_test))))
    # print('Obliczam dane: {}/{}'.format(k, len(ground_truth_validate)))
    for file in os.listdir(os.path.join(folder_z_raportami_test,folder)):
        if file.endswith('.txt'):
            file_to_read = os.path.join(folder_z_raportami_test, folder, file)
            file = open(file_to_read, 'r', encoding='utf-8')
            full_raport = file.read()
            file.close()
            full_raport_in_words = t.texts_to_sequences([full_raport])
            texts_from_sequence = create_sequences(full_raport_in_words, seq_len_middle, step)
            predictions_15 = model_middle.predict(texts_from_sequence, batch_size=1024)
            row_middle = create_row(folder, full_raport_in_words, predictions_15, t, ground_truth_train.columns)
            full_table = full_table.append(row_middle)

full_table.to_csv('final_result_middle_GRU_bigger.tsv', sep='\t', index=False)