import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from collections import OrderedDict
# import PyPDF2
from copy import copy
from fuzzywuzzy import fuzz
from sklearn.metrics import classification_report, f1_score
import re


def compare_two_dataframes(ground_truth_dataframe, result_dataframe, tokenizer):
    ground_truth_dataframe.astype({'id': 'int32'})
    result_dataframe.astype({'id': 'int32'})
    ground_truth_dataframe.sort_values(by=['id'], inplace=True, ignore_index=True)
    result_dataframe.sort_values(by=['id'], inplace=True, ignore_index=True)

    print(ground_truth_dataframe.head())
    print(result_dataframe.head())

    # Poprawimy oryginalną ground_truth_table, żeby ulica nie zawierała numeru
    for j in range(len(ground_truth_dataframe)):
        if ground_truth_dataframe.street_no[j] in ground_truth_dataframe.street[j]:
            ground_truth_dataframe.street[j] = ground_truth_dataframe.street[j].replace(' '+ground_truth_dataframe.street_no[j], '')

    words_to_ignore = ['sa', 'SA', 'S.A.', 's.a.', 'spółka', 'akcyjna', 'Spółka', 'Akcyjna', 's', 'S', 'a', 'A', 'SPÓŁKA', 'AKCYJNA']
    for j in range(len(ground_truth_dataframe.company)):
        company_name = ground_truth_dataframe.company[j].lower().split()
        company_result = result_dataframe.company[j].lower().split()
        for word in words_to_ignore:
            if word in company_name:
                company_name.remove(word)
            if word in company_result:
                company_result.remove(word)
        # print(company_name)
        ground_truth_dataframe.company[j] = ' '.join(company_name)
        result_dataframe.company[j] = ' '.join(company_result)

    print(ground_truth_dataframe.head())
    print(result_dataframe.head())
    compare_dataframe = ground_truth_dataframe.eq(result_dataframe)
    # print(compare_dataframe)
    compare_columns = []
    print('zero-jedynkowo')
    for column in compare_dataframe:
        true_count = compare_dataframe[column].sum()
        compare_columns.append(true_count)
        print('W kolumnie {} jest {}/{} wartości pozytywnych, to daje {}'.format(column, true_count, len(compare_dataframe), true_count/len(compare_dataframe)))
    compare_columns.pop()
    evaluation_value = sum(compare_columns)/(compare_dataframe.shape[0]*(compare_dataframe.shape[1]-1))

    # for column_true, column_pred in zip(ground_truth_dataframe, result_dataframe):
        # print(ground_truth_dataframe[column_true])
        # print(result_dataframe[column_pred])
        # print(f1_score(ground_truth_dataframe[column_true].values, result_dataframe[column_pred].values, average=None))

    # print(' ')
    # print('fuzzy_ratio')
    for j in range(len(ground_truth_dataframe.columns)):
        fuzzy_ratio = 0
        for k in range(len(ground_truth_dataframe)):
            if fuzz.WRatio(ground_truth_dataframe.values[k,j], result_dataframe.values[k,j])>0.8:
                fuzzy_ratio = fuzzy_ratio+1
        # print('W kolumnie {} jest {}/{} wartości pozytywnych, to daje {}'.format(ground_truth_dataframe.columns[j], fuzzy_ratio, len(compare_dataframe), fuzzy_ratio / len(ground_truth_dataframe)))

    tmp_stat_employer = []
    tmp_stat_position = []
    for j in range(len(ground_truth_dataframe)):
        people_original = ground_truth_dataframe.people[j]
        employers_o, employers_in_tokens, positions_o, positions_in_tokens = people_finder(people_original, tokenizer)
        people_got = result_dataframe.people[j]
        employers_g, employers_in_tokens, positions_g, positions_in_tokens = people_finder(people_got, tokenizer)
        # print(len(employers_o))
        # print(employers_o)

        single_true_count_employer = 0
        single_true_count_position = 0
        for employer in employers_o:
            if employer in employers_g:
                single_true_count_employer += 1
        for position in positions_o:
            if position in positions_g:
                single_true_count_position += 1
        single_true_count_employer = single_true_count_employer/len(employers_o)
        single_true_count_position = single_true_count_position/len(positions_o)
        tmp_stat_employer.append(single_true_count_employer)
        tmp_stat_position.append(single_true_count_position)

    evaluation_value_employers = sum(tmp_stat_employer)/len(tmp_stat_employer)
    evaluation_value_position = sum(tmp_stat_position)/len(tmp_stat_position)
    return evaluation_value, evaluation_value_employers, evaluation_value_position


def read_embedding(file_with_embedding_matrix, embedding_dim, tokenizer, stopwords):
    word_index = tokenizer.word_index
    embeddings_index = {}
    f = open(file_with_embedding_matrix, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and word not in stopwords:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def find_long_name(predictions, predicted_tags, tokenizer, text, numbers):
    # numbers to indeksy, które będziemy sprawdzać.
    indeks = None
    for j in range(len(predicted_tags)):
        if predicted_tags[j] == numbers[0] and predictions[j,numbers[0]] == np.max(predictions[:,numbers[0]]):
            indeks = j
            sequence = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[j])].capitalize()
            # sequence = text[j].capitalize()
            # print(sequence)
            # print(text[j])
            # break
    if indeks is None:
        return 'UNKNOWN'
    # if numbers[1] not in set(predicted_tags):
    #     return sequence
    i = indeks+1
    count = 0
    while i < len(predicted_tags):
        if predicted_tags[i] == numbers[1]:
            count += 1
            i += 1
        else:
            break
    # print(count)
    for b in range(count):
        # sequence = sequence + ' ' + text[indeks + b + 1].capitalize()
        sequence = sequence + ' ' + list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[indeks + b + 1])].capitalize()
    # print(sequence)
    return sequence

def find_long_name_xgboost(predicted_tags, tokenizer, text, numbers):
    # numbers to indeksy, które będziemy sprawdzać.
    indeks = None
    for j in range(len(predicted_tags)):
        if predicted_tags[j] == numbers[0]:# and predictions[j,numbers[0]] == np.max(predictions[:,numbers[0]]):
            indeks = j
            sequence = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[j])].capitalize()
            # sequence = text[j].capitalize()
            # print(sequence)
            # print(text[j])
            # break
    if indeks is None:
        return 'UNKNOWN'
    # if numbers[1] not in set(predicted_tags):
    #     return sequence
    i = indeks+1
    count = 0
    while i < len(predicted_tags):
        if predicted_tags[i] == numbers[1]:
            count += 1
            i += 1
        else:
            break
    # print(count)
    for b in range(count):
        # sequence = sequence + ' ' + text[indeks + b + 1].capitalize()
        sequence = sequence + ' ' + list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[indeks + b + 1])].capitalize()
    # print(sequence)
    return sequence

def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key
    return 'no_key'

def create_row(folder, text_in_words, text, predictions, tokenizer, columns):
    text[0] = text[0][7:len(text[0])-8]
    print(predictions.shape)
    # print(len(text[0]))
    missing_indexes = [0, 4, 18, 19, 65, 67, 68]
    for ind in missing_indexes:
        predictions = np.insert(predictions, ind, 0, axis=1)

    # mapping = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16,
    #            16: 17, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31,
    #            29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44,
    #            42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57,
    #            55: 58, 56: 59, 57: 60, 58: 61, 59: 62, 60: 63, 61: 64, 62: 66, 63: 69}
    # mapping = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
    #            15: 17, 16: 20, 17: 21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26, 23: 27, 24: 28, 25: 29, 26: 30, 27: 31,
    #            28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44,
    #            41: 45, 42: 46, 43: 47, 44: 48, 45: 49, 46: 50, 47: 51, 48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57,
    #            54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 66, 62: 69}
    # new_tags = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    #             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    #             56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 69]


    predicted_tags = []
    for i in range(predictions.shape[0]):
        # print(predictions[i, :, :].shape)
        if predictions.shape == 3:
            new_row = np.argmax(predictions[i, :, :], axis=1)
        else:
            new_row = np.argmax(predictions[i, :])
        # print(new_row.shape)
        predicted_tags.append(new_row)
        # print('*')
        # print(predictions[i, :].shape)
        # print(new_row)
    # print(predicted_tags)
    # print(predicted_tags)
    predicted_tags = np.hstack(predicted_tags)
    # predicted_tags = [mapping[j] for j in predicted_tags]
    # predicted_tags = np.asarray(predicted_tags)
    # print(predicted_tags)
    print(predicted_tags)
    # print(type(predicted_tags))
    print(set(predicted_tags))

    row = pd.DataFrame(data=[len(columns)*['UNKNOWN']], columns=columns)
    row['id'] = str(folder)
    for j in range(len(predicted_tags)):
        if predicted_tags[j]==20:# and predictions[j,20]==np.max(predictions[:,20]):
            # print('znalazłem miasto!')
            # print(text[0][j])
            word = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # word = text[0][j]
            print(word)
            print(predictions[j,20])
            row['city'] = word.capitalize()
            break

    months = {'styczeń': '01', 'luty': '02', 'marzec': '03', 'kwiecień': '04', 'maj': '05', 'czerwiec': '06', 'lipiec': '07', 'sierpień': '08', 'wrzesień': '09', 'październik': '10', 'listopad': '11',
              'grudzień': '12', 'stycznia': '01', 'lutego': '02', 'marca': '03', 'kwietnia': '04', 'maja': '05', 'czerwca': '06', 'lipca': '07', 'sierpnia': '08', 'września': '09',
              'października': '10', 'listopada': '11', 'grudnia': '12'}

    drawing_date = 'UNKNOWN'
    period_to_year = 3000

    for j in range(len(predicted_tags)-3):
        if (predicted_tags[j:j+3]==[7, 8, 9]).all() and (predictions[j,7]==np.max(predictions[:,7]) or predictions[j,8]==np.max(predictions[:,8]) or predictions[j,9]==np.max(predictions[:,9])):
            # print('znalazłem drawing_date')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j+2]
            # month = text[0][j+1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['drawing_date']= str(year) + '-' + str(month) + '-' + str(day)
            drawing_date = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j+3]==[9, 8, 7]).all() and (predictions[j,7]==np.max(predictions[:,7]) or predictions[j,8]==np.max(predictions[:,8]) or predictions[j,9]==np.max(predictions[:,9])):
            # print('znalazłem drawing_date')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['drawing_date'] = str(year) + '-' + str(month) + '-' + str(day)
            drawing_date = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [13, 14, 15]).all() and (predictions[j,13]==np.max(predictions[:,13]) or predictions[j,14]==np.max(predictions[:,14]) or predictions[j,15]==np.max(predictions[:,15])):
            # print('znalazłem period_to')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j + 2]
            # month = text[0][j + 1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)
            period_to_year = year
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [15, 14, 13]).all() and (predictions[j, 13] == np.max(predictions[:, 13]) or predictions[j, 14] == np.max(predictions[:, 14]) or predictions[j, 15] == np.max(predictions[:, 15])):
            # print('znalazłem period_to')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)
            period_to_year = year
            # print(str(day) + '-' + str(month) + '-' + str(year))
    for j in range(len(predicted_tags)-3):
        if (predicted_tags[j:j + 3] == [10, 11, 12]).all() and (predictions[j,10]==np.max(predictions[:,10]) or predictions[j,11]==np.max(predictions[:,11]) or predictions[j,12]==np.max(predictions[:,12]))\
                and int(predicted_tags[j+2])<=int(period_to_year):
            # print('znalazłem period_from')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j + 2]
            # month = text[0][j + 1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [12, 11, 10]).all() and (predictions[j, 10] == np.max(predictions[:, 10]) or predictions[j, 11] == np.max(predictions[:, 11])
                                                                  or predictions[j, 12] == np.max(predictions[:, 12])) and int(predicted_tags[j])<=int(period_to_year):
            # print('znalazłem period_from')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))

    if drawing_date=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==7 and predictions[j,7]==np.max(predictions[:,7]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==8 and predictions[j,8]==np.max(predictions[:,9]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==9 and predictions[j,9]==np.max(predictions[:,9]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year  = text[0][j]
        row['drawing_date'] = str(year) + '-' + str(month) + '-' + str(day)
        drawing_date = str(year) + '-' + str(month) + '-' + str(day)

    if row['period_from'].values=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==10 and predictions[j,10]==np.max(predictions[:,10]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==11 and predictions[j,11]==np.max(predictions[:,11]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==12 and predictions[j,12]==np.max(predictions[:,12]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year = text[0][j]
        row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)

    if row['period_to'].values=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==13 and predictions[j,13]==np.max(predictions[:,13]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==14 and predictions[j,14]==np.max(predictions[:,14]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==15 and predictions[j,15]==np.max(predictions[:,15]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year = text[0][j]
        row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)

    for j in range(len(predicted_tags)-2):
        if (predicted_tags[j:j+2]==[16, 17]).all() and (predictions[j,16]==np.max(predictions[:,16]) or predictions[j,17]==np.max(predictions[:,17])):
            # print('znalazłem kod pocztowy')
            pre = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            post = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+1])]
            # pre = text[0][j]
            # post = text[0][j+1]
            row['postal_code'] = str(pre) + '-' + str(post)
            # print(str(pre) + '-' + str(post))
    if row['postal_code'].values=='UNKNOWN':
        row['postal_code'] = '00-000'

    street = find_long_name(predictions, predicted_tags, tokenizer, text[0], [1, 2])
    skroty = ["ul ", "UL ", "Os ", "os ", "al ", "Al ", "pl ", "Pl ", "s "]
    for skrot in skroty:
        if skrot in street:
            nowy_skrot = skrot.replace(' ', '') + '. '
            street.replace(skrot, nowy_skrot)
            break
    row['street'] = street.title()

    street_no = find_long_name(predictions, predicted_tags, tokenizer, text[0], [3, 4])
    row['street_no'] = street_no.title()

    company = find_long_name(predictions, predicted_tags, tokenizer, text[0], [5, 6])
    SA = [' SA', ' sa', 'Spółka Akcyjna', 'spółka akcyjna', 'Sa']
    S = [' S', ' s']
    spolka = [' Spółka', ' spółka']
    flaga = False
    if company=='UNKNOWN':
        flaga = True
    while not flaga:
        for word in SA:
            if word in company:
                flaga = True
                break
        for word in S:
            if company[-2:] == word:
                company = company.title() + 'A'
                flaga = True
                break
        for word in spolka:
            if company[-7:] == word:
                company = company.title() + ' Akcyjna'
                flaga = True
                break
        if not flaga:
            company = company.title() + ' Spółka Akcyjna'
            flaga = True
    print(company)
    row['company'] = company

    people = "["
    for ind in range(12):
        # print(ind)
        human = find_long_name(predictions, predicted_tags, tokenizer, text[0], [21+ind*4, 22+ind*4]).title()
        position = find_long_name(predictions, predicted_tags, tokenizer, text[0], [24+ind*4, 23+ind*4]).title()
        if human != 'Unknown':
            people = people + "('" + drawing_date + "', '" + human.title() + "', '" + position.title() + "')"
        else:
            break
    people = people + "]"
    people = people.replace(')(', '), (')
    # print(people)
    row['people'] = people

    # print(row)
    return row

def create_row_xgboost(folder, text_in_words, text, predicted_tags, tokenizer, columns):
    text[0] = text[0][7:len(text[0])-8]
    # print(len(text[0]))

    '''
    predicted_tags = []
    for i in range(predictions.shape[0]):
        # print(predictions[i, :, :].shape)
        if predictions.shape == 3:
            new_row = np.argmax(predictions[i, :, :], axis=1)
        else:
            new_row = np.argmax(predictions[i, :])
        # print(new_row.shape)
        predicted_tags.append(new_row)
        # print('*')
        # print(predictions[i, :].shape)
        # print(new_row)
    # print(predicted_tags)
    # print(predicted_tags)
    predicted_tags = np.hstack(predicted_tags)
    # print(set(predicted_tags))
    '''

    row = pd.DataFrame(data=[len(columns)*['UNKNOWN']], columns=columns)
    row['id'] = str(folder)
    for j in range(len(predicted_tags)):
        if predicted_tags[j]==20:# and predictions[j,20]==np.max(predictions[:,20]):
            # print('znalazłem miasto!')
            # print(text[0][j])
            word = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # word = text[0][j]
            print(word)
            row['city'] = word.capitalize()
            break

    months = {'styczeń': '01', 'luty': '02', 'marzec': '03', 'kwiecień': '04', 'maj': '05', 'czerwiec': '06', 'lipiec': '07', 'sierpień': '08', 'wrzesień': '09', 'październik': '10', 'listopad': '11',
              'grudzień': '12', 'stycznia': '01', 'lutego': '02', 'marca': '03', 'kwietnia': '04', 'maja': '05', 'czerwca': '06', 'lipca': '07', 'sierpnia': '08', 'września': '09',
              'października': '10', 'listopada': '11', 'grudnia': '12'}

    drawing_date = 'UNKNOWN'
    period_to_year = 3000

    for j in range(len(predicted_tags)-3):
        if (predicted_tags[j:j+3]==[7, 8, 9]).all():# and (predictions[j,7]==np.max(predictions[:,7]) or predictions[j,8]==np.max(predictions[:,8]) or predictions[j,9]==np.max(predictions[:,9])):
            # print('znalazłem drawing_date')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j+2]
            # month = text[0][j+1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['drawing_date']= str(year) + '-' + str(month) + '-' + str(day)
            drawing_date = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j+3]==[9, 8, 7]).all() :#and (predictions[j,7]==np.max(predictions[:,7]) or predictions[j,8]==np.max(predictions[:,8]) or predictions[j,9]==np.max(predictions[:,9])):
            # print('znalazłem drawing_date')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['drawing_date'] = str(year) + '-' + str(month) + '-' + str(day)
            drawing_date = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [13, 14, 15]).all():# and (predictions[j,13]==np.max(predictions[:,13]) or predictions[j,14]==np.max(predictions[:,14]) or predictions[j,15]==np.max(predictions[:,15])):
            # print('znalazłem period_to')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j + 2]
            # month = text[0][j + 1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)
            period_to_year = year
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [15, 14, 13]).all():# and (predictions[j, 13] == np.max(predictions[:, 13]) or predictions[j, 14] == np.max(predictions[:, 14]) or predictions[j, 15] == np.max(predictions[:, 15])):
            # print('znalazłem period_to')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)
            period_to_year = year
            # print(str(day) + '-' + str(month) + '-' + str(year))
    for j in range(len(predicted_tags)-3):
        if (predicted_tags[j:j + 3] == [10, 11, 12]).all():# and (predictions[j,10]==np.max(predictions[:,10]) or predictions[j,11]==np.max(predictions[:,11]) or predictions[j,12]==np.max(predictions[:,12]))\
                #and int(predicted_tags[j+2])<=int(period_to_year):
            # print('znalazłem period_from')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            # year = text[0][j + 2]
            # month = text[0][j + 1]
            # day = text[0][j]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))
        elif (predicted_tags[j:j + 3] == [12, 11, 10]).all():# and (predictions[j, 10] == np.max(predictions[:, 10]) or predictions[j, 11] == np.max(predictions[:, 11])
                                                             #     or predictions[j, 12] == np.max(predictions[:, 12])) and int(predicted_tags[j])<=int(period_to_year):
            # print('znalazłem period_from')
            year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 1])]
            day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j + 2])]
            # year = text[0][j]
            # month = text[0][j + 1]
            # day = text[0][j + 2]
            if month in months:
                month = months[month]
            if len(day) == 1:
                day = '0' + day
            row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)
            # print(str(day) + '-' + str(month) + '-' + str(year))

    if drawing_date=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==7:# and predictions[j,7]==np.max(predictions[:,7]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==8:# and predictions[j,8]==np.max(predictions[:,9]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==9:# and predictions[j,9]==np.max(predictions[:,9]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year  = text[0][j]
        row['drawing_date'] = str(year) + '-' + str(month) + '-' + str(day)
        drawing_date = str(year) + '-' + str(month) + '-' + str(day)

    if row['period_from'].values=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==10:# and predictions[j,10]==np.max(predictions[:,10]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==11:# and predictions[j,11]==np.max(predictions[:,11]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==12:# and predictions[j,12]==np.max(predictions[:,12]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year = text[0][j]
        row['period_from'] = str(year) + '-' + str(month) + '-' + str(day)

    if row['period_to'].values=='UNKNOWN':
        day = '01'
        month = '01'
        year = '2000'
        for j in range(len(predicted_tags)):
            if predicted_tags[j]==13:# and predictions[j,13]==np.max(predictions[:,13]):
                day = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # day = text[0][j]
            elif predicted_tags[j]==14:# and predictions[j,14]==np.max(predictions[:,14]):
                month = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # month = text[0][j]
            elif predicted_tags[j]==15:# and predictions[j,15]==np.max(predictions[:,15]):
                year = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
                # year = text[0][j]
        row['period_to'] = str(year) + '-' + str(month) + '-' + str(day)

    for j in range(len(predicted_tags)-2):
        if (predicted_tags[j:j+2]==[16, 17]).all():# and (predictions[j,16]==np.max(predictions[:,16]) or predictions[j,17]==np.max(predictions[:,17])):
            # print('znalazłem kod pocztowy')
            pre = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j])]
            post = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(text[0][j+1])]
            # pre = text[0][j]
            # post = text[0][j+1]
            row['postal_code'] = str(pre) + '-' + str(post)
            # print(str(pre) + '-' + str(post))
    if row['postal_code'].values=='UNKNOWN':
        row['postal_code'] = '00-000'

    street = find_long_name_xgboost(predicted_tags, tokenizer, text[0], [1, 2])
    skroty = ["ul ", "UL ", "Os ", "os ", "al ", "Al ", "pl ", "Pl ", "s "]
    for skrot in skroty:
        if skrot in street:
            nowy_skrot = skrot.replace(' ', '') + '. '
            street.replace(skrot, nowy_skrot)
            break
    row['street'] = street.title()

    street_no = find_long_name_xgboost(predicted_tags, tokenizer, text[0], [3, 4])
    row['street_no'] = street_no.title()

    company = find_long_name_xgboost(predicted_tags, tokenizer, text[0], [5, 6])
    SA = [' SA', ' sa', 'Spółka Akcyjna', 'spółka akcyjna', 'Sa']
    S = [' S', ' s']
    spolka = [' Spółka', ' spółka']
    flaga = False
    if company=='UNKNOWN':
        flaga = True
    while not flaga:
        for word in SA:
            if word in company:
                flaga = True
                break
        for word in S:
            if company[-2:] == word:
                company = company.title() + 'A'
                flaga = True
                break
        for word in spolka:
            if company[-7:] == word:
                company = company.title() + ' Akcyjna'
                flaga = True
                break
        if not flaga:
            company = company.title() + ' Spółka Akcyjna'
            flaga = True
    print(company)
    row['company'] = company

    people = "["
    for ind in range(12):
        # print(ind)
        human = find_long_name_xgboost(predicted_tags, tokenizer, text[0], [21+ind*4, 22+ind*4]).title()
        position = find_long_name_xgboost(predicted_tags, tokenizer, text[0], [24+ind*4, 23+ind*4]).title()
        if human != 'Unknown':
            people = people + "('" + drawing_date + "', '" + human.title() + "', '" + position.title() + "')"
        else:
            break
    people = people + "]"
    people = people.replace(')(', '), (')
    # print(people)
    row['people'] = people

    # print(row)
    return row

def complete_row(row, full_raport, all_postal_codes):
    if row.postal_code.values[0] == "00-000":
        # print(row)
        pattern = r'\d{2}-\d{3}'
        postal_codes = re.findall(pattern, full_raport)
        # print(postal_codes)
        if len(postal_codes) == 1 or len(list(set(postal_codes))) == 1:
            row['postal_code'] = postal_codes[0]
        # print(row)
        elif len(postal_codes) == 0:
            print('W tym dokumencie nie ma żadnych kodów pocztowych')
        else:
            flaga = False
            for j in range(len(all_postal_codes)):
                # print(row["city"].values[0])
                # print(all_postal_codes.iloc[j]["MIEJSCOWOŚĆ"])
                # print(row["street"].values[0])
                # print(all_postal_codes.iloc[j]["ADRES"])
                if row["city"].values[0].lower() == all_postal_codes.iloc[j]["MIEJSCOWOŚĆ"].lower() and row["street"].values[0].lower() in all_postal_codes.iloc[j]["ADRES"].lower():
                    row["postal_code"] = all_postal_codes.iloc[j]["KOD POCZTOWY"]
                    flaga = True
                    break
            if not flaga:
                flaga_tmp = False
                for j in range(len(full_raport)):
                    if row["company"].values[0] == full_raport[j]:
                        for postal_code in postal_codes:
                            if postal_code in full_raport[j:j+15]:
                                row["postal_code"] = postal_code
                                flaga_tmp = True
                                break
                    if flaga_tmp:
                        break

    if row.city.values[0] == "UNKNOWN" and row.postal_code.values[0]!='00-000':
        # print(row)
        # print(row.postal_code.values)
        ind = all_postal_codes.index[all_postal_codes['KOD POCZTOWY'] == row.postal_code.values[0]].tolist()
        # print(ind)
        lista_miast = []
        if len(ind)==1 or len(list(set(ind)))==1:
            row['city'] = all_postal_codes['MIEJSCOWOŚĆ'].loc[ind]
        elif len(ind) > 1:
            for single_ind in ind:
                lista_miast.append(all_postal_codes["MIEJSCOWOŚĆ"].loc[single_ind])
            if len(set(lista_miast))==1:
                row['city'] = set(lista_miast)[0]
        # print(row)
    return row

def create_sequences(full_sequence, seq_len, step):
    sequences = []
    # print(full_sequence[0])
    for i in range(0, len(full_sequence[0])-seq_len, step):
        # print(full_sequence[0])
        sequences.append(full_sequence[0][i:i+seq_len])
    return sequences

def company_names(company_name_from_ground_truth, tokenizer):
    company_name_from_ground_truth = text_to_word_sequence(company_name_from_ground_truth)
    company_name_skip_sa = copy(company_name_from_ground_truth)
    # do rozważenia:
    words_to_ignore = ['sa', 'SA', 'S.A.', 's.a.', 'spółka', 'akcyjna', 'Spółka', 'Akcyjna', 's', 'S', 'a', 'A', 'SPÓŁKA', 'AKCYJNA', 's.', 'a.', 'S.', 'A.']
    for word in words_to_ignore:
        if word in company_name_from_ground_truth:
            company_name_skip_sa.remove(word)
    # print(company_name_skip_sa)
    company_names_in_tokens = tokenizer.texts_to_sequences([company_name_from_ground_truth])
    company_name_skip_sa_in_tokens = tokenizer.texts_to_sequences([company_name_skip_sa])

    return company_name_from_ground_truth, company_names_in_tokens, company_name_skip_sa, company_name_skip_sa_in_tokens

def city_name(city, tokenizer):
    city = text_to_word_sequence(city)
    city_in_tokens = tokenizer.texts_to_sequences(city)
    return city, city_in_tokens

def street_name(street_name, street_no, tokenizer):
    # do rozważenia
    # to_remove_from_address = ['ul.', 'ulica', 'Ul.', 'Ulica', 'pl.', 'plac', 'Plac', 'Pl.']
    # for word in to_remove_from_address:
    #     if word in street_name:
    #         street_name.replace(word, '')

    street_name_in_tokens = tokenizer.texts_to_sequences([street_name])
    street_no_in_tokens = tokenizer.texts_to_sequences([street_no])
    if street_no_in_tokens == street_name_in_tokens[-1:-1 - len(street_no_in_tokens)]:
        street = street_name
    else:
        street = street_name.replace(street_no, '')
    # print(street)
    street_in_tokens = tokenizer.texts_to_sequences([street])
    # print(street_in_tokens)

    return street, street_no, street_in_tokens, street_no_in_tokens

def people_finder(people, tokenizer):
    # print(people)
    people = people.replace('(', '')
    people = people.replace(')', '')
    people = people.replace('[', '')
    people = people.replace(']', '')
    people = people.split("', '")
    # print(people)
    employer = []
    position = []
    for k in range(1, len(people), 3):
        employer.append(people[k].replace("'", '').lower())
        position.append(people[k + 1].replace("'", '').lower())

    people_in_tokens = []
    position_in_tokens = []
    for item in employer:
        people_in_tokens.append(tokenizer.texts_to_sequences([item]))
    for item in position:
        position_in_tokens.append(tokenizer.texts_to_sequences([item]))

    flat_list = []
    for sublist in people_in_tokens:
        for item in sublist:
            flat_list.append(item)
    people_in_tokens = flat_list
    # print(people_in_tokens)

    flat_list = []
    for sublist in position_in_tokens:
        for item in sublist:
            flat_list.append(item)
    position_in_tokens = flat_list
    # print(position_in_tokens)

    return employer, people_in_tokens, position, position_in_tokens

def prepare_data(file, folder_z_raportami, t, tags_tokenizer, seq_len, step, data_type):
    ground_truth_table = pd.read_csv(file, sep=';')
    # żeby było łatwiej analizować daty, to rozbijemy je na dzień, miesiąc i rok - bo tak też odbędzie się tokenizacja naszego tekstu,
    # poza tym widzę, że raczej nie ma dat w formacie dd.mm.rrrr a raczej dd miesiąc rok
    months = {'01': 'stycznia', '02': 'lutego', '03': 'marca', '04': 'kwietnia', '05': 'maja', '06': 'czerwca',
              '07': 'lipca',
              '08': 'sierpnia', '09': 'września', '10': 'października', '11': 'listopada', '12': 'grudnia'}
    months_simple = {'01': 'styczeń', '02': 'luty', '03': 'marzec', '04': 'kwiecień', '05': 'maj', '06': 'czerwiec',
                     '07': 'lipiec',
                     '08': 'sierpień', '09': 'wrzesień', '10': 'październik', '11': 'listopad', '12': 'grudzień'}
    # print(ground_truth_table.head())
    drawing_date_day = pd.Series([], dtype=str)
    drawing_date_month_no = pd.Series([], dtype=str)
    drawing_date_month = pd.Series([], dtype=str)
    drawing_date_month_simple = pd.Series([], dtype=str)
    drawing_date_year = pd.Series([], dtype=str)
    period_from_day = pd.Series([], dtype=str)
    period_from_month_no = pd.Series([], dtype=str)
    period_from_month = pd.Series([], dtype=str)
    period_from_month_simple = pd.Series([], dtype=str)
    period_from_year = pd.Series([], dtype=str)
    period_to_day = pd.Series([], dtype=str)
    period_to_month_no = pd.Series([], dtype=str)
    period_to_month = pd.Series([], dtype=str)
    period_to_month_simple = pd.Series([], dtype=str)
    period_to_year = pd.Series([], dtype=str)
    postal_code_pre = pd.Series([], dtype=str)
    postal_code_post = pd.Series([], dtype=str)
    for i in range(len(ground_truth_table)):
        # print(ground_truth_train['drawing_date'][i])
        drawing_date_day[i] = ground_truth_table['drawing_date'][i][8:10]
        drawing_date_month_no[i] = ground_truth_table['drawing_date'][i][5:7]
        drawing_date_month[i] = months[ground_truth_table['drawing_date'][i][5:7]]
        drawing_date_month_simple[i] = months_simple[ground_truth_table['drawing_date'][i][5:7]]
        drawing_date_year[i] = ground_truth_table['drawing_date'][i][0:4]
        period_from_day[i] = ground_truth_table['period_from'][i][8:10]
        period_from_month_no[i] = ground_truth_table['period_from'][i][5:7]
        period_from_month[i] = months[ground_truth_table['period_from'][i][5:7]]
        period_from_month_simple[i] = months_simple[ground_truth_table['period_from'][i][5:7]]
        period_from_year[i] = ground_truth_table['period_from'][i][0:4]
        period_to_day[i] = ground_truth_table['period_to'][i][8:10]
        period_to_month_no[i] = ground_truth_table['period_to'][i][5:7]
        period_to_month[i] = months[ground_truth_table['period_to'][i][5:7]]
        period_to_month_simple[i] = months_simple[ground_truth_table['period_to'][i][5:7]]
        period_to_year[i] = ground_truth_table['period_to'][i][0:4]
        postal_code_pre[i] = ground_truth_table['postal_code'][i][0:2]
        postal_code_post[i] = ground_truth_table['postal_code'][i][3:6]
    ground_truth_table.insert(3, 'drawing_date_day', drawing_date_day)
    ground_truth_table.insert(4, 'drawing_date_month_no', drawing_date_month_no)
    ground_truth_table.insert(5, 'drawing_date_month', drawing_date_month)
    ground_truth_table.insert(6, 'drawing_date_month_simple', drawing_date_month_simple)
    ground_truth_table.insert(7, 'drawing_date_year', drawing_date_year)

    ground_truth_table.insert(9, 'period_from_day', period_from_day)
    ground_truth_table.insert(10, 'period_from_month_no', period_from_month_no)
    ground_truth_table.insert(11, 'period_from_month', period_from_month)
    ground_truth_table.insert(12, 'period_from_month_simple', period_from_month_simple)
    ground_truth_table.insert(13, 'period_from_year', period_from_year)

    ground_truth_table.insert(15, 'period_to_day', period_to_day)
    ground_truth_table.insert(16, 'period_to_month_no', period_to_month_no)
    ground_truth_table.insert(17, 'period_to_month', period_to_month)
    ground_truth_table.insert(18, 'period_to_month_simple', period_to_month_simple)
    ground_truth_table.insert(19, 'period_to_year', period_to_year)

    ground_truth_table.insert(21, 'postal_code_pre', postal_code_pre)
    ground_truth_table.insert(22, 'postal_code_post', postal_code_post)

    # print(tags_tokenizer.texts_to_sequences(['o']))

    seq_to_train = []
    seq_to_return = []
    max_numer_pracownika = -1
    for i in ground_truth_table.id:
        tmp_index = ground_truth_table.index[ground_truth_table.id == i].tolist()[0]
        # print(tmp_index)
        company_name = ground_truth_table.company[tmp_index]
        company_name, company_name_in_tokens, company_name_skip_sa, company_name_skip_sa_in_tokens = company_names(company_name, t)
        # print(company_name_in_tokens)

        city = ground_truth_table.city[tmp_index]
        city, city_in_tokens = city_name(city, t)

        street_name_tmp = ground_truth_table.street[tmp_index]
        street_no = ground_truth_table.street_no[tmp_index]
        street, street_no, street_in_tokens, street_no_in_tokens = street_name(street_name_tmp, street_no, t)

        people = ground_truth_table.people[tmp_index]
        employers, employers_in_tokens, positions, positions_in_tokens = people_finder(people, t)
        if len(employers)>max_numer_pracownika:
            max_numer_pracownika = len(employers)

        print('przygotowuję dane: {}/{}'.format(tmp_index + 1, len(ground_truth_table)))
        folder = os.path.join(folder_z_raportami, str(i))
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                file_to_read = os.path.join(folder, file)
                # print(file_to_read)
                # try:
                file = open(file_to_read, 'r', encoding='utf-8')
                full_raport = file.read()
                file.close()
                # print(full_raport)
                full_raport_in_words = t.texts_to_sequences([full_raport])
                # print(full_raport_in_words)
                texts_from_sequence = t.sequences_to_texts(full_raport_in_words)[0].split(" ")
                sequence_to_return = ['o'] * len(texts_from_sequence)

                numer_pracownika = -1
                for employer, position in zip(employers_in_tokens, positions_in_tokens):
                    numer_pracownika += 1
                    # print(employer, position)
                    for j in range(len(texts_from_sequence) - len(employer) - len(position)):
                        text_part = full_raport_in_words[0][j:j + len(employer) + len(position)]
                        if (text_part == employer + position) and (
                                sequence_to_return[j:j + len(employer) + len(position)] == ['o'] * (
                                len(employer) + len(position))):
                            # print('znalazłem pracownika i jego zawód')
                            sequence_to_return[j] = 'human_' + str(numer_pracownika) + '_start'
                            for k in range(j + 1, j + len(employer)):
                                sequence_to_return[k] = 'human_' + str(numer_pracownika) + '_continue'
                            sequence_to_return[j + len(employer)] = 'position_' + str(numer_pracownika) + '_start'
                            for k in range(j + 1 + len(employer), j + len(employer) + len(position)):
                                sequence_to_return[k] = 'position_' + str(numer_pracownika) + '_continue'
                            # print(sequence_to_return[j:j + len(employer) + len(position)+1])

                numer_pracownika = -1
                for employer in employers_in_tokens:
                    # print(employer)
                    numer_pracownika += 1
                    for j in range(len(texts_from_sequence)-len(employer)):
                        text_part = full_raport_in_words[0][j:j + len(employer)]
                        # print('*')
                        # print(text_part)
                        # print(employer)
                        # print(sequence_to_return[j:j+len(employer)])
                        # print(['o']*len(employer))
                        if (text_part == employer) and (sequence_to_return[j:j+len(employer)]==['o']*len(employer)):
                            # print('Znalazlem samego pracownika')
                            sequence_to_return[j] = 'human_' + str(numer_pracownika) + '_start'
                            for k in range(j + 1, j + len(employer)):
                                sequence_to_return[k] = 'human_' + str(numer_pracownika) + '_continue'
                            # print(sequence_to_return[j:j + len(employer)+1])
                numer_pracownika = -1
                for position in positions_in_tokens:
                    numer_pracownika += 1
                    for j in range(len(texts_from_sequence)-len(position)):
                        text_part = full_raport_in_words[0][j:j + len(position)]
                        if (text_part == position) and (sequence_to_return[j:j+len(position)]==['o']*len(position)):
                            # print('Znalazłem samą pozycję')
                            sequence_to_return[j] = 'position_' + str(numer_pracownika) + '_start'
                            for k in range(j + 1, j + len(position)):
                                sequence_to_return[k] = 'position_' + str(numer_pracownika) + '_continue'
                            # print(sequence_to_return[j:j + len(position)])
                # print(sequence_to_return)
                for j in range(len(texts_from_sequence)-len(city_in_tokens[0])):
                    # print(full_raport_in_words[0][j:j + len(city_in_tokens[0])])
                    text_part = full_raport_in_words[0][j:j + len(city_in_tokens[0])]

                    if (text_part == city_in_tokens[0]) and (len(city_in_tokens[0]) == 1):
                        # print('znalazłem miasto')
                        sequence_to_return[j] = 'city'
                    elif text_part == city_in_tokens[0]:
                        # print('znalazłem miasto wieloczłonowe')
                        sequence_to_return[j] = 'city_start'
                        for k in range(j+1,j+len(city_in_tokens[0])):
                            sequence_to_return[k] = 'city_continue'

                for j in range((len(texts_from_sequence) - 2)):
                    # print('Szukam dat')
                    word_1 = texts_from_sequence[j]
                    word_2 = texts_from_sequence[j + 1]
                    word_3 = texts_from_sequence[j + 2]
                    # print(ground_truth_train.period_to_month_no[tmp_index])
                    if ((str(word_1) == str(int(ground_truth_table.drawing_date_day.values[tmp_index])))
                        or (str(word_1) == str(ground_truth_table.drawing_date_day.values[tmp_index]))) & \
                            ((str(word_2) == str(ground_truth_table.drawing_date_month.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.drawing_date_month_simple.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.drawing_date_month_no.values[tmp_index]))) & \
                            (str(word_3) == str(ground_truth_table.drawing_date_year.values[tmp_index])):
                        # print('znalazłem datę - drawing date')
                        sequence_to_return[j] = 'drawing_date_day'
                        sequence_to_return[j + 1] = 'drawing_date_month'
                        sequence_to_return[j + 2] = 'drawing_date_year'
                    elif ((str(word_1) == str(int(ground_truth_table.period_from_day.values[tmp_index])))
                          or (str(word_1) == str(ground_truth_table.period_from_day.values[tmp_index]))) & \
                            ((str(word_2) == str(ground_truth_table.period_from_month.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.period_from_month_no.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.period_from_month_simple.values[tmp_index]))) & \
                            (str(word_3) == str(ground_truth_table.period_from_year.values[tmp_index])):
                        # print('znalazłem datę - period from')
                        sequence_to_return[j] = 'period_from_day'
                        sequence_to_return[j + 1] = 'period_from_month'
                        sequence_to_return[j + 2] = 'period_from_year'
                    elif ((str(word_1) == str(int(ground_truth_table.period_to_day.values[tmp_index])))
                          or (str(word_1) == str(ground_truth_table.period_to_day.values[tmp_index]))) & \
                            ((str(word_2) == str(ground_truth_table.period_to_month.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.period_to_month_no.values[tmp_index]))
                             or (str(word_2) == str(ground_truth_table.period_to_month_simple.values[tmp_index]))) & \
                            (str(word_3) == str(ground_truth_table.period_to_year.values[tmp_index])):
                        # print('znalazłem datę - period to')
                        sequence_to_return[j] = 'period_to_day'
                        sequence_to_return[j + 1] = 'period_to_month'
                        sequence_to_return[j + 2] = 'period_to_year'

                for j in range(len(texts_from_sequence) - 1):
                    # print('Szukam kodu pocztowego')
                    word_1 = texts_from_sequence[j]
                    word_2 = texts_from_sequence[j + 1]
                    if (word_1 == ground_truth_table.postal_code_pre[tmp_index]) & (
                            word_2 == postal_code_post[tmp_index]):
                        # print('Znalazłem kod pocztowy!')
                        sequence_to_return[j] = 'postal_code_pre'
                        sequence_to_return[j + 1] = 'postal_code_post'
                # print(company_name_in_tokens)
                # for j in range(len(texts_from_sequence) - len(company_name_in_tokens[0])):
                #     # print('Szukam nazwy firmy')
                #     text_part = full_raport_in_words[0][j:j + len(company_name_in_tokens[0])]
                #     if (text_part == company_name_in_tokens[0]):
                #         # print('znalazłem nazwę firmy!')
                #         sequence_to_return[j] = 'company_start'
                #         for k in range(j + 1, j + len(company_name_in_tokens[0])):
                #             sequence_to_return[k] = 'company_continue'
                #         sequence_to_return[j + len(company_name_in_tokens[0])] = 'company_continue'
                #         # print(sequence_to_return[j:j+len(company_name_in_tokens[0])])
                # # print(company_name_skip_sa)
                for j in range(len(texts_from_sequence) - len(company_name_skip_sa_in_tokens[0])):
                    # print('Szukam nazwy firmy')
                    text_part = full_raport_in_words[0][j:j + len(company_name_skip_sa_in_tokens[0])]
                    if text_part == company_name_skip_sa_in_tokens[0]:# and sequence_to_return[j:j+len(company_name_skip_sa_in_tokens)]==['o']*len(company_name_skip_sa_in_tokens):
                        # print('znalazłem nazwę firmy!')
                        sequence_to_return[j] = 'company_start'
                        for k in range(j + 1, j + len(company_name_skip_sa_in_tokens[0])):
                            sequence_to_return[k] = 'company_continue'
                        sequence_to_return[j + len(company_name_skip_sa_in_tokens[0])] = 'company_continue'
                        # print(sequence_to_return[j:j+len(company_name_skip_sa_in_tokens[0])])

                for j in range(len(texts_from_sequence) - len(street_in_tokens[0]) - len(street_no_in_tokens[0])):
                    # print('Szukam adresu')
                    text_part = full_raport_in_words[0][j:j + len(street_in_tokens[0]) + len(street_no_in_tokens[0])]
                    if text_part == street_in_tokens[0] + street_no_in_tokens[0]:
                        # print('znalazłem adres!')
                        sequence_to_return[j] = 'street_start'
                        # print(j, j+1, j+1+len(street_in_tokens[0]))
                        # street_in_tokens[0]
                        if len(street_in_tokens[0]) > 1:
                            for k in range(j + 1, j + len(street_in_tokens[0])):
                                sequence_to_return[k] = 'street_continue'
                        sequence_to_return[j+len(street_in_tokens[0])] = 'street_no_start'
                        if len(street_no_in_tokens) > 1:
                            for k in range(j + 1 + len(street_in_tokens[0]), j + len(street_no_in_tokens[0]) + len(street_in_tokens[0])):
                                sequence_to_return[k] = 'street_no_continue'

                # print(sequence_to_return)
                texts_from_sequence = create_sequences(full_raport_in_words, seq_len, step)
                # print(sequence_to_return)
                sequence_to_return = tags_tokenizer.texts_to_sequences([sequence_to_return])
                # print(len(sequence_to_return))
                # print(sequence_to_return)
                # print(np.unique(sequence_to_return))
                # print(len(sequence_to_return[0]))
                # print(len(full_raport_in_words[0]))
                if len(sequence_to_return) != len(full_raport_in_words):
                    print('error')
                    print(len(sequence_to_return))
                    print(len(full_raport_in_words))
                sequence_to_return = create_sequences(sequence_to_return, seq_len, step)
                seq_to_return.extend(sequence_to_return)
                seq_to_train.extend(texts_from_sequence)

    X_train = np.asarray(seq_to_train)
    # print(X_train.shape)

    y_train = np.asarray(seq_to_return)
    # print(y_train.shape)

    X_file = 'X_' + data_type
    y_file = 'y_' + data_type
    np.save(X_file, X_train)
    np.save(y_file, y_train)

    print('Najwięcej pracowników w zbiorze {} to: {}'. format(data_type, max_numer_pracownika))

    return X_train, y_train


def prepare_data_with_morfeusz(file, folder_z_raportami, t_token, t_basic, tags_tokenizer, seq_len, step, data_type):
    ground_truth_table = pd.read_csv(file, sep=';')
    # żeby było łatwiej analizować daty, to rozbijemy je na dzień, miesiąc i rok - bo tak też odbędzie się tokenizacja naszego tekstu,
    # poza tym widzę, że raczej nie ma dat w formacie dd.mm.rrrr a raczej dd miesiąc rok
    months = {'01': 'stycznia', '02': 'lutego', '03': 'marca', '04': 'kwietnia', '05': 'maja', '06': 'czerwca',
              '07': 'lipca',
              '08': 'sierpnia', '09': 'września', '10': 'października', '11': 'listopada', '12': 'grudnia'}
    months_simple = {'01': 'styczeń', '02': 'luty', '03': 'marzec', '04': 'kwiecień', '05': 'maj', '06': 'czerwiec',
                     '07': 'lipiec',
                     '08': 'sierpień', '09': 'wrzesień', '10': 'październik', '11': 'listopad', '12': 'grudzień'}
    # print(ground_truth_table.head())
    drawing_date_day = pd.Series([], dtype=str)
    drawing_date_month_no = pd.Series([], dtype=str)
    drawing_date_month = pd.Series([], dtype=str)
    drawing_date_month_simple = pd.Series([], dtype=str)
    drawing_date_year = pd.Series([], dtype=str)
    period_from_day = pd.Series([], dtype=str)
    period_from_month_no = pd.Series([], dtype=str)
    period_from_month = pd.Series([], dtype=str)
    period_from_month_simple = pd.Series([], dtype=str)
    period_from_year = pd.Series([], dtype=str)
    period_to_day = pd.Series([], dtype=str)
    period_to_month_no = pd.Series([], dtype=str)
    period_to_month = pd.Series([], dtype=str)
    period_to_month_simple = pd.Series([], dtype=str)
    period_to_year = pd.Series([], dtype=str)
    postal_code_pre = pd.Series([], dtype=str)
    postal_code_post = pd.Series([], dtype=str)
    for i in range(len(ground_truth_table)):
        # print(ground_truth_train['drawing_date'][i])
        drawing_date_day[i] = ground_truth_table['drawing_date'][i][8:10]
        drawing_date_month_no[i] = ground_truth_table['drawing_date'][i][5:7]
        drawing_date_month[i] = months[ground_truth_table['drawing_date'][i][5:7]]
        drawing_date_month_simple[i] = months_simple[ground_truth_table['drawing_date'][i][5:7]]
        drawing_date_year[i] = ground_truth_table['drawing_date'][i][0:4]
        period_from_day[i] = ground_truth_table['period_from'][i][8:10]
        period_from_month_no[i] = ground_truth_table['period_from'][i][5:7]
        period_from_month[i] = months[ground_truth_table['period_from'][i][5:7]]
        period_from_month_simple[i] = months_simple[ground_truth_table['period_from'][i][5:7]]
        period_from_year[i] = ground_truth_table['period_from'][i][0:4]
        period_to_day[i] = ground_truth_table['period_to'][i][8:10]
        period_to_month_no[i] = ground_truth_table['period_to'][i][5:7]
        period_to_month[i] = months[ground_truth_table['period_to'][i][5:7]]
        period_to_month_simple[i] = months_simple[ground_truth_table['period_to'][i][5:7]]
        period_to_year[i] = ground_truth_table['period_to'][i][0:4]
        postal_code_pre[i] = ground_truth_table['postal_code'][i][0:2]
        postal_code_post[i] = ground_truth_table['postal_code'][i][3:6]
    ground_truth_table.insert(3, 'drawing_date_day', drawing_date_day)
    ground_truth_table.insert(4, 'drawing_date_month_no', drawing_date_month_no)
    ground_truth_table.insert(5, 'drawing_date_month', drawing_date_month)
    ground_truth_table.insert(6, 'drawing_date_month_simple', drawing_date_month_simple)
    ground_truth_table.insert(7, 'drawing_date_year', drawing_date_year)

    ground_truth_table.insert(9, 'period_from_day', period_from_day)
    ground_truth_table.insert(10, 'period_from_month_no', period_from_month_no)
    ground_truth_table.insert(11, 'period_from_month', period_from_month)
    ground_truth_table.insert(12, 'period_from_month_simple', period_from_month_simple)
    ground_truth_table.insert(13, 'period_from_year', period_from_year)

    ground_truth_table.insert(15, 'period_to_day', period_to_day)
    ground_truth_table.insert(16, 'period_to_month_no', period_to_month_no)
    ground_truth_table.insert(17, 'period_to_month', period_to_month)
    ground_truth_table.insert(18, 'period_to_month_simple', period_to_month_simple)
    ground_truth_table.insert(19, 'period_to_year', period_to_year)

    ground_truth_table.insert(21, 'postal_code_pre', postal_code_pre)
    ground_truth_table.insert(22, 'postal_code_post', postal_code_post)

    # print(tags_tokenizer.texts_to_sequences(['o']))

    seq_to_train = []
    seq_to_train_basic = []
    seq_to_return = []
    max_numer_pracownika = -1
    for i in ground_truth_table.id:
        tmp_index = ground_truth_table.index[ground_truth_table.id == i].tolist()[0]
        # print(tmp_index)
        company_name = ground_truth_table.company[tmp_index]
        company_name, company_name_in_tokens, company_name_skip_sa, company_name_skip_sa_in_tokens = company_names(company_name, t_basic)
        # print(company_name_in_tokens)

        city = ground_truth_table.city[tmp_index]
        city, city_in_tokens = city_name(city, t_basic)

        street_name_tmp = ground_truth_table.street[tmp_index]
        street_no = ground_truth_table.street_no[tmp_index]
        street, street_no, street_in_tokens, street_no_in_tokens = street_name(street_name_tmp, street_no, t_basic)

        people = ground_truth_table.people[tmp_index]
        employers, employers_in_tokens, positions, positions_in_tokens = people_finder(people, t_basic)
        if len(employers)>max_numer_pracownika:
            max_numer_pracownika = len(employers)

        print('przygotowuję dane: {}/{}'.format(tmp_index + 1, len(ground_truth_table)))
        # folder = os.path.join(folder_z_raportami, str(i))
        # for file in os.listdir(folder_z_raportami):
        # if file.endswith('.csv'):
        csv_file = str(i) + '.csv'
        file_to_read = os.path.join(folder_z_raportami, csv_file)

        full_raport = pd.read_csv(file_to_read, sep=',', usecols=['basic', 'token'], dtype='str')
        full_raport.fillna("na", inplace=True)
        # print(full_raport)
        for k in range(len(full_raport)):
            if '_' in full_raport['basic'][k]:
                full_raport['basic'][k] = full_raport['token'][k]
            if len(full_raport['basic'][k].split(" ")) != len(full_raport['token'][k].split(" ")):
                print(full_raport['basic'][k])
                print(full_raport['token'][k])
        full_raport_basic = full_raport['basic'].tolist()
        full_raport_token = full_raport['token'].tolist()
        for k in range(len(full_raport_basic)):
            tmp_basic = full_raport_basic[k]
            tmp_token = full_raport_token[k]
            if len(t_basic.texts_to_sequences([tmp_basic])[0]) != len(t_token.texts_to_sequences([tmp_token])[0]):
                full_raport_basic[k] = full_raport_token[k]
        full_raport_basic = ' '.join(full_raport_basic)
        full_raport_token = ' '.join(full_raport_token)
        full_raport_basic_in_words = t_basic.texts_to_sequences([full_raport_basic])
        full_raport_token_in_words = t_token.texts_to_sequences([full_raport_token])
        texts_from_sequence = t_token.sequences_to_texts(full_raport_token_in_words)[0].split(" ")
        sequence_to_return = ['o'] * len(texts_from_sequence)
        full_raport_in_words = full_raport_token_in_words

        numer_pracownika = -1
        for employer, position in zip(employers_in_tokens, positions_in_tokens):
            numer_pracownika += 1
            # print(employer, position)
            for j in range(len(texts_from_sequence) - len(employer) - len(position)):
                text_part = full_raport_in_words[0][j:j + len(employer) + len(position)]
                if (text_part == employer + position) and (
                        sequence_to_return[j:j + len(employer) + len(position)] == ['o'] * (
                        len(employer) + len(position))):
                    # print('znalazłem pracownika i jego zawód')
                    sequence_to_return[j] = 'human_' + str(numer_pracownika) + '_start'
                    for k in range(j + 1, j + len(employer)):
                        sequence_to_return[k] = 'human_' + str(numer_pracownika) + '_continue'
                    sequence_to_return[j + len(employer)] = 'position_' + str(numer_pracownika) + '_start'
                    for k in range(j + 1 + len(employer), j + len(employer) + len(position)):
                        sequence_to_return[k] = 'position_' + str(numer_pracownika) + '_continue'
                    # print(sequence_to_return[j:j + len(employer) + len(position)+1])

        numer_pracownika = -1
        for employer in employers_in_tokens:
            # print(employer)
            numer_pracownika += 1
            for j in range(len(texts_from_sequence)-len(employer)):
                text_part = full_raport_in_words[0][j:j + len(employer)]
                # print('*')
                # print(text_part)
                # print(employer)
                # print(sequence_to_return[j:j+len(employer)])
                # print(['o']*len(employer))
                if (text_part == employer) and (sequence_to_return[j:j+len(employer)]==['o']*len(employer)):
                    # print('Znalazlem samego pracownika')
                    sequence_to_return[j] = 'human_' + str(numer_pracownika) + '_start'
                    for k in range(j + 1, j + len(employer)):
                        sequence_to_return[k] = 'human_' + str(numer_pracownika) + '_continue'
                    # print(sequence_to_return[j:j + len(employer)+1])
        numer_pracownika = -1
        for position in positions_in_tokens:
            numer_pracownika += 1
            for j in range(len(texts_from_sequence)-len(position)):
                text_part = full_raport_in_words[0][j:j + len(position)]
                if (text_part == position) and (sequence_to_return[j:j+len(position)]==['o']*len(position)):
                    # print('Znalazłem samą pozycję')
                    sequence_to_return[j] = 'position_' + str(numer_pracownika) + '_start'
                    for k in range(j + 1, j + len(position)):
                        sequence_to_return[k] = 'position_' + str(numer_pracownika) + '_continue'
                    # print(sequence_to_return[j:j + len(position)])
        # print(sequence_to_return)
        for j in range(len(texts_from_sequence)-len(city_in_tokens[0])):
            # print(full_raport_in_words[0][j:j + len(city_in_tokens[0])])
            text_part = full_raport_in_words[0][j:j + len(city_in_tokens[0])]

            if (text_part == city_in_tokens[0]) and (len(city_in_tokens[0]) == 1):
                # print('znalazłem miasto')
                sequence_to_return[j] = 'city'
            elif text_part == city_in_tokens[0]:
                # print('znalazłem miasto wieloczłonowe')
                sequence_to_return[j] = 'city_start'
                for k in range(j+1,j+len(city_in_tokens[0])):
                    sequence_to_return[k] = 'city_continue'

        for j in range((len(texts_from_sequence) - 2)):
            # print('Szukam dat')
            word_1 = texts_from_sequence[j]
            word_2 = texts_from_sequence[j + 1]
            word_3 = texts_from_sequence[j + 2]
            # print(ground_truth_train.period_to_month_no[tmp_index])
            if ((str(word_1) == str(int(ground_truth_table.drawing_date_day.values[tmp_index])))
                or (str(word_1) == str(ground_truth_table.drawing_date_day.values[tmp_index]))) & \
                    ((str(word_2) == str(ground_truth_table.drawing_date_month.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.drawing_date_month_simple.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.drawing_date_month_no.values[tmp_index]))) & \
                    (str(word_3) == str(ground_truth_table.drawing_date_year.values[tmp_index])):
                # print('znalazłem datę - drawing date')
                sequence_to_return[j] = 'drawing_date_day'
                sequence_to_return[j + 1] = 'drawing_date_month'
                sequence_to_return[j + 2] = 'drawing_date_year'
            elif ((str(word_1) == str(int(ground_truth_table.period_from_day.values[tmp_index])))
                  or (str(word_1) == str(ground_truth_table.period_from_day.values[tmp_index]))) & \
                    ((str(word_2) == str(ground_truth_table.period_from_month.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.period_from_month_no.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.period_from_month_simple.values[tmp_index]))) & \
                    (str(word_3) == str(ground_truth_table.period_from_year.values[tmp_index])):
                # print('znalazłem datę - period from')
                sequence_to_return[j] = 'period_from_day'
                sequence_to_return[j + 1] = 'period_from_month'
                sequence_to_return[j + 2] = 'period_from_year'
            elif ((str(word_1) == str(int(ground_truth_table.period_to_day.values[tmp_index])))
                  or (str(word_1) == str(ground_truth_table.period_to_day.values[tmp_index]))) & \
                    ((str(word_2) == str(ground_truth_table.period_to_month.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.period_to_month_no.values[tmp_index]))
                     or (str(word_2) == str(ground_truth_table.period_to_month_simple.values[tmp_index]))) & \
                    (str(word_3) == str(ground_truth_table.period_to_year.values[tmp_index])):
                # print('znalazłem datę - period to')
                sequence_to_return[j] = 'period_to_day'
                sequence_to_return[j + 1] = 'period_to_month'
                sequence_to_return[j + 2] = 'period_to_year'

        for j in range(len(texts_from_sequence) - 1):
            # print('Szukam kodu pocztowego')
            word_1 = texts_from_sequence[j]
            word_2 = texts_from_sequence[j + 1]
            if (word_1 == ground_truth_table.postal_code_pre[tmp_index]) & (
                    word_2 == postal_code_post[tmp_index]):
                # print('Znalazłem kod pocztowy!')
                sequence_to_return[j] = 'postal_code_pre'
                sequence_to_return[j + 1] = 'postal_code_post'
        for j in range(len(texts_from_sequence) - len(company_name_skip_sa_in_tokens[0])):
            # print('Szukam nazwy firmy')
            text_part = full_raport_in_words[0][j:j + len(company_name_skip_sa_in_tokens[0])]
            if text_part == company_name_skip_sa_in_tokens[0]:# and sequence_to_return[j:j+len(company_name_skip_sa_in_tokens)]==['o']*len(company_name_skip_sa_in_tokens):
                # print('znalazłem nazwę firmy!')
                sequence_to_return[j] = 'company_start'
                for k in range(j + 1, j + len(company_name_skip_sa_in_tokens[0])):
                    sequence_to_return[k] = 'company_continue'
                sequence_to_return[j + len(company_name_skip_sa_in_tokens[0])] = 'company_continue'
                # print(sequence_to_return[j:j+len(company_name_skip_sa_in_tokens[0])])

        for j in range(len(texts_from_sequence) - len(street_in_tokens[0]) - len(street_no_in_tokens[0])):
            # print('Szukam adresu')
            text_part = full_raport_in_words[0][j:j + len(street_in_tokens[0]) + len(street_no_in_tokens[0])]
            if text_part == street_in_tokens[0] + street_no_in_tokens[0]:
                # print('znalazłem adres!')
                sequence_to_return[j] = 'street_start'
                # print(j, j+1, j+1+len(street_in_tokens[0]))
                # street_in_tokens[0]
                if len(street_in_tokens[0]) > 1:
                    for k in range(j + 1, j + len(street_in_tokens[0])):
                        sequence_to_return[k] = 'street_continue'
                sequence_to_return[j+len(street_in_tokens[0])] = 'street_no_start'
                if len(street_no_in_tokens) > 1:
                    for k in range(j + 1 + len(street_in_tokens[0]), j + len(street_no_in_tokens[0]) + len(street_in_tokens[0])):
                        sequence_to_return[k] = 'street_no_continue'

        # print(sequence_to_return)
        texts_from_sequence = create_sequences(full_raport_in_words, seq_len, step)
        texts_from_sequence_basic = create_sequences(full_raport_basic_in_words, seq_len, step)
        # print(sequence_to_return)
        sequence_to_return = tags_tokenizer.texts_to_sequences([sequence_to_return])
        # print(i)
        # print(len(sequence_to_return[0]))
        # print(len(full_raport_in_words[0]))
        # print(len(full_raport_basic_in_words[0]))
        if len(sequence_to_return[0]) != len(full_raport_in_words[0]) or len(sequence_to_return[0]) != len(full_raport_basic_in_words[0]):
            print('error')
            print(len(sequence_to_return[0]))
            print(len(full_raport_in_words[0]))
            print(len(full_raport_basic_in_words[0]))
            break
        sequence_to_return = create_sequences(sequence_to_return, seq_len, step)
        seq_to_return.extend(sequence_to_return)
        seq_to_train.extend(texts_from_sequence)
        seq_to_train_basic.extend(texts_from_sequence_basic)

    X_train = np.asarray(seq_to_train)
    X_train_basic = np.asarray(seq_to_train_basic)
    # print(X_train.shape)

    y_train = np.asarray(seq_to_return)
    # print(y_train.shape)

    X_file = 'X_' + data_type + '_token'
    X_file_basic = 'X_' + data_type + '_basic'
    y_file = 'y_' + data_type + '_token'
    np.save(X_file, X_train)
    np.save(X_file_basic, X_train_basic)
    np.save(y_file, y_train)

    print('Najwięcej pracowników w zbiorze {} to: {}'. format(data_type, max_numer_pracownika))

    return X_train, X_train_basic, y_train

