from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import csv

texts, labels = [], []
def prep_data():
    with open('amazon_review_full_csv/train.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            newrow = row[1] + ' ' + row[2]
            labels.append(row[0])
            texts.append(newrow)

    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    return train_x, valid_x, train_y, valid_y,trainDF

#a,b,c,d,e = prep_data()

def countVec(a,b,e):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(e)

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(a)
    xvalid_count =  count_vect.transform(b)
    return xtrain_count, xvalid_count


def tf_idf(a,b,e):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(e['text'])
    xtrain_tfidf =  tfidf_vect.transform(a)
    xvalid_tfidf =  tfidf_vect.transform(b)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(e['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(a)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(b)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(e['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(a)
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(b)

    return xtrain_tfidf, xvalid_tfidf, xtrain_tfidf_ngram, xvalid_tfidf_ngram, xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

def word_emb(df,tx,vx):
    embeddings_index = {}
    for i, line in enumerate(open('wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # create a tokenizer
    token = text.Tokenizer()
    token.fit_on_texts(df)
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(tx), maxlen=70)
    print(train_seq_x)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(vx), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
#countVec(a,b,c,d,e)

def apply_pos(DF):
    trainDF['char_count'] = DF['text'].apply(len)
    trainDF['word_count'] = DF['text'].apply(lambda x: len(x.split()))
    trainDF['word_density'] = DF['char_count'] / (trainDF['word_count']+1)
    trainDF['punctuation_count'] = DF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = DF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = DF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    pos_family = {
        'noun' : ['NN','NNS','NNP','NNPS'],
        'pron' : ['PRP','PRP$','WP','WP$'],
        'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
        'adj' :  ['JJ','JJR','JJS'],
        'adv' : ['RB','RBR','RBS','WRB']
    }


    def check_pos_tag(x, flag):
        cnt = 0
        try:
            wiki = textblob.TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    DF['noun_count'] = DF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
    DF['verb_count'] = DF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
    DF['adj_count'] = DF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
    DF['adv_count'] = DF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
    DF['pron_count'] = DF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

    return DF

#word_emb(e,a,b)
