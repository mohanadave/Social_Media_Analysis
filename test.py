import datetime

start = datetime.datetime.now()
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS,get_single_color_func
import matplotlib.pyplot as plt
import json
from wordcloud import ImageColorGenerator
from PIL import Image
import numpy as np
import collections

end = datetime.datetime.now()
print('Time taken for importing libraries:- {}'.format(end - start))
wordnet_lemmatizer = WordNetLemmatizer()
import string

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def preprocess(document):
    sentence = document.lower()
    stopwords = nltk.corpus.stopwords.words('english')
    punctuations = string.punctuation
    #     print(punctuations)
    sentence_words = nltk.word_tokenize(sentence)
    for word in sentence_words:
        if word in punctuations or word in stopwords:
            sentence_words.remove(word)
    lem_v = []
    lem_n = []
    for word in sentence_words:
        temp_v = wordnet_lemmatizer.lemmatize(word, pos='v')
        lem_v.append(temp_v)
        temp_n = wordnet_lemmatizer.lemmatize(temp_v, pos='n')
        lem_n.append(temp_n)
    return ' '.join(lem_n)


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


def load_vectors(filename):
    start = datetime.datetime.now()
    vector_loaded = pickle.load(open(filename, 'rb'))
    tfidf = TfidfVectorizer(vocabulary=vector_loaded)
    # tfidf.fit()
    # print(tfidf.vocabulary)
    end = datetime.datetime.now()
    print('Time taken for loading vectors:- {} seconds'.format(end - start))
    return tfidf


def load_model(model_file):
    start = datetime.datetime.now()
    loaded_model = pickle.load(open(model_file, 'rb'))
    end = datetime.datetime.now()
    print('Time taken for loading model:- {} seconds'.format(end - start))
    return loaded_model


# def classify(model=None,reply_text):
#     # url="http://3.16.1.236:8000/toxic"
#     # for full_text in reply_text:
#     #     payload={"comment":full_text , "source":"api"}
#     #     response=requests.post(url,data=payload)
#     #     json_response=(response.json())
#     # print(json_response['response'])
#     payload={"comment":reply_text , "source":"api"}
#     response = requests.post(url, data=payload)
#     print(response.text)
#     json_response = response.json()
#     return json_response['response']

def classify(text, model_name, vocab_name):
    vectorizer = load_vectors(vocab_name)
    model = load_model(model_name)
    start = datetime.datetime.now()
    text_vec = vectorizer.fit_transform([text])
    end = datetime.datetime.now()
    print('Time taken to create vector text:- {} seconds'.format(end - start))
    start = datetime.datetime.now()
    vectorizer.stop_words = 'english'
    op = model.predict(text_vec)
    end = datetime.datetime.now()
    print('Time taken for prediction:- {} seconds'.format(end - start))
    return str(op[0])

def merge_replies_tweets(user=None):
    all_embed = []
    user = user
    file_ptr_tweets = open('Data/Twitter/{}/tweets.json'.format(user[1:]))
    tweets_data = json.load(file_ptr_tweets)
    file_ptr_replies = open('Data/Twitter/{}/replies.json'.format(user[1:]))
    replies_data = json.load(file_ptr_replies)

    for tweet in tweets_data:
        d = {}
        tweet_id = tweet['id_str']
        replies = replies_data.get(tweet_id)
        d['tweet'] = tweet
        d['replies'] = replies
        date_list = str(d['tweet']['created_at']).split(' ')
        # print(date_list)
        date = "{} {} {} at {}".format(date_list[1], date_list[2], date_list[-1], date_list[3])
        d['tweet']['created_at'] = date
        all_embed.append(d)
    temp_file = open('Data/Twitter/{}/temp.json'.format(user[1:]),'w')
    temp_file.write(json.dumps(all_embed, indent=4))
    temp_file.close()

if __name__ == '__main__':
    import os
    # user='@narendramodi'
    # merge_replies_tweets(user)
    # path='Data/Twitter/{}'.format(user[1:])
    # fname = open(os.path.join(path,'temp.json'))
    # json_data = json.load(fname)
    # fname.close()
    # d = {}
    # fname = open(os.path.join(path,'with_reply_type.json'), 'w')
    # comment = []
    # only_response = []
    # count = 0
    # for tweet in json_data:
    #     # if count==1:
    #     #     break
    #     for reply in tweet['replies']:
    #         reply['reply_emojiless'] = deEmojify(reply['full_text'])
    #         reply['reply_type'] = classify(reply['reply_emojiless'],
    #                                        'lr_50_iter_balanced_rs_2019_07_01_16_22_04.smamodel',
    #                                        'tfidf_lemma_2019_07_01_16_22_04.smavec')
    #         only_response.append(reply['reply_type'])
    #     d_key = tweet['tweet']['id_str']
    #     d[d_key] = only_response
    #     # print(d)
    #     comment.append(tweet)
    #     count += 1
    # fname.write(json.dumps(comment, indent=4))
    #
    # fname.close()

    # # classify()
    # # load_vectors('tfidf_2019_06_30_19_49_37.smavec')
    # test_message = 'Hello World'
    # classify(test_message, 'lr_50_iter_balanced_rs_2019_07_01_16_22_04.smamodel',
    #          'tfidf_lemma_2019_07_01_16_22_04.smavec')
    #
    # loaded_model = pickle.load(open('lr_50_iter_balanced_rs_2019_07_01_16_22_04.smamodel', 'rb'))
    # vector_loaded = pickle.load(open('tfidf_lemma_2019_07_01_16_22_04.smavec', 'rb'))
    # tfidf = TfidfVectorizer(vocabulary=vector_loaded)
    # text_vec = tfidf.fit_transform(
    #     [preprocess("Don't discuss just take action and we need to first all mother fucker gaddar who live in India")])
    # print(text_vec.shape)
    # tfidf.stop_words = 'english'
    # # print(vectorizer.stop_words)
    # print(tfidf.inverse_transform(text_vec))
    # op = loaded_model.predict(text_vec)
    # print(op)
    stopwords = set(STOPWORDS)


    def show_wordcloud(tweet_freq,data, title=None):
        mask = np.array(Image.open("img/twitter_mask.png"))
        for tweet in tweet_freq:
            freq=dict(tweet_freq[tweet])
            wordcloud = WordCloud(
                background_color='white',
                contour_color='steelblue',
                contour_width=3,
                stopwords=stopwords,
                max_words=1000,
                max_font_size=60,
                scale=6,
                mask=mask,
                random_state=1  # chosen at random by flipping a coin; it was heads
            ).generate_from_frequencies(freq)
            # fig = plt.figure(figsize=[7, 7])
            # plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
            # plt.axis("off")
            # print(data[tweet]['0'])

            # for tweet in data:
            print(tweet)
            class_dis={
                '0':[],
                '50':[],
                '100':[]
            }
            classes=list(data[tweet].keys())
            for clas in classes:
                print('class',clas)
                    # replies[x] = preprocess(replies[x])
                replies=data[tweet][clas]
                for x in range(len(replies)):
                    replies[x] = preprocess(replies[x])
                whole_doc = ' '.join(replies)
                all_words = whole_doc.split(' ')
                all_words_set = list(set(all_words))
                temp = []
                for word in all_words_set:
                    temp_w = ''
                    for char in word:
                        if char.isalpha():
                            temp_w += char
                    temp.append(temp_w.lower())
                all_words_set = temp
                # all_words_set=[word.strip('\n').lower().replace('#','').replace('@','').replace('\n',' ') for word in all_words_set]
                class_dis[clas]=all_words_set





            color_to_words = {
                # words below will be colored with a green single color function
                '#00ff00': class_dis['0'],
                # will be colored with a red single color function
                'yellow': class_dis['50'],
                'red': class_dis['100'],
            }
            print(json.dumps(color_to_words,indent=2))
            default_color = 'grey'
            grouped_color_func = GroupedColorFunc(color_to_words, default_color)
            wordcloud.recolor(color_func=grouped_color_func)
            fig = plt.figure(1, figsize=(12, 12))
            plt.axis('off')
            if title:
                fig.suptitle(title, fontsize=20)
                fig.subplots_adjust(top=2.3)

            plt.imshow(wordcloud)
            plt.show()
            wordcloud.to_file('/Users/daksh/Desktop/social-media-analysis/project/static/wordclouds/narendramodi/{}.jpg'.format(tweet))


    def get_all_replies(file):
        file_ptr = open(file)
        replies = {}
        data = json.load(file_ptr)
        accept = ["100","50","0"]
        for tweet in data[:3]:
            id_str=tweet['tweet']['id_str']
            replies[id_str]={
                '0':[],
                '50':[],
                '100':[],
            }
            for reply in tweet['replies']:
                if reply['reply_type'] in accept:
                    text = reply['reply_emojiless']
                    if reply['reply_type']=='0':
                        replies[id_str]['0'].append(text)
                    if reply['reply_type']=='50':
                        replies[id_str]['50'].append(text)
                    if reply['reply_type']=='100':
                        replies[id_str]['100'].append(text)
        print(len(replies))
        return replies


    def create_word_frequency(data):
        tweet_freq={}
        for tweet in data:
            replies=data[tweet]['0']+data[tweet]['50']+data[tweet]['100']
            for x in range(len(replies)):
                replies[x] = preprocess(replies[x])

            whole_doc = ' '.join(replies)
            all_words = whole_doc.split(' ')
            all_words_set = list(set(all_words))
            temp=[]
            for word in all_words_set:
                temp_w=''
                for char in word:
                    if char.isalpha():
                        temp_w+=char
                temp.append(temp_w.lower())
            all_words_set=temp
            # all_words_set=[word.strip('\n').lower().replace('#','').replace('@','').replace('\n',' ') for word in all_words_set]
            freq = {}
            # print(STOPWORDS)
            for word in all_words_set:
                STOPWORDS.update([''])
                if word not in STOPWORDS:
                    freq[word] = all_words.count(word)
        # freq=collections.OrderedDict(freq)
            sorted_x = sorted(freq.items(), key=lambda kv: kv[1])
            tweet_freq[tweet]=sorted_x
            # print(sorted_x[-1:30:-1])
        return tweet_freq

    data = get_all_replies(
        '/Users/daksh/Desktop/social-media-analysis/Data/Twitter/narendramodi/with_reply_type.json')
    tweet_freq = create_word_frequency(data)
    # print(tweet_freq.keys())
    show_wordcloud(tweet_freq,data)
