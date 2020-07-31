import re
import sys
from utils import write_status
from nltk.stem.porter import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords

our_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'his', 'him', 'himself',
                   'she', 'her', 'hers', 'herself',  'it', 'its', 'itself', 'they', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                   'this', 'that', 'those', 'these', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'until', 'while', 'of', 'at', 'by', 'during', 'before', 'after', 'now',
                    'will', 'other', 'each', 'both', 'any', 'all', 'how', 'why', 'where', 'when', 'then', 'here', 'once', 'again', 'further']

def preprocess_word(word):
    # Remove minimum punctuation
    word = word.strip('\'",')
    return word


contraction_dict = {"aint": "is not", "arent": "are not","cant": "cannot", "'cause": "because", "couldve": "could have", "couldnt": "could not",
       "didnt": "did not",  "doesnt": "does not", "dont": "do not", "hadnt": "had not", "hasnt": "has not", "havent": "have not", "hed": "he would","he'll": "he will", "hes": "he is", "howd": "how did", "howdy": "how do you", "howll": "how will",
       "hows": "how is",  "idve": "i would have", "illve": "i will have","im": "i am", "its": "it is", "ive": "i have", "idve": "i would have", "illve": "i will have", "im": "i am", "ive": "i have", "isnt": "is not",
       "itd": "it would", "itdve": "it would have", "itll": "it will", "itllve": "it will have","lets": "let us", "maam": "madam", "maynt": "may not", "mightve": "might have","mightnt": "might not","mightntve": "might not have",
       "mustve": "must have", "mustnt": "must not", "mustntve": "must not have", "neednt": "need not", "needntve": "need not have",
       "shantve": "shall not have", "shed": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
       "sos": "so as", "thiss": "this is","thatd": "that would", "thatdve": "that would have", "thats": "that is", "thered": "there would", "theredve": "there would have", "theres": "there is", "heres": "here is","theyd": "they would", "theydve": "they would have",
       "theyll": "they will", "theyllve": "they will have", "theyre": "they are", "theyve": "they have", "tove": "to have", "wasnt": "was not", "wed": "we would", "wedve": "we would have", "well": "we will", "wellve": "we will have", "were": "we are", "weve": "we have",
       "werent": "were not", "whatll": "what will", "whatllve": "what will have", "whatre": "what are",  "whats": "what is", "whatve": "what have", "whens": "when is", "whenve": "when have", "whered": "where did", "wheres": "where is", "whereve": "where have", "wholl": "who will",
       "whollve": "who will have", "whos": "who is", "whove": "who have", "whys": "why is", "whyve": "why have", "willve": "will have", "wont": "will not", "wontve": "will not have", "wouldve": "would have", "wouldnt": "would not", "wouldntve": "would not have", "yall": "you all",
       "yalld": "you all would","yalldve": "you all would have","yallre": "you all are","yallve": "you all have","youd": "you would", "youdve": "you would have", "youll": "you will", "youllve": "you will have", "youre": "you are", "youve": "you have",
       "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
       "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
       "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will",
       "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def _get_contractions(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re


def replace_contractions(text):
    contractions, contractions_re = _get_contractions(contraction_dict)
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


def split_hashtag_to_words_all_possibilities(hashtag):
    all_possibilities = []

    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]
    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

    for split_pos in possible_split_positions[:1]:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)

            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2)

            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]

    return all_possibilities


def is_valid_word(word):
    # Check if word begins with an alphabet
    if ((re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)):
        if ("xx" in word or word=="x" or word=="xo"):
            return False
        else:
            return True
    else:
        return False


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D, xd
    tweet = re.sub(r'(:\s?D|:-D|x-?D|x-?d|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def correct_spell(tweet):
    """
    Function that uses the three dictionaries that we described above and replace noisy words
    Arguments: tweet (the tweet)
    """
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dico.keys():
            tweet[i] = dico[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

######### remove unicodes #########
def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    text = re.sub(r'x([00-96])+',r'',text)
    return text

######### #########

def replaceSlang(tweet, slang_dict):
    processed_tweet = []
    for word in tweet.split():
        if word in slang_dict:
            word = slang_dict[word]
        processed_tweet.append(word)

    tweet = ' '.join(processed_tweet)

    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    tweet = tweet.strip(' "\'')

    words = tweet.split()
    for word in words:
        word = preprocess_word(word)
        new_word = ''.join([i for i in word if not i==","])
        word=new_word
        processed_tweet.append(word)
    tweet = ' '.join(processed_tweet)

    return tweet

def preprocess_csv(csv_file_name, processed_file_name, test_file):

    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):-1]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):-1]
            tweet = line
            processed_tweet = preprocess_tweet(tweet)
            if (test_file==0):
                save_to_file.write('%s,%d,%s\n' %
                                    (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                    (tweet_id, processed_tweet))
            write_status(i + 1, total)
    save_to_file.close()
    print ('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name


def find_max_length(file):
    data = pd.read_csv(file, header=None,  index_col=0)
    #data.columns=["Label", "Sentence"]
    data.columns=["Sentence"]
    data = data.dropna()
    print(data.shape)

    lengths=[]
    tweets = data["Sentence"].tolist()
    for tweet in tweets:
        l = tweet.split()
        lengths.append(len(l))

    print(len(lengths))
    print("max length: ", max(lengths))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('Usage: python preprocess.py <raw-CSV> <test>')
        exit()
    use_stemmer = False
    csv_file_name = sys.argv[1]
    processed_file_name = sys.argv[1][:-4] + '-not-processed.csv'
    if use_stemmer:
        porter_stemmer = PorterStemmer()
        processed_file_name = sys.argv[1][:-4] + '-processed-stemmed.csv'
    preprocess_csv(csv_file_name, processed_file_name, int(sys.argv[2]))
