import re
import sys
from utils import write_status
import pandas as pd

our_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'his', 'him' 'himself',
                   'she', 'her', 'hers', 'herself',  'it', 'its', 'itself', 'they', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                   'this', 'that', 'those', 'these', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'until', 'while', 'of', 'at', 'by', 'during', 'before', 'after', 'now',
                    'will', 'other', 'each', 'both', 'any', 'all', 'how', 'why', 'where', 'when', 'then', 'here', 'once', 'again', 'further']

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

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{6+}', '######', x)
        x = re.sub('[0-9]{5}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        x = re.sub('[0-9]{1}', '#', x)
    return x

def is_valid_word(word):
    # Check if word begins with an alphabet
    if ((re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)):
        # if (word in stopwords.words('english')):
        #     return False
        if ("xx" in word or word=="x" or word=="xo"):
            return False
        else:
            return True
    else:
        return False

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?(\)\s?)+|:-(\)\s?)+|(\(\s?)+:|(\(\s?)+-:|:\'+(\)\s?)+|=(\)\s?)+)', ' <happyfaceone> ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D, xd
    tweet = re.sub(r'(:\s?(d\s?)+\s|:-(d\s?)+\s|x(\s?-?\s?)+(d\s?)+)\s', ' <happyfacetwo> ', tweet)

    tweet = re.sub(r'(:(\s?p)+\s|:-(p\s?)+\s)', ' <tongueface> ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<(3\s?)+\s?|:(\*\s?)+\s)', ' <heart> ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;(\s?-?\s?)+(\)\s?)+|;(\s?-?\s?)+(d\s?)+\s|(\(\s?)+(\s?-?\s?)+;)', ' <wink> ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?(\(\s?)+|:-(\(\s?)+|(\)\s?)+:|(\)\s?)+-:|:\s?(\\\s?)+|:\s?(/\s?)+)', ' <sadface> ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,(\(\s?)+|:\'(\(\s?)+|:"(\(\s?)+)', ' <cryface> ', tweet)
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


def preprocess_tweet(tweet, slang_dict):
    processed_tweet = []
    # Convert to lower case OK
    tweet = tweet.lower()
    # Replaces URLs with the word URL OK
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' <url> ', tweet)
    # Replace @handle with the word USER_MENTION OK
    tweet = re.sub(r'@[\S]+', ' <user> ', tweet)
    # Replaces #hashtag with hashtag OK
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet) OK
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space OK
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet OK
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)                         # WATCH OUT

    tweet = removeUnicode(tweet)

    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    words = tweet.split()
    for word in words:
        if word in contraction_dict:
            correct_word = contraction_dict[word]
            processed_tweet.append(correct_word)
        else:
            word = clean_numbers(word)
            word = preprocess_word(word)
            processed_tweet.append(word)

    tweet = ' '.join(processed_tweet)

    return tweet

def preprocess_csv(csv_file_name, processed_file_name, test_file):

    """ Creates a dictionary with slangs and their equivalents and replaces them """
    with open('./infrastructure/slang.txt') as file:
         slang_dict = dict(map(str.strip, line.partition('\t')[::2]) for line in file if line.strip())

    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r', encoding='utf-8') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id = line[:line.find(',')]
                line = line[1 + line.find(','):-1]
            processed_tweet = preprocess_tweet(line, slang_dict)
            if (test_file == 0):
                save_to_file.write('%s\n' %
                                   (processed_tweet))
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
    csv_file_name = sys.argv[1]
    processed_file_name = sys.argv[1][:-4] + '_preprocessed_1.txt'
    preprocess_csv(csv_file_name, processed_file_name, int(sys.argv[2]))
