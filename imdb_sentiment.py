import nltk
from nltk.corpus import stopwords
import pickle

stop = stopwords.words('english')

def get_word_features(reviews):
        all_words = []
        for (words, sentiment) in reviews:
          all_words.extend(words)
        wordlist = nltk.FreqDist(all_words)
        word_features = []
        for feature in wordlist:
            if wordlist[feature] > 10000:
                word_features.append(feature)
        print(len(word_features))
        return word_features

def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features



train_pos_reviews=[]
train_neg_reviews=[]
test_pos_reviews=[]
test_neg_reviews=[]


with open('./imdb/train-pos.txt','r') as f:
    for line in f:
        train_pos_reviews.append((line,'pos'))
f.close()

with open('./imdb/train-neg.txt','r') as f:
    for line in f:
        train_neg_reviews.append((line,'neg'))
f.close()

with open('./imdb/test-pos.txt','r') as f:
    for line in f:
        test_pos_reviews.append((line,'pos'))
f.close()

with open('./imdb/test-neg.txt','r') as f:
    for line in f:
        test_pos_reviews.append((line,'neg'))
f.close()

train_reviews = []
test_reviews = []

for (words, sentiment) in train_pos_reviews + train_neg_reviews:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    train_reviews.append((words_filtered, sentiment))

for (words, sentiment) in test_pos_reviews + test_neg_reviews:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    test_reviews.append((words_filtered, sentiment))

print("Reviews filtered")

word_features = get_word_features(train_reviews+test_reviews)
print("Word Features Generated")

training_set = nltk.classify.apply_features(extract_features, train_reviews)
test_set = nltk.classify.apply_features(extract_features, test_reviews)
print("Training and Test Sets Created")

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Model generated")

f = open('imdb_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

# f = open('imdb_classifier.pickle', 'rb')
# classifier = pickle.load(f)
# f.close()


print(nltk.classify.accuracy(classifier,test_set))
