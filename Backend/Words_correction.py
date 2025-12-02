from textblob import TextBlob

def correct_word(word):
    return str(TextBlob(word).correct())

def correct_sentence(sentence):
    return str(TextBlob(sentence).correct())
