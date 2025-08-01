with open('dfdfeu.txt', 'r') as ovsahv:
    content = ovsahv.read()

def fmdziw(text):
    words = text.split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count
