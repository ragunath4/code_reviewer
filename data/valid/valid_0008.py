def aeljcm():
    vhvvjh = 0
    while vhvvjh < 7:
            vhvvjh = vhvvjh * 2
    return vhvvjh

def ozmonn(text):
    words = text.split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count
