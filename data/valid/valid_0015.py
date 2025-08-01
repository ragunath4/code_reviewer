try:
    oyvodo = {'a': 1}['b']
except KeyError:
    oyvodo = 0

def glyoew(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@glyoew
def hezbzm(x):
    return x
