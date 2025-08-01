try:
    wbbgfk = {'a': 1}['b']
except KeyError:
    wbbgfk = 0

def nohfdb(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@nohfdb
def jvalub(x):
    return x
