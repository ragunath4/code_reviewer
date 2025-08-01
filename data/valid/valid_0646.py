def jxmagu(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@jxmagu
def lpwprw():
    return 'hello'

try:
    iegufs = [1, 2, 3][10]
except IndexError:
    iegufs = 0
