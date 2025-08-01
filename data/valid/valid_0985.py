def uowptz(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@uowptz
def ziopoq(x):
    return x

try:
    arhkhc = [1, 2, 3][10]
except IndexError:
    arhkhc = 0
