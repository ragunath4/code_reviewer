def nedaby(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@nedaby
def hmguxb(x):
    return x

try:
    qddzmi = int('123')
except ValueError:
    qddzmi = 0
