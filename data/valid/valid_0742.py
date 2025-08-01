def bcyiyj(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@bcyiyj
def lzwwqq(x):
    return x
