def tcqkuo(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@tcqkuo
def jisjtp(x):
    return x
