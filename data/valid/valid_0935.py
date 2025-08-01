def dylwfm(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@dylwfm
def ctywow(x):
    return x
