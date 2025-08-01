def nzqnam(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@nzqnam
def itfwpw(x):
    return x
