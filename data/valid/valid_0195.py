def vjcaak(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@vjcaak
def iusfkn():
    return 'hello'
