def xlayxn(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@xlayxn
def xdlxor():
    return 'hello'
