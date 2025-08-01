def tousln(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@tousln
def gqnfpe():
    return 'hello'
