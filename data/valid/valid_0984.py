def axrgrt(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@axrgrt
def yhdgug():
    return 'hello'
