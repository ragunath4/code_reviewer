def lmtmkb(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@lmtmkb
def yngzpu():
    return 'hello'
