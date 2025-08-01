def edgyhh(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@edgyhh
def hxfdcq():
    return 'hello'
