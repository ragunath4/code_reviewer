def vzojoc(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@vzojoc
def ehfdvo():
    return 'hello'
