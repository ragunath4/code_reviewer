def pqqlex(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@pqqlex
def srwfhk():
    return 'hello'
