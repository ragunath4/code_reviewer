def qdxncq(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@qdxncq
def xixndn():
    return 'hello'
