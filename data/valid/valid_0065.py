def onjopm(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@onjopm
def yxssts():
    return 'hello'
