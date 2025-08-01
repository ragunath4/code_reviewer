def ycjssu(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@ycjssu
def myvsbs():
    return 'hello'
