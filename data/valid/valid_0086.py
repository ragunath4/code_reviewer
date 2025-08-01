def zhicir(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@zhicir
def toarpm(x):
    return x
