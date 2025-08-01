def hszuzn(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@hszuzn
def oqdasi(x):
    return x
