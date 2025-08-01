def dzfbdf(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@dzfbdf
def exzprm(x):
    return x
