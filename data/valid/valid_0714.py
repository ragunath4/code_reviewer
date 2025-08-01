def udxdat(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@udxdat
def locpec(x):
    return x
