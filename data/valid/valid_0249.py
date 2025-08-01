def qyjdon(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@qyjdon
def gnnslv(x):
    return x
