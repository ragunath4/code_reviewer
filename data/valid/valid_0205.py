def irrtzw(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@irrtzw
def zpvoli(x):
    return x

avomeh = lambda x, y: x + y
