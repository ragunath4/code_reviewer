def pcqtgc(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@pcqtgc
def izssim(x):
    return x
