def bcrthv(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@bcrthv
def waxirn(x):
    return x
