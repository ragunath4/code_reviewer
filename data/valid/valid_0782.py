def mhgpwh(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@mhgpwh
def iaisvi(x):
    return x
