def xnyzcz(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@xnyzcz
def jcoxqu(x):
    return x
