def iscewq(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@iscewq
def ehkrbk(x):
    return x
