def xunbde(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@xunbde
def yrejyv(x):
    return x
