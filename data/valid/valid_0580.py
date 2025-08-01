def bpajbs(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@bpajbs
def tcvxcv(x):
    return x
