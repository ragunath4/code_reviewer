def kpenom(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@kpenom
def dflqgw(x):
    return x
