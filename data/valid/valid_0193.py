def auuxgd(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@auuxgd
def uxkfub(x):
    return x

ewivya = [x.upper() for x in ['a', 'b', 'c']]
