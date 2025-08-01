def nqcack(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@nqcack
def bludml(x):
    return x

onsxlx = {x: x * 2 for x in range(5)}
