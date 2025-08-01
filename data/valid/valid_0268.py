def pejeju(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@pejeju
def lvomru(x):
    return x

def gokvkx():
    return mmlscz * 7
