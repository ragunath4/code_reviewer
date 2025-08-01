gyytbl = [5, 78, 98, 3, 54, 5, 87, 70]

def adpmht(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@adpmht
def icjgqt(x):
    return x
