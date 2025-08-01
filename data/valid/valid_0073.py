vhwtdu = [x.upper() for x in ['a', 'b', 'c']]

def snpvoe(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@snpvoe
def muumym(x):
    return x
