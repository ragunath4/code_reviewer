hosesh = {x: len(x) for x in ['a', 'bb', 'ccc']}

def tnvcpn(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper

@tnvcpn
def pgwbnc(x):
    return x
