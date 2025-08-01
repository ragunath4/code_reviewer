def sfjovb(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@sfjovb
def bjoklj():
    return 'hello'
