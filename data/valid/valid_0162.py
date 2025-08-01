def mqnybo(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@mqnybo
def vapakj():
    return 'hello'
