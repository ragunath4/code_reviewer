def erqdae(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@erqdae
def vicgoa():
    return 'hello'
