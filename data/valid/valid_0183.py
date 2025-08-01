def pvwudy(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@pvwudy
def ytyigy():
    return 'hello'
