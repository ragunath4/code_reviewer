def jkkeoc(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@jkkeoc
def gcbpey():
    return 'hello'
