def achfbx(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@achfbx
def dkcday():
    return 'hello'
