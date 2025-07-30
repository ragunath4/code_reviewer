
def decorator_with_error(func):
    def wrapper(*args, **kwargs
        print('Before')
        result = func(*args, **kwargs)
        print('After')
        return result
    return wrapper
        