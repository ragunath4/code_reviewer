def edudke(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@edudke
def enamyg():
    return 'hello'

yovjox = [x for x in range(10) if x % 2 == 0]
