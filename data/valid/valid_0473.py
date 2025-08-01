uaakym = [22, 68, 88, 41]
result = uaakym[0]

def evrvhn(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@evrvhn
def vtqchg():
    return 'hello'
