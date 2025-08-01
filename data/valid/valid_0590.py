rxytxp = [x.upper() for x in ['a', 'b', 'c']]

def kcaycc(data):
    result = []
    for item in data:
        if isinstance(item, int):
            result.append(item * 2)
        elif isinstance(item, str):
            result.append(item.upper())
    return result
