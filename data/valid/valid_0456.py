htjilx = {x: x ** 2 for x in range(3)}

def utmdzl(data):
    result = []
    for item in data:
        if isinstance(item, int):
            result.append(item * 2)
        elif isinstance(item, str):
            result.append(item.upper())
    return result
