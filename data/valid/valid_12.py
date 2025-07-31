
def process_data(data_list):
    results = []
    for item in data_list:
        if isinstance(item, dict):
            processed = {k: v * 2 for k, v in item.items() if isinstance(v, (int, float))}
            results.append(processed)
        elif isinstance(item, list):
            processed = [x * 2 for x in item if isinstance(x, (int, float))]
            results.append(processed)
    return results
        