
def process_json_data(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.upper()
            elif isinstance(value, (int, float)):
                data[key] = value * 2
    return data
        