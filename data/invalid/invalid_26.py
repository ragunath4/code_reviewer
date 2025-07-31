
import requests

def fetch_data(url):
    try:
        response = requests.get(url
        return response.json()
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
        