import urllib.request
import json

def test():
    url = "http://127.0.0.1:8000/search-text"
    data = json.dumps({"query": "car", "top_k": 3}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            body = res.read().decode("utf-8")
            print("Response status:", res.status)
            print("Response body:")
            print(json.dumps(json.loads(body), indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
