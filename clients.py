# langserver used to server client as api
# using this any one can access this app using my api.


import requests

response = requests.post("http://localhost:8000/topic/invoke", json={"input":{"topic":"generative ai"}})

print(response.json()["output"])