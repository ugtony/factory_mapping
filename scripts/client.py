import requests

url = "http://localhost:8000/localize"
files = {'file': open('data/brazil360/query/frames-000021_F.jpg', 'rb')}
data = {'fov': 100} # 選擇性參數

resp = requests.post(url, files=files, data=data)
print(resp.json())