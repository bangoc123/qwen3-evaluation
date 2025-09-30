import requests

BASE_URL = "http://localhost:80"  # Change to your server address if deployed

# Example payload for /eval
payload = {
"metrics": ['f1'],
"questions": ["Dung lượng RAM của nokia 3210 4g là bao nhiêu?"],
"ground_truths": ["Dung lượng RAM của nokia 3210 4g là 64MB. và 32MB"],
"response_llms": ["Dung lượng RAM của nokia là 64MB"],
"contexts": ["Product Title: nokia 3210 4g - chính hãng Product Specifications: Công nghệ màn hình: IPS<br> Kích thước màn hình: 2.4 inch<br> Độ phân giải: 2MP<br> Hệ điều hành: S30+<br> Bộ nhớ trong: 128MB3<br> RAM: 64MB<br> Mạng di động: 2G, 3G, 4G, Hỗ trợ VoLTE2<br> Số khe SIM: Hai SIM Nano SIM + Nano SIM<br> Dung lượng pin: 1450mAh<br> Promotions: Price: 1,590,000 ₫ Colors: Màu Vàng, Xanh, Màu Đen"]

}

# Call /eval endpoint
response = requests.post(f"{BASE_URL}/eval", json=payload)

if response.status_code == 200:
    print("Eval Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

# Call /health-check endpoint
health = requests.get(f"{BASE_URL}/health-check")
print("Health Check:", health.json())