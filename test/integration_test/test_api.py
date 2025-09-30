import unittest
import requests

# Change this to your real server URL (local or deployed)
BASE_URL = "http://localhost:80"


class TestDocumentEmbeddingAPI(unittest.TestCase):

    def test_health_check(self):
        """Test /health-check endpoint"""
        url = f"{BASE_URL}/health-check"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, f"Health check failed: {response.text}")
        data = response.json()
        print("Health Check Response:", data)
        self.assertEqual(data["status"], "ok")

    def test_eval_success(self):
        """Test /eval endpoint with valid metrics"""
        url = f"{BASE_URL}/eval"
        payload = {
            "metrics": ['f1'],
            "questions": ["Dung lượng RAM của nokia 3210 4g là bao nhiêu?"],
            "ground_truths": ["Dung lượng RAM của nokia 3210 4g là 64MB. và 32MB"],
            "response_llms": ["Dung lượng RAM của nokia là 64MB"],
            "contexts": ["Product Title: nokia 3210 4g - chính hãng Product Specifications: Công nghệ màn hình: IPS<br> Kích thước màn hình: 2.4 inch<br> Độ phân giải: 2MP<br> Hệ điều hành: S30+<br> Bộ nhớ trong: 128MB3<br> RAM: 64MB<br> Mạng di động: 2G, 3G, 4G, Hỗ trợ VoLTE2<br> Số khe SIM: Hai SIM Nano SIM + Nano SIM<br> Dung lượng pin: 1450mAh<br> Promotions: Price: 1,590,000 ₫ Colors: Màu Vàng, Xanh, Màu Đen"]
        }

        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200, f"Eval failed: {response.text}")
        data = response.json()
        print("Eval Response:", data)
        self.assertIn("results", data)

    def test_eval_invalid_metric(self):
        """Test /eval with invalid metric"""
        url = f"{BASE_URL}/eval"
        payload = {
            "metrics": ["invalid_metric"],
            "questions": ["What is AI?"],
            "response_llms": ["AI is artificial intelligence"],
            "ground_truths": ["Artificial intelligence is the simulation of human intelligence processes by machines"],
            "contexts": ["General AI context"]
        }

        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 400, "Invalid metric should return 400")
        data = response.json()
        print("Invalid Metric Response:", data)
        self.assertIn("detail", data)

    def test_eval_empty_input(self):
        """Test /eval with completely empty input"""
        url = f"{BASE_URL}/eval"
        payload = {}
        response = requests.post(url, json=payload)

        # FastAPI + Pydantic usually returns 422 for missing required fields
        self.assertEqual(response.status_code, 422, "Empty input should return 422")
        data = response.json()
        print("Empty Input Response:", data)
        self.assertIn("detail", data)
    
    def test_eval_mismatched_questions_responses(self):
        """Test /eval when questions and response_llms lengths are mismatched"""
        url = f"{BASE_URL}/eval"
        payload = {
            "metrics": ["accuracy"],
            "questions": ["What is AI?", "Define ML"],  # 2 questions
            "response_llms": ["AI is artificial intelligence"],  # only 1 response
            "ground_truths": ["Artificial intelligence ...", "Machine learning ..."],
            "contexts": ["AI context", "ML context"]
        }

        response = requests.post(url, json=payload)

        # Depending on your evaluator, this might be 400 or 500
        self.assertIn(response.status_code, [400, 500], "Mismatch should return 400 or 500")
        data = response.json()
        print("Mismatched Q/A Response:", data)
        self.assertIn("detail", data)

if __name__ == "__main__":
    unittest.main(verbosity=2)
