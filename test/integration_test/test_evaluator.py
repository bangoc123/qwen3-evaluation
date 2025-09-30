import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Make sure Python can find evaluator.py in the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluator import Evaluator
from constants import PRECISION, RECALL, F1, GROUNDEDNESS, NOISE_SENSITIVITY


class LoggingTestResult(unittest.TextTestResult):
    """Custom TestResult to log pass/fail of each test."""

    def addSuccess(self, test):
        super().addSuccess(test)
        logger.info(f"✅ PASS: {test}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        logger.error(f"❌ FAIL: {test}")

    def addError(self, test, err):
        super().addError(test, err)
        logger.error(f"💥 ERROR: {test}")


class LoggingTestRunner(unittest.TextTestRunner):
    """Custom runner using our LoggingTestResult."""

    resultclass = LoggingTestResult


class TestEvaluatorIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evaluator = Evaluator(
            metrics=[],
            model_type="online",
            model_name="gemini",
            model_version="gemini-2.5-pro"
        )

    def test_eval_precision_recall_f1(self):
        question = "Điện thoại di động Mobell M539 4G có những màu nào?"
        response_llm = "Dưới đây là danh sách màu sắc cho các sản phẩm điện thoại di động Mobell: - Mobell M539 4G: vàng - Mobell M139 4G: đỏ, đen, xanh - Mobell F309 4G: vàng, đen, đỏ, xanh Các sản phẩm này đều chính hãng và có thể giảm giá tối đa 500,000 ₫ khi mở thẻ tín dụng TPBank EVO."
        ground_truth = "Điện thoại di động Mobell M539 4G có màu Gold và Red."


        self.evaluator = Evaluator(
            metrics=[PRECISION, RECALL, F1],
            model_type="online",
            model_name="gemini",
            model_version="gemini-2.5-pro"
        )
        results = self.evaluator.eval(
            questions=[question],
            response_llms=[response_llm],
            ground_truths=[ground_truth],
        )

        result = results[0]
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertEqual(result["precision"], 0.2)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.29)

    def test_eval_groundedness(self):
        self.evaluator = Evaluator(
            metrics=[GROUNDEDNESS],
            model_type="online",
            model_name="gemini",
            model_version="gemini-2.5-pro"
        )

        question = "Màu sắc có sẵn của oscal tiger 12 là gì?"
        response_llm = "Màu sắc có sẵn của oscal tiger 12 là Xanh Thiên Thanh, Xám Vân Vũ, Tím Bồng Bềnh."
        context = "Reference id 7a4f053b-177e-4ce4-b5b8-f2ccd457c9f5: Product Title: oscal tiger 12 (12+12gb/256gb) - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nIPS LCD 120Hz<br> Độ phân giải:\n1080x2460, Camera chính 64MP Samsung ISOCELLGW3, Camera phụ 2MP, Camera trước 13MP Samsung ISOCELL3L6<br> Kích thước màn hình:\n6.78 inch<br> Hệ điều hành:\nDokeOS 4.0 trên nền tảng Android 13<br> Vi xử lý:\nMediaTek Helio G99<br> Bộ nhớ trong:\n256GB<br> RAM:\n12GB<br> Mạng di động:\n2G, 3G, 4G<br> Số khe SIM:\n2 nano SIM<br> Dung lượng pin:\n5000 mAh<br>\n\nPromotions:\n- KM 1<br>- Ưu đãi Trả góp 0% - Trả trước 0đ.<br>- KM 2<br>- 1 đổi 1 trong 100 ngày đầu nếu có lỗi do nhà sản xuất.<br>- KM 3<br>- Ưu đãi dành riêng cho khách hàng sở hữu OSCAL - Tặng ngay SIM MobiFone/VNSKY dung lượng truy cập cao<br>\n\nPrice:\n3,990,000 ₫\n\nColors:\nXanh Thiên Thanh, Xám Vân Vũ, Tím Bồng Bềnh\n\nReference id 3e8feca3-27ed-44d6-ad12-d7c71fc62430: Product Title: điện thoại tcl 408 (4gb/64gb) - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nIPS LCD<br> Độ phân giải:\nHD+ (720 x 1612), Camera chính 50MP f/1.8, Camera chiều sâu 2MP f/2.4, 8MP f/2.0<br> Kích thước màn hình:\n6.6 inch<br> Hệ điều hành:\nAndroid 12<br> Vi xử lý:\nMediaTek MT6762<br> Bộ nhớ trong:\n64GB<br> RAM:\n4GB<br> Mạng di động:\n2G, 3G, 4G<br> Số khe SIM:\n2 nano SIM<br> Dung lượng pin:\n5000 mAh<br>\n\nPromotions:\n- Giảm 5% không giới hạn khuyến mãi qua Homepaylater<br>- Giảm 50% tối đa 700k khi mở thẻ tín dụng Vpbank trên SenID<br>- Giảm 20% tối đa 500k khi mở thẻ tín dụng TPBank EVO<br>- Giảm 1% tối đa 100.000đ khi thanh toán qua Zalopay<br>\n\nPrice:\n2,090,000 ₫\n\nColors:\nXanh Dương, Màu Xám\n\nReference id b214023d-e91f-472f-8b8a-2a541a879d67: Product Title: điện thoại tcl 40 se (6gb/256gb) - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nIPS LCD, 90Hz<br> Độ phân giải:\nHD+ (720 x 1600), Camera chính 50MP f/1.8, Camera chiều sâu 2MP f/2.4, Camera macro 2MP f/2.4, 8MP f/2.0<br> Kích thước màn hình:\n6.75 inch<br> Hệ điều hành:\nAndroid 13<br> Vi xử lý:\nMediaTek Helio G37<br> Bộ nhớ trong:\n256GB<br> RAM:\n6GB<br> Mạng di động:\n2G, 3G, 4G<br> Số khe SIM:\n2 nano SIM<br> Dung lượng pin:\n5010 mAh<br>\n\nPromotions:\n- Giảm 5% không giới hạn khuyến mãi qua Homepaylater<br>- Giảm 50% tối đa 700k khi mở thẻ tín dụng Vpbank trên SenID<br>- Giảm 20% tối đa 500k khi mở thẻ tín dụng TPBank EVO<br>- Mở thẻ tín dụng VIB - Nhận Voucher 600.000đ<br>- Giảm 1% tối đa 100.000đ khi thanh toán qua Zalopay<br>\n\nPrice:\n2,990,000 ₫\n\nColors:\nMàu Tím, Màu Xám\n\nReference id 4b8ff72d-489a-4aa5-8adb-9757fa24fd6a: Product Title: điện thoại tcl 408 (4gb/128gb) - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nIPS LCD<br> Độ phân giải:\nHD+ (720 x 1612), Camera chính 50MP f/1.8, Camera chiều sâu 2MP f/2.4, 8MP f/2.0<br> Kích thước màn hình:\n6.6 inch<br> Hệ điều hành:\nAndroid 12<br> Vi xử lý:\nMediaTek MT6762<br> Bộ nhớ trong:\n128GB<br> RAM:\n4GB<br> Mạng di động:\n2G, 3G, 4G<br> Số khe SIM:\n2 nano SIM<br> Dung lượng pin:\n5000 mAh<br>\n\nPromotions:\n- Giảm 5% không giới hạn khuyến mãi qua Homepaylater<br>- Giảm 50% tối đa 700k khi mở thẻ tín dụng Vpbank trên SenID<br>- Giảm 20% tối đa 500k khi mở thẻ tín dụng TPBank EVO<br>- Giảm 1% tối đa 100.000đ khi thanh toán qua Zalopay<br>\n\nPrice:\n2,190,000 ₫\n\nColors:\nMàu Xám, Xanh Dương"

        results = self.evaluator.eval(
            questions=[question],
            response_llms=[response_llm],
            contexts=[context]
        )

        result = results[0]
        self.assertIn("groundedness", result)
        self.assertEqual(result['groundedness'], 1.0)
    def test_eval_noise_sensitivity(self):
        self.evaluator = Evaluator(
            metrics=[NOISE_SENSITIVITY],
            model_type="online",
            model_name="gemini",
            model_version="gemini-2.5-pro"
        )

        question = "Điện thoại Samsung Galaxy A34 5G có bao nhiêu camera và độ phân giải của chúng là bao nhiêu?"
        response_llm = "Samsung Galaxy A34 5G có 4 camera: 8MP F2.2 (Siêu Rộng), 48MP F1.8 OIS (Rộng), 5MP F2.4 (Cận cảnh), 13MP F2.2. Độ phân giải của các camera là FHD+, 1080x2400 pixels."
        context = "Reference id f6d0fab3-bc07-40e2-8a69-421becdd1a84: Product Title: điện thoại samsung galaxy a34 5g 8gb/128gb - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nSuper AMOLED , 120Hz ,19.5:9<br> Độ phân giải:\nFHD+, 8MP F2.2 (Siêu Rộng), 48MP F1.8 OIS (Rộng), 5MP F2.4 (Cận cảnh), 13MP F2.2<br> Kích thước màn hình:\n6.6 inch<br> Vi xử lý:\nDimensity 1080 – 5nm, (Octa Core 2 x 2.6GHz + 6 x 2.0GHz)<br> Bộ nhớ trong:\n128GB<br> RAM:\n8GB<br> Mạng di động:\n5G<br> Dung lượng pin:\n5000 mAh<br>\n\nPromotions:\n\n\nPrice:\n\n\nColors:\n\n\nReference id 09eb373b-be88-4c3b-b141-2fd1a67c4fbd: Product Title: điện thoại samsung galaxy a34 5g 8gb/256gb - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nSuper AMOLED , 120Hz ,19.5:9<br> Độ phân giải:\nFHD+, 8MP F2.2 (Siêu Rộng), 48MP F1.8 OIS (Rộng), 5MP F2.4 (Cận cảnh), 13MP F2.2<br> Kích thước màn hình:\n6.6 inch<br> Vi xử lý:\nMediaTek Dimensity 1080<br> Bộ nhớ trong:\n256GB<br> RAM:\n8GB<br> Mạng di động:\n5G<br> Dung lượng pin:\n5000 mAh<br>\n\nPromotions:\n- Ưu đãi trả góp 0% qua Shinhan Finance hoặc Mirae Asset Finance<br>- Giảm 5% không giới hạn khuyến mãi qua Homepaylater<br>- Giảm thêm tới 700.000đ khi thanh toán qua Kredivo.<br>- Giảm 50% tối đa 700k khi mở thẻ tín dụng Vpbank trên SenID<br>- Giảm 20% tối đa 500k khi mở thẻ tín dụng TPBank EVO<br>- Mở thẻ tín dụng VIB - Nhận Voucher 600.000đ<br>- Giảm 1% tối đa 100.000đ khi thanh toán qua Zalopay<br>\n\nPrice:\n5,990,000 ₫\n\nColors:\nMàu Đen, Bạc, Xanh\n\nReference id 8719a4f9-e404-4203-a4d7-7fabf1193a86: Product Title: điện thoại samsung galaxy m34 5g 8gb/128gb - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nInfinity U, Super AMOLED<br> Độ phân giải:\nFHD+ (1080x2400), 19.5.9, 120Hz, 8MP (F2.2), 50MP (F2.2), 2MP (F2.4), 13MP (F2.2)<br> Kích thước màn hình:\n6.5 inch<br> Vi xử lý:\nExynos 1280 5nm, 8 nhân, 2.4GH2<br> Bộ nhớ trong:\n128GB<br> RAM:\n8GB<br> Dung lượng pin:\n6000 mAh<br>\n\nPromotions:\n- KM 1<br>- Sản phẩm đang thuộc chương trình Flash sale (Số lượng có hạn)<br>\n\nPrice:\n6,990,000 ₫\n\nColors:\nXanh Nhạt, Xanh đậm\n\nReference id 11e34103-6584-48fc-a4a5-3a028d86f7f6: Product Title: điện thoại samsung galaxy a33 5g - chính hãng\n\nProduct Specifications:\nCông nghệ màn hình:\nSuper AMOLED<br> Độ phân giải:\nFull HD+ (1080 x 2400 Pixels), Chính 48 MP & Phụ 8 MP, 5 MP, 2 MP, 13 MP<br> Kích thước màn hình:\n6.4 inch<br> Hệ điều hành:\nAndroid 12<br> Vi xử lý:\nExynos 1280 8 nhân<br> Bộ nhớ trong:\n128 GB<br> RAM:\n6GB<br> Mạng di động:\nHỗ trợ 5G<br> Số khe SIM:\n2 Nano SIM (SIM 2 chung khe thẻ nhớ)<br> Dung lượng pin:\n5000 mAh, 25 W, Sạc pin nhanh<br>\n\nPromotions:\n\n\nPrice:\n\n\nColors:"
        results = self.evaluator.eval(
            questions=[question],
            response_llms=[response_llm],
            contexts=[context]
        )

        result = results[0]
        self.assertIn("noise_sensitivity", result)
        self.assertEqual(result['noise_sensitivity'], 0.17)
    def test_eval_invalid_metrics(self):
        evaluator = Evaluator(metrics=["invalid_metric"])
        with self.assertRaises(ValueError):
            evaluator.eval(
                questions=["Q"],
                response_llms=["R"],
                ground_truths=["G"],
                contexts=["C"]
            )

    def test_eval_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.evaluator.eval(
                questions=["Q1", "Q2"],
                response_llms=["R1"],
                ground_truths=["G1"],
                contexts=["C1"]
            )

    def test_eval_empty_lists(self):
        with self.assertRaises(ValueError):
            self.evaluator.eval(
                questions=[],
                response_llms=[],
                ground_truths=[],
                contexts=[]
            )


if __name__ == "__main__":
    unittest.main(testRunner=LoggingTestRunner, verbosity=2)
