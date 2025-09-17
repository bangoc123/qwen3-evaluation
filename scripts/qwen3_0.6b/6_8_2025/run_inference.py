import unittest
from test_noisesensitivy import NoisesensitivyTest
from test_groundedness import GroundednessTest
from test_f1_score import F1Test

if __name__ == '__main__':
    # loader = unittest.TestLoader()
    # suite = unittest.TestSuite()

    # suite.addTests(loader.loadTestsFromName('test_f1_score.F1Test'))
    # # suite.addTests(loader.loadTestsFromName('test_groundedness.GroundednessTest'))
    # # suite.addTests(loader.loadTestsFromName('test_noisesensitivy.NoisesensitivyTest'))
    

    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(suite)

    # Sau khi test xong, gọi retry function
    print("\n" + "="*80)
    print("Starting retry for empty label_statements...")
    print("="*80)
    
    # Tạo instance của test class
    # test_instance = NoisesensitivyTest()
    # test_instance = GroundednessTest()
    test_instance = F1Test()
    test_instance.setUpClass()  # Gọi setUp nếu cần
    
    # Gọi retry function
    test_instance.retry_empty_label_statements()