import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromName('test_f1_score.F1Test'))
    suite.addTests(loader.loadTestsFromName('test_groundedness.GroundednessTest'))
    suite.addTests(loader.loadTestsFromName('test_noisesensitivy.NoisesensitivyTest'))
    

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
