import sys
import os
import unittest
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_generator import generate_single_reading
from src.core.database import init_db

class TestDetectionLogic(unittest.TestCase):
    
    def setUp(self):
        init_db()

    def test_generator_output(self):
        """Verify the shape of the generated readings."""
        reading = generate_single_reading(anomaly=False)
        self.assertEqual(reading.shape, (1, 3))
        
        reading_anomaly = generate_single_reading(anomaly=True)
        self.assertEqual(reading_anomaly.shape, (1, 3))

    def test_anomaly_values(self):
        """Ensure anomaly values are significantly higher than normal ones."""
        normal = generate_single_reading(anomaly=False)
        anomaly = generate_single_reading(anomaly=True)
        
        # Check temperature (First column)
        self.assertGreater(anomaly[0][0], normal[0][0])
        # Check speed (Second column)
        self.assertGreater(anomaly[0][1], normal[0][1])

if __name__ == "__main__":
    unittest.main()
