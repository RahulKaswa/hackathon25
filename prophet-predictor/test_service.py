#!/usr/bin/env python3
"""
Test script for the enhanced predictive scaling service.
This script validates the key functionality and performance improvements.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


class PredictorServiceTester:
    """Test class for the predictor service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check passed")
                print(f"   Version: {health_data.get('version', 'unknown')}")
                print(f"   Models: {len(health_data.get('models', {}))}")
                print(f"   Cache: {health_data.get('cache', {}).get('enabled', 'unknown')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_predict_endpoint(self) -> bool:
        """Test the prediction endpoint."""
        try:
            # Test general prediction
            response = self.session.get(f"{self.base_url}/predict")
            if response.status_code == 200:
                predictions = response.json()
                print("✅ General prediction endpoint working")
                print(f"   Metrics predicted: {len(predictions)}")
                
                # Test specific metric prediction
                for metric_name in predictions.keys():
                    response = self.session.get(f"{self.base_url}/predict?metric={metric_name}")
                    if response.status_code == 200:
                        metric_pred = response.json()
                        print(f"   ✅ {metric_name}: {metric_pred.get('next_value', 'N/A')}")
                        print(f"      Algorithm: {metric_pred.get('algorithm', 'unknown')}")
                        print(f"      Trend: {metric_pred.get('trend', 'unknown')}")
                    break
                return True
            else:
                print(f"❌ Prediction endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Prediction endpoint error: {e}")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test the Prometheus metrics endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                metrics_text = response.text
                if "predicted_load" in metrics_text:
                    print("✅ Metrics endpoint working")
                    print(f"   Metrics size: {len(metrics_text)} bytes")
                    return True
                else:
                    print("❌ No predicted_load metrics found")
                    return False
            else:
                print(f"❌ Metrics endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics endpoint error: {e}")
            return False
    
    def test_config_endpoint(self) -> bool:
        """Test the config endpoint (if debug is enabled)."""
        try:
            response = self.session.get(f"{self.base_url}/config")
            if response.status_code == 200:
                config_data = response.json()
                print("✅ Config endpoint working (debug mode enabled)")
                print(f"   Prometheus URL: {config_data.get('prometheus', {}).get('url', 'unknown')}")
                return True
            elif response.status_code == 403:
                print("ℹ️  Config endpoint protected (debug mode disabled)")
                return True
            else:
                print(f"❌ Config endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Config endpoint error: {e}")
            return False
    
    def performance_test(self, num_requests: int = 10) -> Dict[str, Any]:
        """Run a simple performance test."""
        print(f"🚀 Running performance test with {num_requests} requests...")
        
        start_time = time.time()
        successful_requests = 0
        response_times = []
        
        for i in range(num_requests):
            req_start = time.time()
            try:
                response = self.session.get(f"{self.base_url}/predict")
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
            except Exception as e:
                print(f"   Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        results = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests * 100,
            "total_time": total_time,
            "average_response_time": avg_response_time,
            "requests_per_second": successful_requests / total_time if total_time > 0 else 0
        }
        
        print(f"📊 Performance Results:")
        print(f"   Success Rate: {results['success_rate']:.1f}%")
        print(f"   Average Response Time: {results['average_response_time']:.3f}s")
        print(f"   Requests/Second: {results['requests_per_second']:.1f}")
        
        return results
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("🧪 Starting Predictor Service Tests\n")
        
        all_passed = True
        
        # Test health endpoint
        if not self.test_health_endpoint():
            all_passed = False
        print()
        
        # Test prediction endpoint
        if not self.test_predict_endpoint():
            all_passed = False
        print()
        
        # Test metrics endpoint
        if not self.test_metrics_endpoint():
            all_passed = False
        print()
        
        # Test config endpoint
        if not self.test_config_endpoint():
            all_passed = False
        print()
        
        # Performance test
        self.performance_test()
        print()
        
        if all_passed:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return all_passed


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the enhanced predictor service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--perf-requests", type=int, default=10, help="Number of requests for performance test")
    args = parser.parse_args()
    
    tester = PredictorServiceTester(args.url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
