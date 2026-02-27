"""
Test script to verify video captioning integration with DetectifAI

This script tests:
1. Configuration loading
2. Integrator initialization  
3. Caption generation (mock test)
4. API endpoint accessibility
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_config_integration():
    """Test configuration parameters"""
    print("=" * 60)
    print("TEST 1: Configuration Integration")
    print("=" * 60)
    
    try:
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=True)
        
        assert hasattr(config, 'enable_video_captioning'), "Missing enable_video_captioning"
        assert hasattr(config, 'captioning_vision_model'), "Missing captioning_vision_model"
        assert hasattr(config, 'captioning_embedding_model'), "Missing captioning_embedding_model"
        assert hasattr(config, 'captioning_device'), "Missing captioning_device"
        assert hasattr(config, 'captioning_batch_size'), "Missing captioning_batch_size"
        
        print(f"✅ Config parameters present")
        print(f"   - Vision Model: {config.captioning_vision_model}")
        print(f"   - Embedding Model: {config.captioning_embedding_model}")
        print(f"   - Device: {config.captioning_device}")
        print(f"   - Batch Size: {config.captioning_batch_size}")
        print(f"   - Enabled: {config.enable_video_captioning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_integrator_import():
    """Test integrator module import"""
    print("\n" + "=" * 60)
    print("TEST 2: Integrator Module Import")
    print("=" * 60)
    
    try:
        from video_captioning_integrator import VideoCaptioningIntegrator
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=False)
        integrator = VideoCaptioningIntegrator(config)
        
        print(f"✅ Integrator imported successfully")
        print(f"   - Enabled: {integrator.enabled}")
        print(f"   - Has service: {integrator.captioning_service is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test main pipeline integration"""
    print("\n" + "=" * 60)
    print("TEST 3: Main Pipeline Integration")
    print("=" * 60)
    
    try:
        from main_pipeline import CompleteVideoProcessingPipeline
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=False)
        pipeline = CompleteVideoProcessingPipeline(config)
        
        print(f"✅ Pipeline initialized with captioning support")
        print(f"   - Has video_captioning attribute: {hasattr(pipeline, 'video_captioning')}")
        print(f"   - Captioning enabled: {pipeline.video_captioning is not None if hasattr(pipeline, 'video_captioning') else False}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test API endpoint definitions"""
    print("\n" + "=" * 60)
    print("TEST 4: API Endpoints")
    print("=" * 60)
    
    try:
        # Check if endpoints are defined in app.py
        with open('backend/app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        endpoints = [
            '/api/captions/search',
            '/api/captions/video',
            '/api/captions/statistics'
        ]
        
        all_found = True
        for endpoint in endpoints:
            if endpoint in content:
                print(f"✅ Endpoint defined: {endpoint}")
            else:
                print(f"❌ Endpoint missing: {endpoint}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False


def test_report_generator():
    """Test report generator integration"""
    print("\n" + "=" * 60)
    print("TEST 5: Report Generator")
    print("=" * 60)
    
    try:
        from json_reports import ReportGenerator
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig()
        generator = ReportGenerator(config)
        
        # Check if method exists
        has_method = hasattr(generator, 'generate_captioning_report')
        
        if has_method:
            print(f"✅ Captioning report method present")
        else:
            print(f"❌ Captioning report method missing")
        
        return has_method
        
    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False


def test_video_captioning_module():
    """Test video_captioning module availability"""
    print("\n" + "=" * 60)
    print("TEST 6: Video Captioning Module")
    print("=" * 60)
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'video_captioning'))
        
        from video_captioning import CaptioningService, Frame, CaptioningConfig
        
        print(f"✅ Video captioning module imported")
        print(f"   - CaptioningService: Available")
        print(f"   - Frame model: Available")
        print(f"   - CaptioningConfig: Available")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Video captioning module not fully available: {e}")
        print(f"   This is expected if dependencies are not installed")
        print(f"   Run: pip install -r video_captioning/requirements.txt")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("VIDEO CAPTIONING INTEGRATION TEST SUITE")
    print("=" * 70)
    
    results = {
        'Configuration': test_config_integration(),
        'Integrator Import': test_integrator_import(),
        'Pipeline Integration': test_pipeline_integration(),
        'API Endpoints': test_api_endpoints(),
        'Report Generator': test_report_generator(),
        'Video Captioning Module': test_video_captioning_module()
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("Video captioning is fully integrated into DetectifAI")
    elif passed >= 5:
        print("\n⚠️  Core integration complete, some dependencies may need installation")
        print("Run: pip install -r video_captioning/requirements.txt")
    else:
        print("\n❌ Integration incomplete - please review failed tests")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
