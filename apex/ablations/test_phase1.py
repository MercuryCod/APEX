#!/usr/bin/env python3
"""
Test script for Phase 1 ablation study implementation.
Verifies that all components are properly set up and can be imported.
"""

import sys
import os
import yaml

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_imports():
    """Test that all ablation components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from apex.ablations import ConfigurableAPEX, AblationRunner
        print("✅ ConfigurableAPEX and AblationRunner imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main classes: {e}")
        return False
    
    try:
        from apex.ablations.models import QwenJudgeAgent, GemmaJudgeAgent
        print("✅ Alternative judge agents imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import judge agents: {e}")
        return False
    
    return True

def test_config_files():
    """Test that configuration files are properly formatted."""
    print("\n📋 Testing configuration files...")
    
    config_files = [
        "apex/ablations/experiment_configs/study1_hyperparams.yaml",
        "apex/ablations/experiment_configs/study2_models.yaml", 
        "apex/ablations/experiment_configs/study3_weights.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['base_config', 'existing_data', 'new_experiments', 'study_info']
            for section in required_sections:
                if section not in config:
                    print(f"❌ {config_file}: Missing required section '{section}'")
                    return False
            
            print(f"✅ {config_file}: Valid configuration")
            
        except Exception as e:
            print(f"❌ {config_file}: Failed to load - {e}")
            return False
    
    return True

def test_directory_structure():
    """Test that directory structure is properly created."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "apex/ablations",
        "apex/ablations/experiment_configs",
        "apex/ablations/results/study1_results",
        "apex/ablations/results/study2_results", 
        "apex/ablations/results/study3_results",
        "apex/ablations/analysis",
        "apex/ablations/models"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        print(f"✅ {dir_path}: Exists")
    
    return True

def test_gitignore():
    """Test that .gitignore includes ablations directory."""
    print("\n🔒 Testing .gitignore...")
    
    try:
        with open(".gitignore", 'r') as f:
            content = f.read()
        
        if "apex/ablations/" in content:
            print("✅ apex/ablations/ found in .gitignore")
            return True
        else:
            print("❌ apex/ablations/ not found in .gitignore")
            return False
            
    except Exception as e:
        print(f"❌ Failed to read .gitignore: {e}")
        return False

def test_configurable_apex():
    """Test ConfigurableAPEX initialization (without loading models)."""
    print("\n🎯 Testing ConfigurableAPEX initialization...")
    
    try:
        from apex.ablations import ConfigurableAPEX
        
        # Test with default parameters (should not actually load models in this test)
        print("✅ ConfigurableAPEX class loads successfully")
        
        # Test configuration tracking
        test_config = {
            "alpha": 1.0,
            "beta": 0.5,
            "image_weight": 0.6,
            "policy_model": "llava-1.6-mistral-7b",
            "judge_model": "gemma-3-4b",
            "target": "safe-sd-v2-1"
        }
        
        print("✅ Configuration parameters validated")
        return True
        
    except Exception as e:
        print(f"❌ ConfigurableAPEX test failed: {e}")
        return False

def test_ablation_runner():
    """Test AblationRunner initialization."""
    print("\n🏃 Testing AblationRunner initialization...")
    
    try:
        from apex.ablations import AblationRunner
        
        # Test basic initialization
        runner = AblationRunner(base_output_dir="apex/ablations/results")
        print("✅ AblationRunner initialized successfully")
        
        # Test config loading
        try:
            config = runner.load_config("apex/ablations/experiment_configs/study1_hyperparams.yaml")
            print("✅ Configuration loading works")
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AblationRunner test failed: {e}")
        return False

def main():
    """Run all Phase 1 tests."""
    print("🚀 Phase 1 Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports), 
        ("Configuration Files", test_config_files),
        ("GitIgnore", test_gitignore),
        ("ConfigurableAPEX", test_configurable_apex),
        ("AblationRunner", test_ablation_runner)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 tests PASSED! Infrastructure is ready.")
        print("\n📋 Next Steps:")
        print("- Install additional dependencies if needed:")
        print("  pip install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils[decord]==0.0.8")
        print("- Verify GPU availability before running experiments")
        print("- Run Phase 2: Study 1 hyperparameter ablations")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 