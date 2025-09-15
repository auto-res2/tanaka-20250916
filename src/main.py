#!/usr/bin/env python3
"""
MAPLE: Mixed-precision Adaptive Patch-Level Execution
Main experiment runner with configuration support.
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path

from .evaluate import MapleExperiments
from .preprocess import DataPreprocessor

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="MAPLE: Mixed-precision Adaptive Patch-Level Execution")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--full-experiment", action="store_true", help="Run full comprehensive experiments")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        config_path = "config/smoke_test.yaml"
        config = load_config(config_path)
        setup_logging(config.get('logging', {}).get('level', 'INFO'))
        
        experiments = MapleExperiments(config)
        
        experiments.run_smoke_test()
        
    elif args.full_experiment:
        config_path = "config/full_experiment.yaml"
        config = load_config(config_path)
        setup_logging(config.get('logging', {}).get('level', 'INFO'))
        
        experiments = MapleExperiments(config)
        
        experiments.run_full_experiment()
        
    else:
        print("Please specify either --smoke-test or --full-experiment")
        sys.exit(1)

if __name__ == "__main__":
    main()
