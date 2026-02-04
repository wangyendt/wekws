#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 wayne_scripts/config.yaml 配置的命令行工具"""

import sys
from pathlib import Path
from pywayne.tools import read_yaml_config


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 read_config.py <config_key>", file=sys.stderr)
        sys.exit(1)
    
    key = sys.argv[1]
    script_dir = Path(__file__).parent.parent / 'wayne_scripts'
    config_file = script_dir / 'config.yaml'
    
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    
    config = read_yaml_config(str(config_file))
    
    if key not in config:
        print(f"Error: key not found: {key}", file=sys.stderr)
        sys.exit(1)
    
    value = config[key]
    
    # 转换相对路径为绝对路径
    if key == 'download_dir':
        value_path = Path(value)
        if not value_path.is_absolute():
            value_path = (script_dir / value_path).resolve()
        value = str(value_path)
    
    print(value)


if __name__ == '__main__':
    main()
