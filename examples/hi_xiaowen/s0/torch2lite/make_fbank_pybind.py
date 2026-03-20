#!/usr/bin/env python3

from pathlib import Path

import fbank_pybind


def main():
    module = fbank_pybind._load_extension()
    print("built_module:", module.__file__)
    print("build_dir:", str((Path(__file__).resolve().parent / ".fbank_pybind_build").resolve()))


if __name__ == "__main__":
    main()
