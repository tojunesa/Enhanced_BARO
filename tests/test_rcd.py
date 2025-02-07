"""Tests."""
import os
import shutil
from os import path
import subprocess
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile


def test_rcd():
    command = ["python", "main-ase.py", "--method", "rcd", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_ht():
    command = ["python", "main-ase.py", "--method", "ht", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_e_diagnosis():
    command = ["python", "main-ase.py", "--method", "e_diagnosis", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
