#!/bin/bash
set -e
echo "DISTRO=${WSL_DISTRO_NAME:-unknown}"
/home/alecs/pml_venv/bin/python -m pip install -q ipykernel
/home/alecs/pml_venv/bin/python -m ipykernel install --user --name pml-wsl-gpu --display-name "Python (PML WSL GPU)"
/home/alecs/pml_venv/bin/python /mnt/c/Users/Alecs/PML/Project_1/_check_wsl_gpu.py
