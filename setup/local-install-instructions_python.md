# 🧪 Setup Instructions for `data_science_2025`

This guide will help you set up a Python virtual environment and install the required dependencies for the `data_science_2025` project.

---

## 📦 1. Clone the Repository and Set Up the Environment

Open a terminal and run the following commands:

```sh
# Clone the repository
git clone https://github.com/mikesips/data_science_2025.git

# Navigate into the project directory
cd data_science_2025

# Create a virtual environment (named 'data_science_2025')
python3 -m venv data_science_2025

# Activate the virtual environment (Linux/macOS)
source data_science_2025/bin/activate

# On Windows, use:
data_science_2025\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install required dependencies (Linux/macOS)
python -m pip install -r requirements/requirements_venv.txt

# On Windows, use:
python -m pip install -r requirements\requirements_venv.txt

```

## 📦 2. Check your install and environment

To confirm that your environment is correctly configured, run the check_env.py script:
```sh
# Linux/macOS
python check_environment.py requirements/requirements_venv.yml

# On Windows, use:
python check_environment.py requirements\requirements_venv.yml
```

If everything is correctly installed, the output should look like:
```
3.11.2 (main, Apr 28 2025, 14:11:48) [GCC 12.2.0]
[ OK ] Python 3.11 is compatible.

[INFO] Loading requirements from requirements.yml

[INFO] Checking installed packages

[ OK ] dask version 2025.5.1
[ OK ] pystac_client version 0.8.6
[ OK ] matplotlib version 3.10.3
[ OK ] geopandas version 1.1.0
[ OK ] rioxarray version 0.19.0
[ OK ] shapely version 2.1.1

[INFO] Environment check complete. All installable requirements have been processed.

```

## 📦 3. Alternatively, use bash script
```sh
# Clone the repository
git clone https://github.com/mikesips/data_science_2025.git

# Navigate into the project directory
cd data_science_2025

# Create a virtual environment (named 'data_science_2025')
python3 -m venv data_science_2025

# Activate the virtual environment (Linux/macOS)
source data_science_2025/bin/activate

# Run Bash Script
./run_env_setup.sh