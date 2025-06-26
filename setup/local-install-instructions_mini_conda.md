# 🧪 Setup Instructions for `data_science_2025`

## 📦 1. Install Miniconda (Optional)

> ⚠️ Skip this step if you already have `conda` installed.

1. Download the appropriate [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html) for your operating system (Windows, macOS, or Linux).
2. Follow the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for your platform.

---

## 📦 2. Clone the Repository and Create the Environment

Open a terminal and run the following commands:
```sh
# Clone this repository
git clone https://github.com/mikesips/data_science_2025.git

# Create a conda environment using the provided environment.yml (Linux/macOS)
conda env create -f requirements/requirements_conda.yml

# On Windows, use:
conda env create -f requirements\requirements_conda.yml

# go to repo
cd data_science_2025
```

## 📦 3. Check your install

To make sure you have all the necessary packages installed, we **strongly
recommend** you to execute the `check_env.py` script located at the root of
this repository:

```sh
# Activate your conda environment
conda activate data_science_2025

# Check your conda environment (Linux/macOS)
python check_env.py ./requirements/requirements_venv.yml

# On Windows, use:
python check_env.py requirements\requirements_venv.yml

```

Make sure that there is no `FAIL` in the output when running the `check_env.py`
script, i.e. that its output looks similar to this:

```
```
