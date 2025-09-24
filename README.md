
---

# Inverse Projection Testing

This repository contains code and notebooks for the **Inverse Projection Framework** — analysis, visualization, and experiments. It includes data, results, and a primary Jupyter notebook to run the main projection analysis.

---

## Repository Structure

| Folder / File | Description |
|----------------|-------------|
| `datasets/` | Sample or required input data files to run the analysis. |
| `results/` | Generated outputs, plots, etc., stored here. |
| `main_projection_analysis.ipynb` | The main notebook that executes the framework’s analysis pipeline. |
| `pyproject.toml` | Configuration file for Poetry, listing dependencies. |
| `poetry.lock` | Locked versions of the dependencies. |

---

## Prerequisites

You’ll need:

- Python 3.12  
- Poetry (for dependency management)  
- A working environment for Jupyter / notebooks  

---

## How to Set Up Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Waqar351/inverse_projection_testing.git
   cd inverse_projection_testing
````

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```
---

## How to Run the Notebook

1. Activate your environment (if using virtualenv):

   ```bash
   poetry shell
   ```

2. Launch Jupyter Notebook or Jupyter Lab from the repo directory:

   ```bash
   jupyter notebook main_projection_analysis.ipynb
   ```

---
