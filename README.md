# aind-smartspim-classification

Code for classifying the cell candidate outputs from [aind-smartspim-segmentation](https://github.com/AllenNeuralDynamics/aind-smartspim-segmentation) within the SmartSPIM pipeline.
It uses the [aind-large-scale-prediction](https://github.com/AllenNeuralDynamics/aind-large-scale-prediction) package to efficiently process large amounts of image data.

This repository takes as input the cell proposals that will be classified by the CellFinder model. The output is a CSV with the following columns:

- **Cell Counts**: Number of positive cells.
- **Cell Likelihood Mean**: Mean of the probabilities of a cell being a cell.
- **Cell Likelihood STD**: Standard deviation of the probability of a cell being a cell.
- **Noncell Counts**: Number of negative cells.
- **Noncell Likelihood Mean**: Mean of the probabilities of negative cells.
- **Noncell Likelihood STD**: Standard deviation of the probabilities of negative cells.

---

## Prerequisites

- Python 3.9
- A CUDA-capable GPU (the pipeline is designed to run on GPU; see `environment/Dockerfile` for the full hardware/software stack)
- [Conda](https://docs.conda.io/en/latest/) — the production environment uses the `cell_class` conda environment defined in `environment/environment.yml`

For **local development** outside of Code Ocean, a CUDA GPU is not strictly required to run tests, but is needed to run the full pipeline.

---

## Installation

From the **repository root**, install the package and its dependencies:

```bash
pip install -e .
```

To install development tools (linters, test runner, coverage):

```bash
pip install -e ".[dev]"
```

> **Note**: The Python package (`aind_smartspim_classification`) lives inside `code/`. The `pyproject.toml` at the root configures `setuptools` to find it there automatically.

---

## Running in Code Ocean

The capsule entry point is `code/run`. It activates the `cell_class` conda environment, sets `KERAS_BACKEND=torch`, and executes `code/run_capsule.py`. Inputs are mounted at `/data` and outputs are written to `/results`.

For SLURM-based execution, use `code/run_slurm` instead.

---

## Contributing

### Linters and testing

After installing with `pip install -e ".[dev]"`, the following tools are available:

- Run the test suite with coverage:

```bash
pytest code/tests/ --cov=aind_smartspim_classification --cov-report=term-missing
```

- Check documentation coverage:

```bash
interrogate .
```

- Check code style:

```bash
flake8 code/
```

- Auto-format code:

```bash
black code/
```

- Sort imports:

```bash
isort code/
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages:

```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml)
- **ci**: Changes to CI configuration files and scripts
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation

To generate RST source files for Sphinx documentation, run from the repo root:

```bash
sphinx-apidoc -o doc_template/source/ code/
```

Then build HTML:

```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```

More info on Sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
