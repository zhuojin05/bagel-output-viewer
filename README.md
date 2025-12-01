# **BAGEL Output Viewer**
The simple & flexible tools summarise structure quality & generates PAE heatmaps from the outputs from the MCMC protein design platform ***BAGEL*** ([Lála *et al.*, 2025](https://www.biorxiv.org/content/early/2025/07/08/2025.07.05.663138)).
- [github.com/softnanolab/bagel](https://github.com/softnanolab/bagel)


| Script | Description |
|-------------------------|------------------|
| `view_bagel.py` | 1) Generate PAE heatmaps from all `.pae` files; 2) Ranks all structures according to folding metrics & energies from outputted `energies.csv`, and generates a 3D energy landscape of 
| `generate_landscape.py` | Converts the 5D "energy landscape" (currently hard-coded for energy terms: `pTM`, `global_pLDDT`, `hydrophobic`, `cross_PAE`, `separation`) into a 3D representation in `.png`, `.gif` & `.csv` formats.



## Installation

1. clone repository from Github. 
```bash
git clone https://github.com/zhuojin05/bagel-output-viewer.git
```
2. Install uv (if not already installed).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
3. Set up dependencies from `pyproject.toml` and creates `.venv/`. 
```bash 
uv sync
```
<details >
<summary>( Optional ) For non-uv users: Set up dependencies using pip. </summary>

```bash
pip install numpy matplotlib pandas
```
</details>

## Usage
### A) Running `view_bagel.py`:

Run script from the `scripts/` working directory. 
```bash
uv run python view_bagel.py
```
After succesfully running the `view_bagel.py` script, the user will be prompted to input path to the BAGEL output folder containing simulated structures and folding metrics (PAE, pLDDT). 


### Input format - absolute path to BAGEL output
The user should enter the absolute path to the folder which directly contains `.pae`, `.plddt`, etc. files corresponding to selected simulated structures by BAGEL.

```bash
simulated_METHOD_YYMMDD_HHMMSS/best/structures/
    ├── *.pae
    ├── *.plddt
```
( * = represents the ID of the simulated structures. )

The user is prompted in the terminal in teh following way:
```bash
=== AlphaFold PAE / pLDDT evaluator & ranker ===
Enter the path to a folder containing your .pae and .plddt files:
> 
```

Enter the absolute path corresponding to the output `best/structures/` folder into the terminal after the `>` sign. 
```bash
> ABSOLUTE_PATH
```
### B) Running `generate_landscape.py`:

Run script from the `scripts/` working directory with the absolute path to the BAGEL output folder as the command-line argument.  
```bash
uv run python generate_landscape.py .../simulated_METHOD_YYMMDD_HHMMSS
```
The inputted BAGEL output folder path has the format of:
```bash
simulated_METHOD_YYMMDD_HHMMSS/
    ├── best/
    ├── current/
    ├── config.csv
    ├── optimization.log
    ├── version.txt
```
Runtime for this script is longer (~10 min).


## Output
Outputs from both `view_bagel.py` and `generate_landscape.py` scripts are saved in separate folders within the `results/` folder:
```bash
bagel-output-viewer/results/
```
### A) Output example for `view_bagel.py`:
```bash
bagel-output-viewer/
└── results/
    └── results_251201_022033/
        ├── structure_quality_summary_251201_022033.csv
        ├── pae_jpgs_251201_022033/
        │   ├── pae_001.jpg
        │   ├── pae_002.jpg
        │   ├── pae_003.jpg
        │   └── ...
        └── (other run-specific files, if any)
```

### B) Output example for `generate_landscape.py`:
```bash
bagel-output-viewer/
└── results/
    └── results_251201_022033/
        ├── pca_landscape.csv
        ├── energy_landscape_pca.png
        └── energy_landscape_pca_trajectory.gif
```

## Troubleshooting
For VS Code users, if you see "Import could not be resolved", ensure VS Code is using the `.venv/` created via `uv sync`:
- Open Command Palette → "Python: Select Interpreter"
- Choose `.venv/bin/python` or `./bagel-output-viewer/bin/python`.

## License
MIT
