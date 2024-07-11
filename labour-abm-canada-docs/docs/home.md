# Labour ABM

This repository contains the code for the agent-based model (ABM) of labour markets,
as described in [Occupational mobility and automation: a data-driven network model](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2020.0898)
by del Rio-Chanona, R., Mealy, P., Pichler, A., Lafond, F., & Farmer, J. D. (2020).

The ABM is implemented in Python (3.12) and uses Pytorch, and requires pre-treated data for
the occupational transition network and the demand scenarios (for instance, those developped by the 
Sustainable Energy Systems Integration & Transitions (SESIT) group at the University of Victoria).

## Installation

First, you need to clone the repository:

```bash
git clone [repository url depending on the remote you are using]
```

### Installation with virtualenv

Then, you need to install the required packages. We recommend using a virtual environment to avoid conflicts with other packages. 
You can create a virtual environment using the following command:

```bash
python3 -m venv venv
```

Then, you can activate the virtual environment:

```bash
source venv/bin/activate
```

Finally, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

and then you can install the package using the following command, making sure you
are in the root directory of the repository (containing the `setup.py` file):

```bash
pip install .
```
### Installing into a conda environment

Alternatively, you can install the package into a conda environment. First, you need to create a conda environment:

```bash
conda create -n labour-abm python=3.12
```
although note that python 3.10 or 3.11 should also work. Then, you can activate the environment:

```bash
conda activate labour-abm
```

Finally, you can install the required packages using pip:

```bash

pip install -r requirements.txt
```

and then you can install the package using the following command, making sure you
are in the root directory of the repository (containing the `setup.py` file):

```bash
pip install .
```

## Usage

You can run the ABM using the following command in your terminal, and making
sure that you have access to a scenario file. An example scenario file is available in 
[net_new_cap_w_jobs.csv](test_labour_abm_canada%2Ftest_data%2Fnet_new_cap_w_jobs.csv).

You may then run using
```bash
python -m labor_abm_canada --scenario "./test_labour_abm_canada/test_data/net_new_cap_w_jobs.csv"
```
or any other scenario. By default, this will run the ABM at the national level.

You can run in other regions by specifying the region in the command line, for instance:

```bash
python -m labor_abm_canada --scenario XXXX --region Alberta.a
```

Note that the available regions are 

```angular2html
all_regions = [
    "Alberta.a",
    "British Columbia.a",
    "Manitoba.a",
    "New Brunswick.a",
    "Newfoundland and Labrador.a",
    "Nova Scotia.a",
    "Ontario.a",
    "Ontario.b",
    "Prince Edward Island.a",
    "Quebec.a",
    "Quebec.b",
    "Saskatchewan.a",
    "National",
]
```

## Testing

You can run the tests using the following command:

```bash
pytest test_labour_abm_canada
```

The tests in the `test_labour_abm_canada` directory can be used as a guideline on how to run this in
Python scripts rather than in the command line.

## License

This code is licensed under the CC-BY-NC 4.0 license. See the [LICENSE](LICENSE) file for more information.
