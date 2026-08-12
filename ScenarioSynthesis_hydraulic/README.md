# Hydraulic scenario synthesis

This directory mirrors the existing `ScenarioSynthesis` pattern for PyDHN
hydraulics:

- `case_generator.py` creates reproducible, physically varied cases on a valid
  meshed district-heating topology.
- `newton_raphson.py` implements the cycle residual, Jacobian, Newton update,
  convergence test, damping, and pressure reconstruction. It uses PyDHN only
  for the component pressure-drop laws.
- `generate_dataset.py` writes inputs, topology matrices, Newton outputs, and
  residual histories to Parquet.
- `test_against_pydhn.py` checks the custom solution against PyDHN.

Install and verify from the repository root:

```bash
python -m pip install -r ScenarioSynthesis_hydraulic/requirements.txt
python -m pytest ScenarioSynthesis_hydraulic/test_against_pydhn.py -q
```

Run one verbose case:

```bash
python -m ScenarioSynthesis_hydraulic.newton_raphson
```

Generate a dataset:

```bash
python -m ScenarioSynthesis_hydraulic.generate_dataset \
  --runs 1000 --seed 0 --min-nodes 4 --max-nodes 32 \
  --workers 4 --batch-size 100 \
  --output out/hydraulic_newton_cases.parquet
```

The default hydraulic tolerance is PyDHN's recommended `100 Pa`. Use
`--error-threshold` to change it and `--overwrite` to replace an existing
output explicitly.

The node count is selected deterministically as
`min_nodes + seed % (max_nodes - min_nodes + 1)`. The Fritz production scripts
in `sbatch/` run 18 array tasks of 2,000 rows and merge them into
`hydraulic_NR_36000_4_to_32.parquet` after all tasks succeed.

The Parquet array columns contain NumPy `.npy` payloads. Read one with:

```python
import io
import numpy as np
import pyarrow.parquet as pq

row = pq.read_table("out/hydraulic_newton_cases.parquet").slice(0, 1).to_pylist()[0]
mass_flow = np.load(io.BytesIO(row["mass_flow_newton"]), allow_pickle=False)
```

`hydraulic_setpoint_edge` stores the consumer/producer setpoints in the exact
edge order used by `edges`; this is the preferred field for graph-surrogate
loading because PyDHN mask order is not necessarily sorted by edge index.

The solver follows the loop equation documented in the
[PyDHN hydraulic simulation guide](https://idiap.github.io/pydhn/get_started/simulation.html#hydraulic-simulation).
