# Thermal scenario synthesis

This directory provides the thermal counterpart to `ScenarioSynthesis`:

- `case_generator.py` creates a random thermal case and obtains its mass-flow
  state with the custom solver in `ScenarioSynthesis_hydraulic`.
- `newton_raphson.py` constructs the flow-oriented incidence matrix, nodal heat
  residual, analytic Jacobian, Newton update, damping, and convergence test. It
  uses PyDHN only for component outlet-temperature laws.
- `generate_dataset.py` writes hydraulic inputs, thermal inputs and outputs,
  topology, and residual histories to Parquet.
- `test_against_pydhn.py` checks the custom result against PyDHN.

Install and verify from the repository root:

```bash
python -m pip install -r ScenarioSynthesis_thermal/requirements.txt
python -m pytest ScenarioSynthesis_thermal/test_against_pydhn.py -q
```

Run one verbose case:

```bash
python -m ScenarioSynthesis_thermal.newton_raphson
```

Generate a dataset:

```bash
python -m ScenarioSynthesis_thermal.generate_dataset \
  --runs 1000 --seed 0 --min-nodes 4 --max-nodes 32 \
  --workers 4 --batch-size 100 \
  --output out/thermal_newton_cases.parquet
```

Use `--overwrite` to replace an existing output explicitly.

The node count is selected deterministically from the seed across the inclusive
4--32 range. The Fritz scripts in `sbatch/` generate 18 chunks of 2,000 rows
and merge them into `thermal_NR_36000_4_to_32.parquet` only after every array
task succeeds.

The Parquet array columns contain NumPy `.npy` payloads. Read one with:

```python
import io
import numpy as np
import pyarrow.parquet as pq

row = pq.read_table("out/thermal_newton_cases.parquet").slice(0, 1).to_pylist()[0]
temperature = np.load(
    io.BytesIO(row["node_temperature_newton_c"]), allow_pickle=False
)
```

For differentiable surrogate losses, every row also contains
`outlet_temperature_derivative` and `outlet_temperature_intercept_c`. Together
they reconstruct the fixed-flow PyDHN component law as
`T_out = derivative * T_in + intercept` without instantiating a network inside
the training loop.

The heat-balance equations are described in the
[PyDHN thermal simulation guide](https://idiap.github.io/pydhn/get_started/simulation.html#thermal-simulation).
