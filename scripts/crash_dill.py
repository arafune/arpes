"""Demo-ing a dill/xarray crash."""

import dill

from arpes.io import example_data

data = example_data.map
data = data.assign_coords(**dict(data.coords))  # without this line there's a crash

print(data)
dill.loads(dill.dumps(data))

print("Hi")
