import arpes.config
from arpes.io import example_data
from arpes.plotting.qt_ktool import ktool

data = example_data.cut
data.attrs["id"] = 0
print(f"data.id : {data.attrs['id']}")
print(f"data.id (spectrum): {data.spectrum.attrs['id']}")
ktool(data)
