from pathlib import Path
import arpes.config
from arpes.io import example_data
from arpes.plotting.basic_tools import bkg_tool

data = example_data.cut


bkg_tool(data)
