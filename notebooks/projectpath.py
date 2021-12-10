"""Sets up the python path so notebooks can refer to project code.

The first line of a notebook should be
`import projectpath`

After this, any python modules in this project can be imported by their normal paths.

Thanks to https://stackoverflow.com/questions/38237284/setting-a-default-sys-path-for-a-notebook
"""

import sys
import pathlib
sys.path[0] = str(pathlib.Path().resolve().parent)
