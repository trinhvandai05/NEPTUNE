#!/usr/bin/env python
"""Print the implementation fingerprint (sha256 of src/neptune, scripts, pyproject).
Freezing it into a preregistration is a deliberate manual edit, not something this
script does: changing it changes the prereg hash and therefore every namespace."""
import _cli  # noqa: F401
from neptune.provenance import implementation_fingerprint

fp = implementation_fingerprint()
print(fp["sha256"])
print(f"({fp['n_files']} files)")
