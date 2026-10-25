"""Load the previous Python block implementation only as a test oracle.

The snapshot is from f6d7c4b73d7040b1b33860f5a2649c3cd3036cc5. It is not a
runtime fallback or a layer backend. Relative imports intentionally address
the shared exact singleton engine, whose separate bit-flip oracles remain.
"""
import importlib.util
from pathlib import Path
import sys

path=Path(__file__).parent/"snapshots/python_training_layer.py"
spec=importlib.util.spec_from_file_location("conv_rosa_transformer._frozen_python_training_oracle",path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
NBitQKVRosa=module.NBitQKVRosa
joint_rows=module._joint_rows
payload_radius=module._payload_radius
