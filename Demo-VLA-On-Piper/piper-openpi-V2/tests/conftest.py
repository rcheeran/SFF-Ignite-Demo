import os
import sys

os.environ.setdefault("SIM_VIEWER", "0")   # no twin window while testing on the desktop

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
