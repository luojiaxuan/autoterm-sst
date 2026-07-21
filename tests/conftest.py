import os
import sys

# The suites import `framework.*` as top-level modules; make that work no
# matter which directory pytest/unittest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
