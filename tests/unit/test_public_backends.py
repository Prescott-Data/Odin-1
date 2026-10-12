"""Public backend modules must work without optional database drivers."""
import subprocess
import sys


def test_backend_contracts_are_importable_without_optional_drivers():
    script = '''
import sys
class RejectDrivers:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'arango', 'gremlin_python'}:
            raise AssertionError('Optional driver imported: ' + fullname)
sys.meta_path.insert(0, RejectDrivers())
from odin.backends import ArangoBackend, ArangoGraphConfig, GraphBackend
from odin import OdinEngine, inspect_schema
import npll
import retrieval
assert ArangoGraphConfig('Nodes', 'Edges', 'predicate').relation_field == 'predicate'
try:
    import retrieval.backends.arango
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Retired backend import remains available')
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
