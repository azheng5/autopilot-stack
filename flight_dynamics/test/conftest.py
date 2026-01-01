import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics.Spacecraft import Spacecraft

@pytest.fixture
def default_spacecraft():
    """Common default spacecraft configuration"""
    id = '01'
    wet_mass = 200
    dry_mass = 50
    Isp = 3000
    Cd = 1
    A_ref = 1
    return Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)