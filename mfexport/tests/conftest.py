import os
from pathlib import Path
import platform
import shutil
import pytest
import flopy.modflow as fm
import flopy.mf6 as mf6
from ..grid import load_modelgrid


@pytest.fixture(scope="session")
def project_root_path():
    """Root folder for the project (with setup.py)"""
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return Path(os.path.normpath(os.path.join(filepath, '../../')))


@pytest.fixture(scope="session", autouse=True)
def test_output_folder(project_root_path):
    """(Re)make an output folder for the tests
    at the begining of each test session."""
    folder = project_root_path / 'mfexport/tests/tmp'
    reset = True
    if reset:
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    return folder


@pytest.fixture(scope="session")
def testdatapath():
    """Smaller datasets for faster test execution."""
    return Path('mfexport/tests/data')


@pytest.fixture(scope='module')
def lpr_output_path(test_output_folder):
    return test_output_folder / 'lpr'


@pytest.fixture(scope='module')
def shellmound_output_path(test_output_folder):
    return test_output_folder / 'shellmound'


@pytest.fixture(scope='module')
def lpr_model():
    namefile = 'lpr_inset.nam'
    model_ws = 'Examples/data/lpr/'
    m = fm.Modflow.load(namefile,
                        model_ws=model_ws, check=False, forgive=False)
    return m


@pytest.fixture(scope='module')
def lpr_modelgrid():
    grid_file = 'Examples/data/lpr/lpr_grid.json'
    return load_modelgrid(grid_file)


@pytest.fixture(scope='module')
def shellmound_simulation(testdatapath):
    sim = mf6.MFSimulation.load('mfsim', 'mf6', 'mf6', sim_ws='{}/shellmound'.format(testdatapath))
    return sim


@pytest.fixture(scope='module')
def shellmound_model(shellmound_simulation):
    return shellmound_simulation.get_model('shellmound')


@pytest.fixture(scope='module')
def shellmound_modelgrid():
    grid_file = 'mfexport/tests/data/shellmound/shellmound_grid.json'
    return load_modelgrid(grid_file)


@pytest.fixture(scope='module')
def shellmound(shellmound_model, shellmound_modelgrid, shellmound_output_path):
    return shellmound_model, shellmound_modelgrid, shellmound_output_path


@pytest.fixture(scope='module')
def lpr(lpr_model, lpr_modelgrid, lpr_output_path):
    return lpr_model, lpr_modelgrid, lpr_output_path


# ugly work-around for fixtures not being supported as test parameters yet
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['shellmound',
                        'lpr'])
def model(request,
          shellmound,
          lpr):
    return {'shellmound': shellmound,
            'lpr': lpr}[request.param]


@pytest.fixture(scope="session")
def bin_path(project_root_path):
    bin_path = project_root_path / "bin"
    platform_info = platform.platform().lower()
    if "linux" in platform_info:
        bin_path = bin_path / "linux"
    elif "mac" in platform_info or "darwin" in platform_info:
        bin_path = bin_path / "mac"
    else:
        bin_path = bin_path / "win"
    return bin_path


@pytest.fixture(scope="session")
def mf6_exe(bin_path):
    version = bin_path.name
    exe_name = 'zbud6'
    if version == "win":
        exe_name += '.exe'
    return bin_path / exe_name