"""Tests for the budget_output.py module
"""
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import flopy
from mfexport.budget_output import (
    aggregate_sfr_flow_ja_face,
    aggregate_mf6_stress_budget,
    get_stress_budget_textlist,
    read_sfr_output, 
    read_maw_output,
    read_mf6_stress_budget_output
)


@pytest.mark.parametrize('use_flopy', (False, True))
def test_read_mf6_sfr_output(shellmound_model, use_flopy):
    
    model_ws = Path(shellmound_model.model_ws)
    mf6_sfr_stage_file = model_ws / f'{shellmound_model.name}.sfr.stage.bin'
    mf6_sfr_budget_file = model_ws / f'{shellmound_model.name}.sfr.out.bin'

    if use_flopy:
        model = shellmound_model
        package_data_file=None
    else:
        package_data_file = model_ws / f'external/{shellmound_model.name}_packagedata.dat'
        model = None

    results = read_sfr_output(mf6_sfr_stage_file=mf6_sfr_stage_file,
                              mf6_sfr_budget_file=mf6_sfr_budget_file,
                              mf6_package_data=package_data_file,
                              model=model)
    
    # different ways of specifying packagedata
    if use_flopy is not None:
        # from flopy object
        results2 = read_sfr_output(mf6_sfr_stage_file=mf6_sfr_stage_file,
                                mf6_sfr_budget_file=mf6_sfr_budget_file,
                                mf6_package_data=shellmound_model.sfr.packagedata,
                                model=None)
        pd.testing.assert_frame_equal(results, results2)
        # from array
        results2 = read_sfr_output(mf6_sfr_stage_file=mf6_sfr_stage_file,
                                mf6_sfr_budget_file=mf6_sfr_budget_file,
                                mf6_package_data=shellmound_model.sfr.packagedata.array,
                                model=None)
        pd.testing.assert_frame_equal(results, results2)
        # from DataFrame
        results2 = read_sfr_output(mf6_sfr_stage_file=mf6_sfr_stage_file,
                                mf6_sfr_budget_file=mf6_sfr_budget_file,
                                mf6_package_data=pd.DataFrame(shellmound_model.sfr.packagedata.array),
                                model=None)
        pd.testing.assert_frame_equal(results, results2)
    
    # check for missing times
    assert len(results['kstpkper'].unique()) == shellmound_model.simulation.tdis.nper.array
    
    # check reported streambed tops against input
    rd = shellmound_model.sfr.packagedata.array
    strtop = dict(zip(rd['ifno'], rd.rtp))
    expected_tops = [strtop[rno] for rno in results['node']]
    assert np.allclose(expected_tops, results['strtop'])
    
    assert np.allclose(results['depth'].round(2), 
                       np.round(results['stage'] - results['strtop'], 2))
    
    # get the observation results for site 07281600
    # (Tallahatchie River at Money, MS)
    sfr_obsfile = model_ws / 'shellmound.sfr.obs.output.csv'
    obs = pd.read_csv(sfr_obsfile)
    # round off the times so they can be matched by pandas
    obs.index = obs['time'].round(2)
    
    # get reach number for site 07281600
    # (Tallahatchie River at Money, MS)
    ra = shellmound_model.sfr_obs.continuous.data[sfr_obsfile.name]
    # one-based reach number
    rno = int(ra[ra['obsname'] == '07281600-stage']['id'][0])
    # zero-based reach number
    node = rno - 1
    
    # subset results
    zero_based_node = node - 1  # kludge to account for zero-based node originally being written to obs input (which MF interprets as 1-based)
    site_results = results.loc[results.node == zero_based_node].copy()
    # round off the times so they can be matched by pandas
    site_results.index = site_results['time'].round(2)
    site_results['expected_stage'] = obs['07281600-STAGE'].round(3)
    site_results['stage'] = site_results['stage'].round(3)
    
    # stages pulled from SFR binary output should match those in observations
    assert np.allclose(site_results['stage'].values, 
                       site_results['expected_stage'].values)

@pytest.fixture(scope='module', params=['model', None])
def model(request):
    class PackageData:
        def __init__(self):
            self.array = None
    class ConnectionData:
        def __init__(self):
            self.array = None
    class PeriodData:
        def __init__(self):
            self.data = None
    class Model:
        def __init__(self):
            self.maw = mock.Mock(spec_set=flopy.mf6.modflow.ModflowGwfmaw)
            self.maw.packagedata = PackageData()
            self.maw.connectiondata = ConnectionData()
            self.maw.perioddata = PeriodData()
            self.maw.name = ['maw_0']
            self.version = 'mf6'
    if request.param == 'model':
        return Model()


@pytest.mark.parametrize('grid_type', ('structured', 'unstructured'))
def test_read_maw_output(model, grid_type, testdatapath):

    maw_head_file = testdatapath / 'maw/maw.head.bin'
    maw_budget_file = testdatapath / 'maw/maw.out.bin'
    mf6_package_data = testdatapath / 'maw/packagedata.dat'
    mf6_connection_data = testdatapath / 'maw/connectiondata.dat'
    mf6_period_data = {
        0: testdatapath / 'maw/perioddata_000.dat',
        1: testdatapath / 'maw/perioddata_001.dat'
    }
    if model is not None:
        df = pd.read_csv(mf6_package_data, sep='\\s+')
        df.rename(columns={'#wellno': 'ifno'}, inplace=True)
        df['ifno'] -= 1
        model.maw.packagedata.array = df.to_records(index=False)
        df = pd.read_csv(mf6_connection_data, sep='\\s+')
        df.rename(columns={'#wellno': 'ifno'}, inplace=True)
        if grid_type == 'structured':
            df['cellid'] = list(zip(df['k']-1, df['i']-1, df['j']-1))
        else:
            # get the node numbers from the output file
            gwf_output = read_mf6_stress_budget_output(maw_budget_file, text='GWF')
            df['cellid'] = gwf_output.loc[gwf_output['time'] == 1.0, 'node2'].values - 1
        df.drop(['k', 'i', 'j'], axis=1, inplace=True)
        df['ifno'] -= 1
        df['icon'] -= 1
        cols = ['ifno', 'icon', 'cellid', 'scrn_top', 'scrn_botm', 
                'hk_skin', 'radius_skin']
        model.maw.connectiondata.array = df[cols].to_records(index=False)
        perioddata = dict()
        for per, f in mf6_period_data.items():
            df = pd.read_csv(f, skiprows=0, 
                             names=['ifno', 'mawsetting', 'mawsetting_data'],
                             sep='\\s+')
            df = df.loc[df['mawsetting'] != 'status']
            df['ifno'] -= 1
            df['mawsetting_data'] = df['mawsetting_data'].astype(float)
            perioddata[per] = df.to_records(index=False)
        model.maw.perioddata.data = perioddata
        
        mf6_package_data = None
        mf6_connection_data = None
        mf6_period_data = None
    elif grid_type == 'unstructured':
        # add correct node numbers to connectiondata
        gwf_output = read_mf6_stress_budget_output(maw_budget_file, text='GWF')
        cd = pd.read_csv(mf6_connection_data, sep='\\s+')
        cd['cellid'] = gwf_output.loc[gwf_output['time'] == 1.0, 'node2'].values
        cols = ['#wellno', 'icon', 'cellid', 'scrn_top', 'scrn_botm', 'hk_skin', 'radius_skin']
        # write to a new connection data file
        mf6_connection_data = mf6_connection_data.parent / f"{mf6_connection_data.stem}-us.dat"
        cd[cols].to_csv(mf6_connection_data, index=False, sep=' ')
        
    results = read_maw_output(
        maw_head_file, maw_budget_file,
        mf6_package_data=mf6_package_data, 
        mf6_connection_data=mf6_connection_data, 
        mf6_period_data=mf6_period_data,
        model=model, grid_type=grid_type)
    zero_based_columns = [c for c in ['ifno', 'icon', 'k'] 
                          if c in results]
    assert results[zero_based_columns].min(axis=0).sum() == 0
    assert not results.isna().any().any()

    
@pytest.mark.parametrize('mf6_stress_budget_output', (
    ('maw/maw.out.bin'),
    ('sfr/sfr.out.bin')
))
def test_aggregate_mf6_stress_budget(mf6_stress_budget_output, 
                                       testdatapath):
    mf6_stress_budget_output = testdatapath / mf6_stress_budget_output
    df = aggregate_mf6_stress_budget(mf6_stress_budget_output)
    # SFR Package should have one row per reach and time
    if 'sfr' in mf6_stress_budget_output.name:
        assert len(df) == (df['node'].max() + 1) *\
            len(df['kstpkper'].unique())
    # MAW Package will have a row for each connection and time
    elif 'maw' in mf6_stress_budget_output.name:
        assert len(df) == 58 * len(df['kstpkper'].unique())
    
    assert not df['node'].isna().any()
    assert df['node'].min() == 0  # should be zero-based
    gwf_output = read_mf6_stress_budget_output(mf6_stress_budget_output, text='GWF')
    # gwf node numbers should be zero-based
    # (read_mf6_stress_budget_output does not modify 1-based values in MODFLOW output)
    # slice > -1 to remove nodes not connected to the GWF solution
    np.array_equal(df.loc[df['node2'] > -1, 'node2'], gwf_output['node2']-1)
    assert not df['kstpkper'].isna().any()
    
    
@pytest.mark.parametrize('mf6_stress_budget_output, text', (
    ('maw/maw.out.bin', 'GWF'),
    ('sfr/sfr.out.bin', 'FLOW-JA-FACE')
))
def test_read_mf6_stress_budget_output(mf6_stress_budget_output, text, 
                                       testdatapath):
    mf6_stress_budget_output = testdatapath / mf6_stress_budget_output
    textlist = get_stress_budget_textlist(mf6_stress_budget_output)
    results = read_mf6_stress_budget_output(mf6_stress_budget_output, text=text)
    

@pytest.mark.parametrize('sfr_budget_output', (
    'sfr/sfr.out.bin', 
    'shellmound/shellmound.sfr.out.bin'
))
def test_aggregate_sfr_flow_ja_face(sfr_budget_output, testdatapath):
    sfr_budget_output = testdatapath / sfr_budget_output
    results = read_mf6_stress_budget_output(sfr_budget_output, 
                                            text='FLOW-JA-FACE')
    # 'FLOW-JA-FACE' results will not include 
    # isolated reaches without upstream or downstream connections
    # not sure what a good test for this is 
    # without involving connectiondata input
    aggregated = aggregate_sfr_flow_ja_face(results)
    assert aggregated.columns.tolist() ==\
        ['time', 'kstpkper', 'rno', 'Qin', 'Qout', 'Qnet', 'Qmean']
    assert not aggregated.isna().any().any()
    