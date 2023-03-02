# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion
from importlib import import_module
import numpy as np
from numbers import Number
from numpy import f2py
import os
import pytest
import sys
import tempfile
from typing import Dict, Union


def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
        source = file.read()
    assert source
    return source


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        f2py.compile(source, modulename=program_name, verbose=True, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name)
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)
    
    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    if normalize_offsets:
        my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    else:
        my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    if normalize_offsets:
        utils.normalize_offsets(sdfg)

    return sdfg


parameters = {
    'KLON': 10,
    'KLEV': 10,
    'KIDIA': 2,
    'KFDIA': 8,
    'NCLV': 10,
    'NCLDQI': 3,
    'NCLDQL': 4,
    'NCLDQR': 5,
    'NCLDQS': 6,
    'NCLDQV': 7,
    'NCLDTOP': 1,
    'NSSOPT': 1,
}


data = {
    'PTSPHY': (0,),
    'R2ES': (0,),
    'R3IES': (0,),
    'R3LES': (0,),
    'R4IES': (0,),
    'R4LES': (0,),
    'RALSDCP': (0,),
    'RALVDCP': (0,),
    'RAMIN': (0,),
    'RKOOP1': (0,),
    'RKOOP2': (0,),
    'RKOOPTAU': (0,),
    'RLMIN': (0,),
    'RLSTT': (0,),
    'RLVTT': (0,),
    'RTHOMO': (0,),
    'RTT': (0,),
    'ZEPSEC': (0,),
    'ZEPSILON': (0,),
    'ZRG_R': (0,),
    'ZQTMST': (0,),
    'IPHASE': (parameters['NCLV'],),
    'PAPH': (parameters['KLON'], parameters['KLEV']+1),
    'PCOVPTOT': (parameters['KLON'], parameters['KLEV']),
    'PFCQLNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQNNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQRNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQSNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQIF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQITUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLTUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQRF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQSF': (parameters['KLON'], parameters['KLEV']+1),
    'PLUDE': (parameters['KLON'], parameters['KLEV']),
    'PSUPSAT': (parameters['KLON'], parameters['KLEV']),
    'PVFI': (parameters['KLON'], parameters['KLEV']),
    'PVFL': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_a': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'tendency_loc_q': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_T': (parameters['KLON'], parameters['KLEV']),
    'ZA': (parameters['KLON'], parameters['KLEV']),
    'ZCONVSINK': (parameters['KLON'], parameters['NCLV']),
    'ZCONVSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZCORQSICE': (parameters['KLON']),
    'ZCOVPTOT': (parameters['KLON'],),
    'ZDA': (parameters['KLON']),
    'ZFALLSINK': (parameters['KLON'], parameters['NCLV']),
    'ZFALLSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZFLUXQ': (parameters['KLON'], parameters['NCLV']),
    'ZFOEALFA': (parameters['KLON'], parameters['KLEV']+1),
    'ZICEFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLI': (parameters['KLON'], parameters['KLEV']),
    'ZLIQFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLNEG': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZPFPLSX': (parameters['KLON'], parameters['KLEV']+1, parameters['NCLV']),
    'ZPSUPSATSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZSOLQA': (parameters['KLON'], parameters['NCLV'], parameters['NCLV']),
    'ZQSICE': (parameters['KLON'], parameters['KLEV']),
    'ZQX': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQX0': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQXFG': (parameters['KLON'], parameters['NCLV']),
    'ZQXN': (parameters['KLON'], parameters['NCLV']),
    'ZQXN2D': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZTP1': (parameters['KLON'], parameters['KLEV']),
}


programs = {
    'cloudsc_1f': 'liq_ice_fractions',
    'cloudsc_3p1': 'ice_supersaturation_adjustment',
    'cloudsc_6': 'update_tendancies',
    'cloudsc_8': 'flux_diagnostics',
    'cloudsc_8a': 'copy_precipitation_arrays',
    'cloudsc_8b': 'fluxes',
    'cloudsc_8c': 'enthalpy_flux_due_to_precipitation',
}


program_parameters = {
    'cloudsc_1f': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDQI', 'NCLDQL'),
    'cloudsc_3p1': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDTOP', 'NCLDQI', 'NCLDQL', 'NCLDQV', 'NSSOPT'),
    'cloudsc_6': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDQV', 'NCLDTOP'),
    'cloudsc_8': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDQL', 'NCLDQI', 'NCLDQR', 'NCLDQS'),
    'cloudsc_8a': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDQL', 'NCLDQI', 'NCLDQR', 'NCLDQS'),
    'cloudsc_8b': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDQL', 'NCLDQI', 'NCLDQR', 'NCLDQS'),
    'cloudsc_8c': ('KLON', 'KLEV', 'KIDIA', 'KFDIA')
}


program_inputs = {
    'cloudsc_1f': ('RLMIN', 'ZQX',),
    'cloudsc_3p1': ('PTSPHY', 'RAMIN', 'RKOOP1', 'RKOOP2', 'RKOOPTAU', 'R2ES', 'R3IES', 'R3LES', 'R4IES', 'R4LES',
                    'RTHOMO', 'RTT', 'ZEPSEC', 'ZEPSILON', 'PSUPSAT', 'ZA', 'ZCORQSICE', 'ZQSICE', 'ZQX', 'ZTP1'),
    'cloudsc_6': ('RALSDCP', 'RALVDCP', 'ZQTMST', 'ZQX', 'ZQX0', 'ZDA', 'ZPSUPSATSRCE', 'ZCONVSRCE',
                  'ZFALLSINK', 'ZFALLSRCE', 'ZCONVSINK', 'ZQXN', 'IPHASE', 'ZCOVPTOT'),
    'cloudsc_8': ('RLSTT', 'RLVTT', 'ZPFPLSX',
                  'PAPH', 'ZFOEALFA', 'PVFL', 'PVFI', 'PLUDE', 'ZQX0', 'ZLNEG', 'ZQXN2D', 'ZRG_R', 'ZQTMST', 'PTSPHY'),
    'cloudsc_8a': ('ZPFPLSX',),
    'cloudsc_8b': ('PAPH', 'ZFOEALFA', 'PVFL', 'PVFI', 'PLUDE', 'ZQX0', 'ZLNEG', 'ZQXN2D', 'ZRG_R', 'ZQTMST', 'PTSPHY'),
    'cloudsc_8c': ('RLSTT', 'RLVTT', 'PFPLSL', 'PFPLSN')
}


program_outputs = {
    'cloudsc_1f': ('ZA', 'ZLIQFRAC', 'ZICEFRAC'),
    'cloudsc_3p1': ('ZSOLQA', 'ZQXFG'),
    'cloudsc_6': ('ZFLUXQ', 'tendency_loc_T', 'tendency_loc_cld', 'tendency_loc_q', 'tendency_loc_a', 'PCOVPTOT'),
    'cloudsc_8': ('PFPLSL', 'PFPLSN', 'PFHPSL', 'PFHPSN',
                  'PFSQLF', 'PFSQIF', 'PFCQNNG', 'PFCQLNG', 'PFSQRF', 'PFSQSF', 'PFCQRNG', 'PFCQSNG', 'PFSQLTUR', 'PFSQITUR'),
    'cloudsc_8a': ('PFPLSL', 'PFPLSN'),
    'cloudsc_8b': ('PFSQLF', 'PFSQIF', 'PFCQNNG', 'PFCQLNG', 'PFSQRF', 'PFSQSF', 'PFCQRNG', 'PFCQSNG', 'PFSQLTUR', 'PFSQITUR'),
    'cloudsc_8c': ('PFHPSL', 'PFHPSN')
}


def get_inputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    inp_data = dict()
    for p in program_parameters[program]:
        inp_data[p] = parameters[p]
    for inp in program_inputs[program]:
        shape = data[inp]
        if shape == (0,):  # Scalar
            inp_data[inp] = rng.random()
        else:
            inp_data[inp] = np.asfortranarray(rng.random(shape))
    return inp_data


def get_outputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    out_data = dict()
    for out in program_outputs[program]:
        shape = data[out]
        if shape == (0,):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(rng.random(shape))
    return out_data


@pytest.mark.parametrize("program, device", "normalize_offsets", [
    pytest.param('cloudsc_1f', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_1f', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_1f', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_1f', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_3p1', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_3p1', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_3p1', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_3p1', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_6', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_6', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_6', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_6', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8a', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_8a', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_8a', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8a', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8b', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_8b', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_8b', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8b', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8c', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_8c', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_8c', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8c', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8', dace.DeviceType.CPU, False),
    pytest.param('cloudsc_8', dace.DeviceType.CPU, True),
    pytest.param('cloudsc_8', dace.DeviceType.GPU, False, marks=pytest.mark.gpu),
    pytest.param('cloudsc_8', dace.DeviceType.GPU, True, marks=pytest.mark.gpu),
])
def test_program(program: str, device: dace.DeviceType, normalize_memlets: bool):

    fsource = read_source(program)
    program_name = programs[program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs_f = get_outputs(program, rng)
    outputs_d = copy.deepcopy(outputs_f)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_f[k]
        assert np.allclose(farr, darr)


if __name__ == "__main__":
    test_program('cloudsc_1f', dace.DeviceType.CPU, False)
    test_program('cloudsc_1f', dace.DeviceType.CPU, True)
    test_program('cloudsc_3p1', dace.DeviceType.CPU, False)
    test_program('cloudsc_3p1', dace.DeviceType.CPU, True)
    test_program('cloudsc_6', dace.DeviceType.CPU, False)
    test_program('cloudsc_6', dace.DeviceType.CPU, True)
    test_program('cloudsc_8a', dace.DeviceType.CPU, False)
    test_program('cloudsc_8a', dace.DeviceType.CPU, True)
    test_program('cloudsc_8b', dace.DeviceType.CPU, False)
    test_program('cloudsc_8b', dace.DeviceType.CPU, True)
    test_program('cloudsc_8c', dace.DeviceType.CPU, False)
    test_program('cloudsc_8c', dace.DeviceType.CPU, True)
    test_program('cloudsc_8', dace.DeviceType.CPU, False)
    test_program('cloudsc_8', dace.DeviceType.CPU, True)