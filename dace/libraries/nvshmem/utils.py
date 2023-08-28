import dace
from dace import SDFG, SDFGState, dtypes, Memlet
from dace.frontend.python.replacements import _define_local_scalar

from functools import partial

NVSHMEM_SIGNAL_TYPE = dace.dtypes.uint64


def _add_edge(libnode, pv, sdfg: SDFG, state: SDFGState, storage_type: dtypes.StorageType,
              array: str, var_name: str, write=False, pointer=True):
    array_range, array_node = None, None

    # Not clean
    if isinstance(array, tuple):
        array_name, array_range = array
        array = sdfg.arrays[array_name]
        # array.storage = storage_type
        array_mem = Memlet.simple(array_name, array_range)
    elif isinstance(array, str):
        array_name = array
        array = sdfg.arrays[array_name]
        # array.storage = storage_type
        array_mem = Memlet.from_array(array_name, array)
    else:
        storage = dtypes.StorageType.Default
        array_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        array_node = state.add_access(array_name)
        array_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'}, '__out = {}'.format(array))
        state.add_edge(array_tasklet, '__out', array_node, None, Memlet.simple(array_name, '0'))
        array_mem = Memlet.simple(array_name, '0')

    # Not very clean
    if array_node is None:
        array_node = (state.add_write if write else state.add_read)(array_name)

    conn = libnode.out_connectors if write else libnode.in_connectors

    if pointer:
        conn[var_name] = dtypes.pointer(array.dtype)

    if write:
        return state.add_edge(libnode, var_name, array_node, None, array_mem)
    else:
        return state.add_edge(array_node, None, libnode, var_name, array_mem)


def make_edge(libnode, pv, sdfg: SDFG, state: SDFGState,
              storage_type: dtypes.StorageType = dtypes.StorageType.GPU_Global):
    return partial(_add_edge, libnode=libnode, pv=pv, sdfg=sdfg, state=state, storage_type=storage_type)


def check_signal_type(sig):
    if sig.dtype != NVSHMEM_SIGNAL_TYPE:
        raise ValueError('NVSHMEM Signals must be uint64_t')
