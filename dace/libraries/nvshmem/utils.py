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


def is_access_contiguous(memlet, data):
    if memlet.other_subset is not None:
        raise ValueError("Other subset must be None, reshape in send not supported")
    # to be contiguous, in every dimension the memlet range must have the same size
    # than the data, except in the last dim, iff all other dims are only one element

    matching = []
    single = []
    for m, d in zip(memlet.subset.size_exact(), data.sizes()):
        if (str(m) == str(d)):
            matching.append(True)
        else:
            matching.append(False)
        if (m == 1):
            single.append(True)
        else:
            single.append(False)

    # if all dims are matching we are contiguous
    if all(x is True for x in matching):
        return True

    # remove last dim, check if all remaining access a single dim
    matching = matching[:-1]
    single = single[:-1]
    if all(x is True for x in single):
        return True

    return False


def _create_vector_ddt(memlet, data):
    if len(data.shape) != 2:
        raise ValueError("Dimensionality of access not supported atm.")

    ddt = dict()
    ddt["blocklen"] = str(memlet.subset.size_exact()[-1])
    ddt["oldtype"] = "some MPI stuff"
    ddt["count"] = "(" + str(memlet.subset.num_elements_exact()) + ")" + "/" + str(ddt['blocklen'])
    ddt["stride"] = str(data.strides[0])
    return ddt


def create_ddt_if_strided(memlet, data):
    if not is_access_contiguous(memlet, data):
        return _create_vector_ddt(memlet, data)

    # I don't know about this
    ddt = dict()

    ddt["blocklen"] = str(memlet.subset.size_exact()[-1])
    ddt["oldtype"] = "some MPI stuff"
    ddt["count"] = "(" + str(memlet.subset.num_elements_exact()) + ")" + "/" + str(ddt['blocklen'])
    ddt["stride"] = 1

    return ddt
