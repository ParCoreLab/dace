import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.frontend.python.replacements import _define_local_scalar
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import dtypes, Memlet, SDFG, SDFGState
from dace.libraries.nvshmem.nodes.node import NVSHMEMNode
from typing import Union
from numbers import Number
import sympy as sp
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandIsendNVSHMEM(ExpandTransformation):

    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (dest, source, count_str, buffer_offset, ddt), pe = node.validate(parent_sdfg, parent_state)
        mpi_dtype_str = dest.dtype.base_type

        if dest.dtype.veclen > 1:
            raise NotImplementedError

        code = ""

        # if not node.nosync and buffer.storage == dtypes.StorageType.GPU_Global:
        #     code += f"""
        #     cudaStreamSynchronize(__dace_current_stream);
        #     """

        if ddt is not None:
            code += f"""static MPI_Datatype newtype;
                        static int init=1;
                        if (init) {{
                           MPI_Type_vector({ddt['count']}, {ddt['blocklen']}, {ddt['stride']}, {ddt['oldtype']}, &newtype);
                           MPI_Type_commit(&newtype);
                           init=0;
                        }}
                            """
            mpi_dtype_str = "newtype"
            count_str = "1"
        buffer_offset = 0

        # in bytes
        nelems = f'{count_str} * sizeof({mpi_dtype_str})'

        # They shiould both use buffer offset
        # code += f"nvshmem_putmem(_dest, &(_source[{buffer_offset}]), {nelems}, _pe);"
        code += f"nvshmem_putmem(&(_dest[{buffer_offset}]), &(_source[{buffer_offset}]), {nelems}, _pe);"

        if ddt is not None:
            code += f"""// MPI_Type_free(&newtype);
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        conn = tasklet.in_connectors
        conn = {c: (dtypes.int32 if c == '_pe' else t) for c, t in conn.items()}
        tasklet.in_connectors = conn
        conn = tasklet.out_connectors
        # conn = {c: (dtypes.pointer(dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in conn.items()}
        conn = {c: t for c, t in conn.items()}
        tasklet.out_connectors = conn
        return tasklet


@dace.library.node
class Isend(NVSHMEMNode):

    # Global properties
    implementations = {
        "NVSHMEM": ExpandIsendNVSHMEM,
    }
    default_implementation = "NVSHMEM"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    nosync = dace.properties.Property(dtype=bool, default=False, desc="Do not sync if memory is on GPU")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_dest", "_source", "_pe"}, outputs={}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: source, dest, i don;t know honestly, count, mpi_dtype, req of the input data
        """

        dest, source, pe = None, None, None

        for e in state.in_edges(self):
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]
            if e.dst_conn == "_source":
                source = sdfg.arrays[e.data.data]
            if e.dst_conn == "_pe":
                pe = sdfg.arrays[e.data.data]

        # if dest.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("Source must be an integer!")
        if pe.dtype.base_type != dace.dtypes.int32:
            raise ValueError("PE must be an integer!")

        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn in ['_dest', '_source']:
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)
                # compute buffer offset
                minelem = data.subset.min_element()
                dims_data = sdfg.arrays[data.data].strides
                buffer_offsets = []
                for idx, m in enumerate(minelem):
                    buffer_offsets += [(str(m) + "*" + str(dims_data[idx]))]
                buffer_offset = "+".join(buffer_offsets)

                # create a ddt which describes the buffer layout IFF the sent data is not contiguous
                ddt = None
                # if dace.libraries.mpi.utils.is_access_contiguous(data, sdfg.arrays[data.data]):
                #     pass
                # else:
                #     ddt = dace.libraries.mpi.utils.create_vector_ddt(data, sdfg.arrays[data.data])
        return (dest, source, count_str, buffer_offset, ddt), pe


@oprepo.replaces('dace.libraries.nvshmem.Isend')
def _isend(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, dest: str, source: str, pe: Union[str, sp.Expr, Number]):

    libnode = Isend('_Isend_')

    # Handling source and dest together
    dest_range, source_range = None, None
    if isinstance(dest, tuple):
        dest_name, dest_range = dest
        source_name, source_range = source
    else:
        dest_name = dest
        source_name = source

    dest = sdfg.arrays[dest_name]
    dest_node = state.add_read(dest_name)

    source = sdfg.arrays[source_name]
    source_node = state.add_read(source_name)

    iconn = libnode.in_connectors
    iconn = {c: (dtypes.pointer(dest.dtype) if c in ['_dest', '_source'] else t) for c, t in iconn.items()}
    libnode.in_connectors = iconn
    oconn = libnode.out_connectors
    oconn = {c: t for c, t in oconn.items()}
    libnode.out_connectors = oconn

    pe_range = None
    if isinstance(pe, tuple):
        pe_name, pe_range = pe
        pe_node = state.add_read(pe_name)
    elif isinstance(pe, str) and pe in sdfg.arrays.keys():
        pe_name = pe
        pe_node = state.add_read(pe_name)
    # elif isinstance(pe, Number):
    #     pe_name = str(pe)
    #     pe_node = state.add_read(pe_name)
    else:   # ??????
        # pass
        storage = dest.storage
        pe_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        pe_node = state.add_access(pe_name)
        pe_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'}, '__out = {}'.format(pe))
        state.add_edge(pe_tasklet, '__out', pe_node, None, Memlet.simple(pe_name, '0'))

    # tag_range = None
    # if isinstance(tag, tuple):
    #     tag_name, tag_range = tag
    #     tag_node = state.add_read(tag_name)
    # if isinstance(tag, str) and tag in sdfg.arrays.keys():
    #     tag_name = tag
    #     tag_node = state.add_read(tag)
    # else:
    #     storage = desc.storage
    #     tag_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
    #     tag_node = state.add_access(tag_name)
    #     tag_tasklet = state.add_tasklet('_set_tag_', {}, {'__out'}, '__out = {}'.format(tag))
    #     state.add_edge(tag_tasklet, '__out', tag_node, None, Memlet.simple(tag_name, '0'))

    if dest_range:
        dest_mem = Memlet.simple(dest_name, dest_range)
    else:
        dest_mem = Memlet.from_array(dest_name, dest)
    if source_range:
        source_mem = Memlet.simple(source_name, source_range)
    else:
        source_mem = Memlet.from_array(source_name, source)
    if pe_range:
        pe_mem = Memlet.simple(pe_name, pe_range)
    else:
        pe_mem = Memlet.simple(pe_name, '0')
    # if tag_range:
    #     tag_mem = Memlet.simple(tag_name, tag_range)
    # else:
    #     tag_mem = Memlet.simple(tag_name, '0')

    state.add_edge(dest_node, None, libnode, '_dest', dest_mem)
    state.add_edge(source_node, None, libnode, '_source', source_mem)
    state.add_edge(pe_node, None, libnode, '_pe', pe_mem)
    # state.add_edge(tag_node, None, libnode, '_tag', tag_mem)

    return None
