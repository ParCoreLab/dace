# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.nvshmem.nodes.node import NVSHMEMNode

import sympy as sp

from numbers import Number
from typing import Union

import dace
from dace import dtypes
from dace.frontend.common import op_repository as oprepo
from dace.memlet import Memlet

from dace.frontend.python.replacements import _define_local_scalar

from dace.sdfg import SDFG, SDFGState

ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'


@dace.library.expansion
class ExpandSendNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str, buffer_offset, ddt), dest = node.validate(parent_sdfg, parent_state)
        # mpi_dtype_str = dace.libraries.nvshmem.utils.MPI_DDT(buffer.dtype.base_type)
        mpi_dtype_str = str(buffer.dtype.base_type)

        if buffer.dtype.veclen > 1:
            raise NotImplementedError

        code = ""

        if ddt is not None:
            raise ValueError('Some sort of DDT thing')
            # code = f"""static MPI_Datatype newtype;
            #             static int init=1;
            #             if (init) {{
            #                MPI_Type_vector({ddt['count']}, {ddt['blocklen']}, {ddt['stride']}, {ddt['oldtype']}, &newtype);
            #                MPI_Type_commit(&newtype);
            #                init=0;
            #             }}
            #                 """
            # mpi_dtype_str = "newtype"
            # count_str = "1"

        buffer_offset = 0
        # code += f"""
        #         MPI_Send(&(_buffer[{buffer_offset}]), {count_str}, {mpi_dtype_str}, _dest, _tag, MPI_COMM_WORLD);
        #         """

        # or embed this in nvshmem_putSIZE
        size_str = f'{count_str} * sizeof({mpi_dtype_str})'

        code += f"""
                // nvshmem_putmem(&(_buffer[{buffer_offset}]), &(_buffer[{buffer_offset}]), {size_str}, _dest);
                // nvshmem_putmem();
                int mype = nvshmem_my_pe();
                // printf("%d\\n", mype);
                """

        if ddt is not None:
            raise ValueError('Some sort of DDT thing')
            # code += f"""// MPI_Type_free(&newtype);
            # """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class Send(NVSHMEMNode):
    # Global properties
    implementations = {
        "NVSHMEM": ExpandSendNVSHMEM,
    }
    default_implementation = "NVSHMEM"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_buffer", "_dest"}, outputs={}, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: buffer, count, mpi_dtype, req of the input data
        """


        buffer, dest = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                buffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_dest":
                dest = sdfg.arrays[e.data.data]

        if dest.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Source must be an integer!")

        count_str = "XXX"
        for _, _, _, dst_conn, data in state.in_edges(self):
            if dst_conn == '_buffer':
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
                # if dace.libraries.nvshmem.utils.is_access_contiguous(data, sdfg.arrays[data.data]):
                #     pass
                # else:
                #     ddt = dace.libraries.nvshmem.utils.create_vector_ddt(data, sdfg.arrays[data.data])

        return (buffer, count_str, buffer_offset, ddt), dest


@oprepo.replaces('dace.libraries.nvshmem.Send')
def _send(pv: ProgramVisitor,
          sdfg: SDFG,
          state: SDFGState,
          dest_buffer: str,
          # source_buffer: str,
          dst: Union[str, sp.Expr, Number]):
    libnode = Send('_Send_')

    buf_range = None
    if isinstance(dest_buffer, tuple):
        buf_name, buf_range = dest_buffer
    else:
        buf_name = dest_buffer
    desc = sdfg.arrays[buf_name]
    buf_node = state.add_write(buf_name)

    desc.storage = dtypes.StorageType.GPU_NVSHMEM

    conn = libnode.in_connectors
    conn = {c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t) for c, t in conn.items()}
    libnode.in_connectors = conn

    dst_range = None
    if isinstance(dst, tuple):
        dst_name, dst_range = dst
        dst_node = state.add_read(dst_name)
    elif isinstance(dst, str) and dst in sdfg.arrays.keys():
        dst_name = dst
        dst_node = state.add_read(dst_name)
    else:
        storage = desc.storage
        dst_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
        dst_node = state.add_access(dst_name)
        dst_tasklet = state.add_tasklet('_set_dst_', {}, {'__out'}, '__out = {}'.format(dst))
        state.add_edge(dst_tasklet, '__out', dst_node, None, Memlet.simple(dst_name, '0'))

    if buf_range:
        buf_mem = Memlet.simple(buf_name, buf_range)
    else:
        buf_mem = Memlet.from_array(buf_name, desc)
    if dst_range:
        dst_mem = Memlet.simple(dst_name, dst_range)
    else:
        dst_mem = Memlet.simple(dst_name, '0')

    state.add_edge(buf_node, None, libnode, '_buffer', buf_mem)
    state.add_edge(dst_node, None, libnode, '_dest', dst_mem)

    return None
