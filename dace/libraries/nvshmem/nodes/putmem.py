import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import SDFG, SDFGState, dtypes, Memlet
from dace.libraries.nvshmem.nodes.node import NVSHMEMNode
from typing import Union
from numbers import Number
import sympy as sp
from dace.frontend.common import op_repository as oprepo
from .. import utils

ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'


@dace.library.expansion
class ExpandPutmemNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        dest, _, count_str, _ = node.validate(parent_sdfg, parent_state)

        # nvshmem_putmem_TYPE?
        dtype_dest = dest.dtype.base_type

        if dtype_dest.dtype.veclen > 1:
            raise NotImplementedError

        code = ""

        # in bytes
        nelems = f'({count_str}) * sizeof({dtype_dest})'

        code += f"nvshmem_putmem(_dest, _source, {nelems}, _pe);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.expansion
class ExpandPutmem_blockNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        dest, _, count_str, _ = node.validate(parent_sdfg, parent_state)

        dtype_dest = dest.dtype.base_type

        if dtype_dest.dtype.veclen > 1:
            raise NotImplementedError

        code = ""

        # in bytes
        nelems = f'({count_str}) * sizeof({dtype_dest})'

        code += f"nvshmemx_putmem_block(_dest, _source, {nelems}, _pe);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.expansion
class ExpandPutmemTaskletNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        dest, source, count_str, pe = node.validate(parent_sdfg, parent_state)
        dtype_dest = dest.dtype

        # node.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

        sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))
        state = sdfg.add_state("{l}_state".format(l=node.label))

        sdfg.add_array('_dest', dest.shape, dtype=dest.dtype, strides=dest.strides)
        sdfg.add_array('_source', source.shape, dtype=source.dtype, strides=source.strides)
        sdfg.add_array('_pe', pe.shape, dtype=pe.dtype, strides=pe.strides)

        code = ""

        nelems = f'({count_str}) * sizeof({dtype_dest})'
        code += f"nvshmem_putmem(_dest, _source, {nelems}, _pe);"

        _, me, mx = state.add_mapped_tasklet('_nvshmem_putmem_',
                                             dict(__i="0:1"),
                                             {},
                                             code,
                                             {},
                                             # schedule=dace.ScheduleType.GPU_ThreadBlock,
                                             language=dtypes.Language.CPP,
                                             external_edges=True)

        return sdfg


@dace.library.expansion
class ExpandPutTaskletNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        dest, source, count_str, pe = node.validate(parent_sdfg, parent_state)
        dtype_dest = dest.dtype

        sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))
        state = sdfg.add_state("{l}_state".format(l=node.label))

        sdfg.add_array('_dest', dest.shape, dtype=dtypes.pointer(dest.dtype), strides=dest.strides)
        sdfg.add_array('_source', source.shape, dtype=source.dtype, strides=source.strides)
        sdfg.add_scalar('_pe', pe.dtype, transient=False)

        code = ""

        code += f"nvshmem_{dtype_dest}_p(__dest, __source, __pe);"

        _, me, mx = state.add_mapped_tasklet('_nvshmem_p_',
                                             dict(__i=f'0:{count_str}'),
                                             {
                                                 '__source': Memlet('_source[__i]'),
                                                 '__pe': Memlet.simple('_pe', '0'),
                                             },
                                             code,
                                             {
                                                 '__dest': Memlet('_dest[__i]')
                                             },
                                             language=dtypes.Language.CPP,
                                             external_edges=True)

        return sdfg


@dace.library.node
class Putmem(NVSHMEMNode):
    # Global properties
    implementations = {
        'putmem': ExpandPutmemNVSHMEM,
        'putmem_block': ExpandPutmem_blockNVSHMEM,

        'putmem_tasklet': ExpandPutmemTaskletNVSHMEM,

        'put_tasklet': ExpandPutTaskletNVSHMEM
    }

    # putmem_block needs to be in a cooperative block
    default_implementation = 'put_tasklet'  # Magically works with 1 GPU
    # default_implementation = 'putmem'

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    nosync = dace.properties.Property(dtype=bool, default=False, desc="Do not sync if memory is on GPU")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_source", "_pe"}, outputs={"_dest"}, _count_label='_dest', **kwargs)

        # self.schedule = dtypes.ScheduleType.GPU_Default
        self.schedule = dtypes.ScheduleType.GPU_Persistent

    def validate(self, sdfg, state):
        """
        :return: dest, source, count_str, pe
        """
        labels = super().validate(sdfg, state)
        dest, source, count_str, pe = labels['_dest'], labels['_source'], labels[NVSHMEMNode.count_key], labels['_pe']

        # Not really necessary?
        if dest.dtype != source.dtype:
            raise ValueError("Dest and Source must be the same type")

        return dest, source, count_str, pe


@oprepo.replaces('dace.libraries.nvshmem.Putmem')
def _putmem(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dest: str, source: str, pe: Union[str, sp.Expr, Number]):
    libnode = Putmem('_Putmem_')

    edge_maker = utils.make_edge(libnode=libnode, pv=pv, sdfg=sdfg, state=state,
                                 storage_type=dtypes.StorageType.GPU_NVSHMEM)

    edge_maker(array=dest, var_name="_dest", write=True, pointer=True)
    edge_maker(array=source, var_name="_source", write=False, pointer=True)
    edge_maker(array=pe, var_name="_pe", write=False, pointer=False)
