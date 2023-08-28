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
class ExpandPutmemSignalNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        dest, _, count_str, *_ = node.validate(parent_sdfg, parent_state)

        # nvshmem_putmem_TYPE?
        dtype_dest = dest.dtype.base_type

        if dtype_dest.dtype.veclen > 1:
            raise NotImplementedError

        code = ""

        # in bytes
        nelems = f'({count_str}) * sizeof({dtype_dest})'

        code += fr"nvshmem_putmem_signal_nbi(_dest, _source, {nelems}, _sig_addr, _signal, NVSHMEM_SIGNAL_SET, _pe);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.node
class PutmemSignal(NVSHMEMNode):
    # Global properties
    implementations = {
        'putmem_signal': ExpandPutmemSignalNVSHMEM
    }

    # putmem_block needs to be in a cooperative block
    default_implementation = 'putmem_signal'  # Magically works with 1 GPU

    implementations['pure'] = implementations[default_implementation]

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    nosync = dace.properties.Property(dtype=bool, default=False, desc="Do not sync if memory is on GPU")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={'_source', '_pe', '_signal'}, outputs={'_dest', '_sig_addr'},
                         _count_label='_dest', **kwargs)

        # self.schedule = dtypes.ScheduleType.GPU_Default
        self.schedule = dtypes.ScheduleType.GPU_Persistent

    def validate(self, sdfg, state):
        """
        :return: dest, source, count_str, pe
        """
        labels = super().validate(sdfg, state)
        dest, source, count_str, sig_addr, signal, pe = labels['_dest'], labels['_source'], labels[
            NVSHMEMNode.count_key], labels['_sig_addr'], labels['_signal'], labels['_pe']

        # Not really necessary?
        if dest.dtype != source.dtype:
            raise ValueError("Dest and Source must be the same type")

        utils.check_signal_type(sig_addr)

        return dest, source, count_str, sig_addr, signal, pe


@oprepo.replaces('dace.libraries.nvshmem.PutmemSignal')
def _putmem_signal(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dest: str, source: str, sig_addr: str,
                   signal: Union[str, sp.Expr, Number], pe: Union[str, sp.Expr, Number]):
    libnode = PutmemSignal('_Putmem_signal_')

    edge_maker = utils.make_edge(libnode=libnode, pv=pv, sdfg=sdfg, state=state,
                                 storage_type=dtypes.StorageType.GPU_NVSHMEM)

    edge_maker(array=dest, var_name='_dest', write=True, pointer=True)
    edge_maker(array=source, var_name='_source', write=False, pointer=True)
    edge_maker(array=sig_addr, var_name='_sig_addr', write=True, pointer=True)
    edge_maker(array=signal, var_name='_signal', write=False, pointer=False)
    edge_maker(array=pe, var_name='_pe', write=False, pointer=False)
