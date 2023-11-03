import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import SDFG, SDFGState, dtypes
from dace.libraries.nvshmem.nodes.node import NVSHMEMNode
from typing import Union
from numbers import Number
import sympy as sp
from dace.frontend.common import op_repository as oprepo
from .. import utils

ProgramVisitor = 'dace.frontend.python.newast.ProgramVisitor'


@dace.library.expansion
class ExpandSignalWaitUntilNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        code = fr'nvshmem_signal_wait_until(_sig_addr, NVSHMEM_CMP_EQ, _signal);'

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.node
class SignalWaitUntil(NVSHMEMNode):
    # Global properties
    implementations = {
        'signal_wait_until': ExpandSignalWaitUntilNVSHMEM
    }

    # putmem_block needs to be in a cooperative block
    default_implementation = 'signal_wait_until'  # Magically works with 1 GPU

    implementations['pure'] = implementations[default_implementation]

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    nosync = dace.properties.Property(dtype=bool, default=False, desc="Do not sync if memory is on GPU")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={'_sig_addr', '_signal'}, outputs={}, **kwargs)

        # self.schedule = dtypes.ScheduleType.GPU_Default
        self.schedule = dtypes.ScheduleType.GPU_Persistent

    def validate(self, sdfg, state):
        """
        :return: dest, source, count_str, pe
        """
        labels = super().validate(sdfg, state)
        sig_addr, signal = labels['_sig_addr'], labels['_signal']

        utils.check_signal_type(sig_addr[0])

        return sig_addr[0], signal[0]


@oprepo.replaces('dace.libraries.nvshmem.SignalWaitUntil')
# nvshmem_signal_wait_until(gpu_flags + 0, NVSHMEM_CMP_EQ, t);
def _putmem_signal(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, sig_addr: str,
                   signal: Union[str, sp.Expr, Number]):
    libnode = SignalWaitUntil('_Putmem_signal_wait_until_')

    edge_maker = utils.make_edge(libnode=libnode, pv=pv, sdfg=sdfg, state=state,
                                 storage_type=dtypes.StorageType.GPU_NVSHMEM)

    edge_maker(array=sig_addr, var_name='_sig_addr', write=False, pointer=True)
    edge_maker(array=signal, var_name='_signal', write=False, pointer=False)
