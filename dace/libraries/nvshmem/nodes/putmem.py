import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import SDFG, SDFGState
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

        # Might need it later
        # if not node.nosync and buffer.storage == dtypes.StorageType.GPU_Global:
        #     code += f"""
        #     cudaStreamSynchronize(__dace_current_stream);
        #     """

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


@dace.library.node
class Putmem(NVSHMEMNode):
    # Global properties
    implementations = {
        "NVSHMEM": ExpandPutmemNVSHMEM,
    }
    default_implementation = "NVSHMEM"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    nosync = dace.properties.Property(dtype=bool, default=False, desc="Do not sync if memory is on GPU")

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_source", "_pe"}, outputs={"_dest"}, _count_label='_dest', **kwargs)

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

    edge_maker = utils.make_edge(libnode=libnode, pv=pv, sdfg=sdfg, state=state)

    edge_maker(array=dest, var_name="_dest", write=True, pointer=True)
    edge_maker(array=source, var_name="_source", write=False, pointer=True)
    edge_maker(array=pe, var_name="_pe", write=False, pointer=False)
