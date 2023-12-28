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
class ExpandPutNVSHMEM(ExpandTransformation):
    environments = [environments.nvshmem.NVSHMEM]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        labels = node.validate(parent_sdfg, parent_state)
        dest, value, pe = labels['_dest'], labels['_value'], labels['_pe']

        dtype_dest = dest.dtype

        code = ""

        # in bytes
        code += f"nvshmem_{dtype_dest}_p(_dest, _value, _pe);"
        # code += f"nvshmem_p(_dest, _value, _pe);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)

        return tasklet


@dace.library.node
class Put(NVSHMEMNode):
    # Global properties
    implementations = {
        "NVSHMEM": ExpandPutNVSHMEM,
    }
    default_implementation = "NVSHMEM"

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_value", "_pe"}, outputs={"_dest"}, **kwargs)

    # def validate(self, sdfg, state):
    #     """
    #     :return: dest, value, pe
    #     """
    #
    #     dest, value, pe = None, None, None
    #
    #     for e in state.in_edges(self):
    #         if e.dst_conn == "_value":
    #             source = sdfg.arrays[e.data.data]
    #         if e.dst_conn == "_pe":
    #             pe = sdfg.arrays[e.data.data]
    #
    #     for e in state.out_edges(self):
    #         if e.src_conn == '_dest':
    #             dest = sdfg.arrays[e.data.data]
    #             break
    #     else:
    #         raise ValueError("_dest not found")
    #
    #     if pe.dtype.base_type != dace.dtypes.int32:
    #         raise ValueError("PE must be an integer!")
    #
    #     return dest, value, pe


@oprepo.replaces('dace.libraries.nvshmem.Put')
def _put(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, dest: str, value: Union[str, sp.Expr, Number],
         pe: Union[str, sp.Expr, Number]):
    libnode = Put('_Put_')

    edge_maker = utils.make_edge(libnode=libnode, pv=pv, sdfg=sdfg, state=state,
                                 storage_type=dace.dtypes.StorageType.GPU_NVSHMEM)

    edge = edge_maker(array=dest, var_name="_dest", write=True, pointer=True)
    edge_maker(array=value, var_name="_value", write=False, pointer=False)
    edge_maker(array=pe, var_name="_pe", write=False, pointer=False)
