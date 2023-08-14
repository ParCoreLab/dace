from typing import Any, Dict, Optional, Set

import dace
from dace import SDFG, properties
from dace import data as dt
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap
from dace.transformation.interstate import GPUTransformSDFG

from dace.transformation.passes.pattern_matching import enumerate_matches
from dace.sdfg import utils as sdutil


@dace.properties.make_properties
class NVSHMEMArray(ppl.Pass):
    """
    Changes GPU array storage types to NVSHMEM
    """

    CATEGORY: str = 'Storage Consolidation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes
        # return ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        # Not sure about the first two
        return {ap.StateReachability, ap.FindAccessStates, GPUTransformSDFG}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Changes NVSHMEM accessed Access nodes storage to GPU_NVSHMEM

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A set of removed data descriptor names that were modified, or None if nothing changed.
        """
        result: Set[str] = set()

        # Any NVSHMEM node
        pattern = sdutil.node_path_graph(dace.libraries.nvshmem.NVSHMEMNode)

        arrays = set()

        # Get arrays that NVSHMEM touches
        # This might break
        for subgraph in enumerate_matches(sdfg, pattern):
            access_nodes = filter(lambda n: isinstance(n, dace.nodes.AccessNode), subgraph.graph.nodes())
            array_names = map(lambda n: n.data, access_nodes)
            array_descs = map(lambda n: (n, sdfg.arrays[n]), array_names)

            # Filter out scalars
            array_descs = filter(lambda a: not isinstance(a[1], dt.Scalar), array_descs)

            # Push to set
            arrays.update(array_descs)

        # If node is completely removed from graph, erase data descriptor
        for name, array in arrays:
            array.storage = dace.StorageType.GPU_NVSHMEM
            result.add(name)

        return result or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Changed {len(pass_retval)} arrays: {pass_retval}.'
