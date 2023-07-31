# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace.properties
from dace.sdfg import nodes


class NVSHMEMNode(nodes.LibraryNode):
    """
    Abstract class representing an MPI library node.
    """
    count_key = 'count_str'

    def __init__(self, name, *args, _count_label=None, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._count_label = _count_label

    @property
    def needs_count(self) -> bool:
        return self._count_label is not None

    # @property
    # def count_label(self) -> str:
    # First output?
    #     return self._count_label is not None

    @property
    def has_side_effects(self) -> bool:
        return True

    def validate(self, sdfg, state):
        """
        :return: dest, source, count_str, pe
        """
        out = dict()

        in_keys = self.in_connectors.keys()
        out_keys = self.out_connectors.keys()

        for e in state.in_edges(self):
            if e.dst_conn in in_keys:
                out[e.dst_conn] = (sdfg.arrays[e.data.data], e.data)

        for e in state.out_edges(self):
            if e.src_conn in out_keys:
                out[e.src_conn] = (sdfg.arrays[e.data.data], e.data)

        if self.needs_count:
            data = out[self._count_label][1]
            dims = [str(d) for d in data.subset.size_exact()]
            count_str = '*'.join(dims)
            out[NVSHMEMNode.count_key] = count_str

        return {k: v[0] for k, v in out.items()}
