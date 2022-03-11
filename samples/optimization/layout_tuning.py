# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.optimization as optim
import numpy as np

from dace.transformation.auto.auto_optimize import auto_optimize

from dace.optimization import data_layout_tuner as dt
from dace.optimization import cutout_tuner as ct

N = 200


@dace.program
def layout_sample(A, B):
    B[:] = np.sum(A, axis=0)


if __name__ == '__main__':
    A = np.random.rand(N, N, N)
    B = np.random.rand(N, N)

    sdfg = layout_sample.to_sdfg(A, B)
    auto_optimize(sdfg, dace.DeviceType.CPU)

    ct.CutoutTuner.dry_run(sdfg, A, B)
    
    tuner = optim.DataLayoutTuner(sdfg)
    report = tuner.optimize(group_by=dt.TuningGroups.Inputs_Outputs)

    print(report)
