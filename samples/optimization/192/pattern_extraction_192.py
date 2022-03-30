# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace import optimization as optim
from dace.sdfg import infer_types

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hg-fvt-192.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)

    sdfg.apply_gpu_transformations()

    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    dreport = sdfg.get_instrumented_data()
    if dreport is None:
        print("Need to extract")
        arguments = {}
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue

            data = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            arguments[name] = data

        
        optim.CutoutTuner.dry_run(sdfg, **arguments)
        print("Goodbye")
        exit()

    print("We got it")

    tuner = optim.OnTheFlyMapFusionTuner(sdfg, i=192, j=192, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)

    tuner = optim.SubgraphFusionTuner(sdfg, i=192, j=192, measurement=dace.InstrumentationType.GPU_Events)
    tuner.optimize(apply=True)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hg-fvt-192-tuned.sdfg"
    sdfg.save(sdfg_path)
