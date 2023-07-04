# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class NVSHMEM:
    cmake_minimum_version = "3.11"
    cmake_packages = ["MPI", "NVSHMEM"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = ["/home/jbaydamirli21/nvshmem/include"]
    cmake_libraries = ["${MPI_CXX_LIBRARIES}"]
    cmake_compile_flags = ["-I${MPI_CXX_HEADER_DIR}"]
    # cmake_link_flags = ["-lnvshmem_host -lnvshmem_device", "${MPI_LINKER_FLAGS}"]
    cmake_link_flags = ["${MPI_LINKER_FLAGS}", "-L/home/jbaydamirli21/nvshmem/lib/", "-lnvshmem_host -lnvshmem_device"]

    # headers = ["mpi.h"]
    # headers = ["mpi.h", 'nvshmem.h', 'nvshmemx.h']
    headers = {
        'frame': ["mpi.h", "nvshmem.h", "nvshmemx.h"],
        'cuda': ["mpi.h", "nvshmem.h", "nvshmemx.h"]
    }

    state_fields = []

    # This might break? There's a check to see if MPI is initialized
    # but I don't have the same for NVSHMEM
    init_code = """
        int t;
        MPI_Initialized(&t);
        if (!t) {
            MPI_Init(NULL, NULL);
        } else {
            printf("MPI is initialized already\\n");
        }

        int rank, ndevices;

        nvshmemx_init_attr_t attr;
        MPI_Comm comm = MPI_COMM_WORLD;
        attr.mpi_comm = &comm;

        cudaGetDeviceCount(&ndevices);
        cudaSetDevice(rank % ndevices);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    """
    finalize_code = """
        //nvshmem_finalize();
        //MPI_Finalize();
    """  # actually if we finalize in the dace program we break pytest :) ok
    dependencies = []
