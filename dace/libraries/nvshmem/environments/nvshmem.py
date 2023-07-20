
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library

# NVSHMEM_HOME = "/home/jbaydamirli21/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/nvshmem-2.7.0-6-svccom42hd6t6fmfru3txongtfpvuynm/"
# MPI_HOME = "/home/jbaydamirli21/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/openmpi-4.1.4-cgf2kyjuumewmbove7jagikdbpo42s6q/"

# NVSHMEM_HOME = "/usr/local/nvshmem/"


@dace.library.environment
class NVSHMEM:
    cmake_minimum_version = "3.11"
    cmake_packages = []
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    # cmake_link_flags = ["-lnvshmem_host -lnvshmem_device", "${MPI_LINKER_FLAGS}"]
    # cmake_link_flags = [f"-L{NVSHMEM_HOME}/lib", f"-L{MPI_HOME}/lib", "-lmpi", "-lnvshmem", "-lnvshmem_host", "-lnvshmem_device", "-lcudart", "-lcudadevrt"]
    cmake_link_flags = []

    # headers = ["mpi.h"]
    # headers = ["mpi.h", 'nvshmem.h', 'nvshmemx.h']
    headers = {
        'frame': ["mpi.h", "nvshmem.h", "nvshmemx.h"],
        'cuda': ["mpi.h", "nvshmem.h", "nvshmemx.h"]
    }

    state_fields = []

    # This might break? There's a check to see if MPI is initialized
    # but I don't have the same for NVSHMEM

    # init_code = ""

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
    # finalize_code = """
    #     //nvshmem_finalize();
    #     //MPI_Finalize();
    # """  # actually if we finalize in the dace program we break pytest :) ok

    finalize_code = ""

    dependencies = []
