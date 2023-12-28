import dace.library
from dace.libraries.mpi import MPI


@dace.library.environment
class NVSHMEM:
    cmake_minimum_version = "3.11"
    cmake_packages = ["NVSHMEM"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["nvshmem::nvshmem"]
    cmake_compile_flags = []
    cmake_link_flags = []

    headers = {
        "frame": ["mpi.h", "nvshmem.h", "nvshmemx.h"],
        "cuda": ["mpi.h", "nvshmem.h", "nvshmemx.h"]
    }

    state_fields = []

    # TODO: MPI initialization probably not necessary
    init_code = """
        int t;
        MPI_Initialized(&t);
        if (!t) {
            MPI_Init(NULL, NULL);
        } else {
            printf("MPI is initialized already\\n");
        }

        nvshmemx_init_attr_t attr;
        MPI_Comm comm = MPI_COMM_WORLD;
        attr.mpi_comm = &comm;

        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    """
    finalize_code = """
        // nvshmem_finalize();
        // MPI_Finalize();
    """

    dependencies = [MPI]
