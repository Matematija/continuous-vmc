import warnings
from jax import random
from jax import numpy as jnp

try:
    from mpi4py import MPI  # type: ignore

    MPI_COMM = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    MPI4PY_COMM = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())

    MPI_N_NODES = MPI_COMM.Get_size()
    MPI_RANK = MPI_COMM.Get_rank()

    import jax

    assert jax.config.omnistaging_enabled, "MPI requires jax omnistaging!"

    import mpi4jax  # type: ignore

except:

    MPI_COMM = None
    MPI4PY_COMM = None
    MPI_N_NODES = 1
    MPI_RANK = 0

    warnings.warn("MPI not found!")

MPI_ACTIVE = MPI_N_NODES > 1

#####################################################
################### Primitives: #####################
#####################################################


def mpi_allreduce_sum(x, *, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return x, token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm, token=token)


def mpi_allreduce_mean(x, *, token=None, comm=MPI_COMM):
    res, token = mpi_allreduce_sum(x, token=token, comm=comm)
    return res / MPI_N_NODES, token


def mpi_allreduce_all(x, *, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return x, token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.allreduce(x, op=MPI.LAND, comm=comm, token=token)


def mpi_allreduce_max(x, *, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return x, token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.allreduce(x, op=MPI.MAX, comm=comm, token=token)


def mpi_reduce_sum(x, *, root=0, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return x, token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.reduce(x, op=MPI.SUM, root=root, comm=comm, token=token)


def mpi_reduce_mean(x, *, root=0, token=None, comm=MPI_COMM):

    res, token = mpi_reduce_sum(x, root=root, token=token, comm=comm)

    if MPI_RANK == root:
        res /= MPI_N_NODES

    return res, token


def mpi_bcast(x, *, root=0, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return x, token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.bcast(x, root=root, token=token, comm=comm)


def mpi_barrier(*, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return token
    else:
        import mpi4jax  # type: ignore

        return mpi4jax.barrier(token=token, comm=comm)


def mpi_scatter_keys(key, *, root=0, token=None, comm=MPI_COMM):

    if not MPI_ACTIVE:
        return key, token
    else:
        import mpi4jax  # type: ignore

        keys = (
            random.split(key, MPI_N_NODES) if MPI_RANK == root else jnp.zeros(2, dtype=jnp.uint32)
        )
        return mpi4jax.scatter(keys, root=root, comm=comm, token=token)


#####################################################


def mpi4py_bcast(x, root=0, comm=MPI4PY_COMM):

    if not MPI_ACTIVE:
        return x
    else:
        return comm.bcast(x, root=root)
