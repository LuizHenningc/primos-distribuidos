#!/usr/bin/env python3
import math
import sys
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Check if there are at least 2 processes
if size < 2:
    if rank == 0:
        print("This program requires at least 2 processes.")
    comm.Barrier()
    MPI.COMM_WORLD.Abort(1)

# Get N from command line or default to 1000000
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 1000000

# Calculate the block size for each process
block_size = math.ceil((N - 1) / size)
start = 2 + rank * block_size
end = min(start + block_size - 1, N)
segment_size = end - start + 1

# Function to mark multiples in a segment
def mark_multiples(is_prime, start, end, prime):
    first_multiple = start if start % prime == 0 else start + (prime - start % prime)
    for i in range(max(first_multiple, prime * prime), end + 1, prime):
        is_prime[i - start] = False

# Each process sieves its own segment
if segment_size > 0:
    is_prime = [True] * segment_size
    if start <= 2 <= end:
        is_prime[2 - start] = True
    else:
        is_prime[0] = False  # Avoid index errors
    sqrt_N = math.isqrt(N)

    # Initial primes from rank 0
    if rank == 0:
        initial_primes = [2]
        for i in range(3, min(sqrt_N + 1, end + 1), 2):
            if is_prime[i - start]:
                initial_primes.append(i)
                mark_multiples(is_prime, start, end, i)
    else:
        initial_primes = None

    # Broadcast initial primes
    initial_primes = comm.bcast(initial_primes, root=0)

    # Mark multiples of initial primes
    for p in initial_primes:
        mark_multiples(is_prime, start, end, p)

    # Find and share new primes up to sqrt(N)
    local_primes = []
    for i in range(start, end + 1):
        if is_prime[i - start]:
            if i <= sqrt_N:
                print(f"Rank {rank} found prime {i}")
                comm.bcast(i, root=rank)
                mark_multiples(is_prime, start, end, i)
            local_primes.append(i)

    # Gather all local primes
    all_primes = comm.gather(local_primes, root=0)
else:
    all_primes = comm.gather([], root=0)

# On root process, combine and print
if rank == 0:
    primes = sorted([p for sublist in all_primes for p in sublist])
    print(f"Primes up to {N}: {primes}")