#!/usr/bin/env python3
import math
import sys
from mpi4py import MPI

# Inicializa o MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Verifica se há pelo menos 2 processos
if size < 2:
    if rank == 0:
        print("Este programa requer pelo menos 2 processos.")
    comm.Barrier()
    MPI.COMM_WORLD.Abort(1)

# Obtém N a partir da linha de comando ou usa o valor padrão de 1000000
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 1000000

# Calcula o tamanho do bloco para cada processo
block_size = math.ceil((N - 1) / size)
start = 2 + rank * block_size
end = min(start + block_size - 1, N)
segment_size = end - start + 1

# Função para marcar múltiplos em um segmento
def mark_multiples(is_prime, start, end, prime):
    first_multiple = start if start % prime == 0 else start + (prime - start % prime)
    for i in range(max(first_multiple, prime * prime), end + 1, prime):
        is_prime[i - start] = False

# Cada processo realiza o crivo em seu próprio segmento
if segment_size > 0:
    is_prime = [True] * segment_size
    if start <= 2 <= end:
        is_prime[2 - start] = True
    else:
        is_prime[0] = False  # Evita erros de índice
    sqrt_N = math.isqrt(N)

    # Primos iniciais do rank 0
    if rank == 0:
        initial_primes = [2]
        for i in range(3, min(sqrt_N + 1, end + 1), 2):
            if is_prime[i - start]:
                initial_primes.append(i)
                mark_multiples(is_prime, start, end, i)
    else:
        initial_primes = None

    # Transmitindo os primos iniciais
    initial_primes = comm.bcast(initial_primes, root=0)

    # Marca os múltiplos dos primos iniciais
    for p in initial_primes:
        mark_multiples(is_prime, start, end, p)

    # Encontra e compartilha novos primos até sqrt(N)
    local_primes = []
    for i in range(start, end + 1):
        if is_prime[i - start]:
            if i <= sqrt_N:
                print(f"Rank {rank} encontrou o primo {i}")
                comm.bcast(i, root=rank)
                mark_multiples(is_prime, start, end, i)
            local_primes.append(i)

    # Coleta todos os primos locais
    all_primes = comm.gather(local_primes, root=0)
else:
    all_primes = comm.gather([], root=0)

# No processo raiz, combina e imprime
if rank == 0:
    primes = sorted([p for sublist in all_primes for p in sublist])
    print(f"Primos até {N}: {primes}")
