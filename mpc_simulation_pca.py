import random
import time
from typing import List

# =========================================================
# 1. PARAMETERS & GLOBALS
# =========================================================
P = 4            # Number of parties
N = 500            # Number of columns in A; we assume P divides N (e.g., 6 % 3 == 0)
m = 500         # Number of rows in A (dimension of each vector)
gamma = 18        # Maximum value is 2^gamma - 1
max_val = 2**gamma - 1

# We choose a prime modulus for the field operations.
# (In a real application, pick a large prime ~128 bits or more.)
PRIME = 2_147_483_647  # 31-bit prime

# BGW threshold (for polynomial sharing). We'll use t=1 for demonstration.
t = 1

# Communication delay (seconds):
MSG_PASS_DELAY = 0.001  # one-to-one or one-to-all, done in parallel if broadcast

# =========================================================
# 2. BGW SECRET SHARING
# =========================================================
def share_secret(s: int) -> List[int]:
    """
    Secret-share an integer s by constructing a random polynomial f(x) of degree t
    with f(0)=s, then evaluating at x=1..P. Returns a list of P shares.
    """
    # Construct the polynomial
    poly = [s] + [random.randint(0, PRIME - 1) for _ in range(t)]
    
    # Evaluate polynomial at x=1..P
    shares = []
    for x in range(1, P+1):
        val = 0
        xpow = 1
        for coeff in poly:
            val = (val + coeff*xpow) % PRIME
            xpow = (xpow * x) % PRIME
        shares.append(val)
    return shares

def reconstruct_secret(shares: List[int]) -> int:
    """
    Reconstruct a secret from its P shares using Lagrange interpolation at x=0.
    (Simplified approach for t=1.)
    """
    secret = 0
    for i in range(P):
        x_i = i + 1
        num = 1
        den = 1
        for j in range(P):
            if j == i:
                continue
            x_j = j + 1
            num = (num * (-x_j)) % PRIME
            den = (den * (x_i - x_j)) % PRIME
        inv_den = pow(den, PRIME - 2, PRIME)
        basis_val = (num * inv_den) % PRIME
        secret = (secret + shares[i]*basis_val) % PRIME
    return secret

# =========================================================
# 3. HELPER: SHARE VECTORS/MATRICES
# =========================================================
def share_vector(vec: List[int]) -> List[List[int]]:
    """
    Secret-share each element of a 1D vector.
    Returns a list of length len(vec), each entry is a list of P shares.
    """
    return [share_secret(val) for val in vec]

def share_matrix_2d(mat: List[List[int]]) -> List[List[List[int]]]:
    """
    Secret-share each element of a 2D matrix.
    If mat is shape (R x C), returns shares of shape (R x C x P).
    """
    R = len(mat)
    C = len(mat[0])
    mat_shares = []
    for r in range(R):
        row_shares = []
        for c in range(C):
            sc = share_secret(mat[r][c])
            row_shares.append(sc)
        mat_shares.append(row_shares)
    return mat_shares

def share_matrix_by_columns(A: List[List[int]]) -> List[List[List[int]]]:
    """
    Partition columns of A (m x N) among P parties, each party has N/P columns.
    Each party secret-shares its columns to the others (simulate 0.1s message passing).
    Returns a 3D list A_shares of shape (m x N x P).
    """
    m_local = len(A)
    N_local = len(A[0])
    if N_local % P != 0:
        raise ValueError("P must divide N exactly.")
    
    # Prepare storage
    A_shares = [[ [None]*P for _ in range(N_local)] for _ in range(m_local)]
    
    cols_per_party = N_local // P
    for party_i in range(P):
        start_col = party_i * cols_per_party
        end_col = (party_i + 1)*cols_per_party
        
        # For each column in [start_col, end_col)
        for c in range(start_col, end_col):
            # Extract column c
            col_vec = [A[r][c] for r in range(m_local)]
            # Secret-share each element
            col_shares = share_vector(col_vec)  # shape (m, P)
            # Place them in A_shares
            for r in range(m_local):
                A_shares[r][c] = col_shares[r]
        
        # Simulate the party_i distributing these shares to all other parties
        time.sleep(MSG_PASS_DELAY)
    
    return A_shares  # shape (m, N, P)

# =========================================================
# 4. LOCAL BGW ARITHMETIC (SIMPLIFIED)
# =========================================================
def secure_add(x_shares: List[int], y_shares: List[int]) -> List[int]:
    """Add two secret-shared values (component-wise mod PRIME)."""
    return [(xx + yy) % PRIME for (xx, yy) in zip(x_shares, y_shares)]

def secure_mult(x_shares: List[int], y_shares: List[int]) -> List[int]:
    """
    Multiply two secret-shared values.  (Naive approach: local multiply,
    ignoring the extra degree-reduction step.)
    """
    prod = [(xx * yy) % PRIME for (xx, yy) in zip(x_shares, y_shares)]
    return prod

def public_mult(x_shares: List[int], pub_val: int) -> List[int]:
    """Multiply secret shares by a public integer."""
    return [(xx * pub_val) % PRIME for xx in x_shares]

# =========================================================
# 5. SECURE MATRIX MULTIPLICATION: A^T A => NxN
# =========================================================
def secure_matmul_transpose(A_shares: List[List[List[int]]]) -> List[List[List[int]]]:
    """
    Compute M = A^T A in a secure manner.
    A is shape (m x N x P).
    Then A^T is shape (N x m x P), but we won't literally transpose shares
    (though we can if it helps conceptual clarity).
    
    M will be shape (N x N), each entry secret-shared among P.
    
    M[i, j] = sum_{r=0..m-1} A[r, i] * A[r, j].
    
    So for each i, j, we do a sum over r:
        partial = secure_mult(A[r][i], A[r][j])
        accumulate
    """
    m_local = len(A_shares)       # number of rows in A
    N_local = len(A_shares[0])    # number of columns in A
    
    # We'll build M_shares as NxN, each M_shares[i][j] is a length-P share vector
    M_shares = [ [ [0]*P for _ in range(N_local)] for _ in range(N_local) ]
    
    for i in range(N_local):
        for j in range(N_local):
            accum = [0]*P
            for r in range(m_local):
                # multiply A[r][i] by A[r][j] in secure manner
                prod = secure_mult(A_shares[r][i], A_shares[r][j])
                accum = secure_add(accum, prod)
            M_shares[i][j] = accum
    
    return M_shares  # shape (N x N x P)

# =========================================================
# 6. MAIN SIMULATION
# =========================================================
if __name__ == "__main__":
    # A) Generate random data (excluded from timing)
    A = [[random.randint(0, max_val) for _ in range(N)] for _ in range(m)]
    
    # Each party i has a private NxN matrix v_i
    v_list = []
    for i in range(P):
        # Generate v_i (N x N)
        v_i = []
        for r in range(N):
            row = [random.randint(0, max_val) for _ in range(N)]
            v_i.append(row)
        v_list.append(v_i)
    
    print(f"Parameters => P={P}, N={N}, m={m}, gamma={gamma}")
    
    # -----------------------------------------------------
    # 1) Time the computation of A^T A
    # -----------------------------------------------------
    start_time = time.time()
    
    # Step 1: Partition columns of A among P parties and secret-share
    A_shares = share_matrix_by_columns(A)  # shape (m, N, P)
    
    # Step 2: Compute M = A^T A in a secure manner
    M_shares = secure_matmul_transpose(A_shares)  # shape (N, N, P)
    
    mid_time = time.time()
    time_ATA = mid_time - start_time
    print(f"Time to compute A^T A (including secret sharing A): {time_ATA:.3f} s")
    
    # -----------------------------------------------------
    # 2) Time overhead for adding sum(v_i)
    #    That is, compute final_M = M + sum_{i=1}^P v_i
    # -----------------------------------------------------
    overhead_start = time.time()
    
    # (a) Each party i secret-shares its matrix v_i => shape (N, N, P)
    #     Then we sum them up.
    # We'll accumulate v_sum_shares in shape (N, N, P).
    
    v_sum_shares = [ [ [0]*P for _ in range(N)] for _ in range(N) ]
    
    for i in range(P):
        # share each element of v_i
        v_i_shares = share_matrix_2d(v_list[i])  # shape (N, N, P)
        
        # simulate sending these shares one-to-all
        time.sleep(MSG_PASS_DELAY)
        
        # accumulate
        for r in range(N):
            for c in range(N):
                v_sum_shares[r][c] = secure_add(v_sum_shares[r][c], v_i_shares[r][c])
    
    # (b) final_M_shares = M_shares + v_sum_shares
    final_M_shares = [ [ None for _ in range(N)] for _ in range(N) ]
    for r in range(N):
        for c in range(N):
            final_M_shares[r][c] = secure_add(M_shares[r][c], v_sum_shares[r][c])
    
    overhead_end = time.time()
    overhead_time = overhead_end - overhead_start
    total_time = overhead_end - start_time
    
    print(f"Overhead time for adding sum(v_i): {overhead_time:.3f} s")
    print(f"Total time for A^T A + sum(v_i): {total_time:.3f} s")
    
    # # -----------------------------------------------------
    # # 3) (Optional) Reconstruct final result & verify
    # #    (excluded from timed portion)
    # # -----------------------------------------------------
    # # Reconstruct final_M in cleartext
    # final_M = []
    # for r in range(N):
    #     row = []
    #     for c in range(N):
    #         val = reconstruct_secret(final_M_shares[r][c])
    #         row.append(val)
    #     final_M.append(row)
    
    # # Cleartext check:
    # # 1) Compute A^T A in the clear
    # #    A^T is (N x m), so (A^T A) is NxN
    # ATA_clear = [[0]*N for _ in range(N)]
    # for i in range(N):
    #     for j in range(N):
    #         ssum = 0
    #         for r in range(m):
    #             ssum = (ssum + A[r][i]*A[r][j]) % PRIME
    #         ATA_clear[i][j] = ssum
    
    # # 2) Add sum(v_i)
    # for i in range(P):
    #     for r in range(N):
    #         for c in range(N):
    #             ATA_clear[r][c] = (ATA_clear[r][c] + v_list[i][r][c]) % PRIME
    
    # # 3) Compare a few random entries
    # print("\nVerification on a few random entries:")
    # checks = min(5, N*N)
    # # pick random (r, c) pairs
    # pairs = []
    # for _ in range(checks):
    #     r_ = random.randint(0, N-1)
    #     c_ = random.randint(0, N-1)
    #     pairs.append((r_, c_))
    
    # for (rr, cc) in pairs:
    #     match = (final_M[rr][cc] == ATA_clear[rr][cc])
    #     print(f"  M[{rr}][{cc}] => secure={final_M[rr][cc]}, clear={ATA_clear[rr][cc]}, match={match}")
