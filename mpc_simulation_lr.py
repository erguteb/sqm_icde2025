import random
import time
from typing import List

# =====================================================
# 1. PARAMETERS & GLOBALS
# =====================================================
P = 4            # Number of parties
N = 500            # Number of columns in A; we assume P divides N (e.g., 6 % 3 == 0)
m = 500         # Number of rows in A (dimension of each vector)
gamma = 18       # Max input is up to 2^gamma - 1
max_val = 2**gamma - 1

# Prime modulus for arithmetic. (Use a large prime in real scenarios.)
PRIME = 2_147_483_647  # 31-bit prime

# Communication delays:
MSG_PASS_DELAY = 0.01   # one-to-one or one-to-all message passing
BROADCAST_DELAY = 0.01  # specifically for broadcasting w or other public data

# BGW threshold t < P/2 for an honest majority
# We'll set t=1 for this example:
t = 1

# =====================================================
# 2. SECRET SHARING (POLYNOMIAL-BASED)
# =====================================================
def share_secret(s: int) -> List[int]:
    """
    Secret-share an integer s by constructing a random polynomial f(x)
    of degree t with f(0)=s, then evaluating at x=1..P.
    Returns a list of P shares.
    (We do not incorporate the 0.1s delay here; the caller
    should simulate the "send to parties" if needed.)
    """
    # Construct random polynomial of degree t
    poly = [s] + [random.randint(0, PRIME - 1) for _ in range(t)]
    shares = []
    for x in range(1, P+1):
        val = 0
        xpow = 1
        for coeff in poly:
            val = (val + coeff * xpow) % PRIME
            xpow = (xpow * x) % PRIME
        shares.append(val)
    return shares

def reconstruct_secret(shares: List[int]) -> int:
    """
    Reconstruct secret from shares using Lagrange interpolation at x=0.
    Simplified approach for t=1, small P.
    """
    secret = 0
    for i in range(P):
        x_i = i+1
        # Lagrange basis polynomial l_i(0)
        num = 1
        den = 1
        for j in range(P):
            if j == i: 
                continue
            x_j = j+1
            num = (num * (-x_j)) % PRIME
            den = (den * (x_i - x_j)) % PRIME
        inv_den = pow(den, PRIME-2, PRIME)
        l_i_0 = (num * inv_den) % PRIME
        secret = (secret + shares[i] * l_i_0) % PRIME
    return secret

# =====================================================
# 3. SHARING DATA STRUCTURES
# =====================================================
def share_vector(vec: List[int]) -> List[List[int]]:
    """
    Secret-share each element of vec.
    Returns a list of length len(vec) where each entry is a list of length P
    (the shares).
    """
    return [share_secret(x) for x in vec]

def share_matrix_columns(A: List[List[int]]) -> List[List[List[int]]]:
    """
    Partition the columns of A (m x N) among the P parties (P divides N).
    Each party i secret-shares the columns it owns (N/P columns).
    We then combine these shares into a 3D array A_shares of shape (m, N, P).
    
    We'll simulate a one-to-all "sending" of shares from each party,
    incurring MSG_PASS_DELAY once per party.
    """
    m_local = len(A)
    N_local = len(A[0])
    if N_local % P != 0:
        raise ValueError("P must divide N exactly.")
    
    # Prepare storage for the final shares: A_shares[r][c] is a list of P shares
    A_shares = [[None for _ in range(N_local)] for _ in range(m_local)]
    
    cols_per_party = N_local // P
    for party_i in range(P):
        # party_i is responsible for columns [party_i * cols_per_party .. (party_i+1)*cols_per_party - 1]
        start_col = party_i * cols_per_party
        end_col = (party_i + 1) * cols_per_party
        # For each column in that range, secret-share it
        for c in range(start_col, end_col):
            # Extract the c-th column as a vector
            col_vector = [A[r][c] for r in range(m_local)]
            col_shares = share_vector(col_vector)
            # Place in A_shares
            for r in range(m_local):
                A_shares[r][c] = col_shares[r]
        
        # Simulate the party_i sending shares to all other parties
        time.sleep(MSG_PASS_DELAY)
    
    return A_shares  # shape (m, N, P)

# =====================================================
# 4. BASIC BGW ARITHMETIC
# =====================================================
def public_mult(shares: List[int], pub_val: int) -> List[int]:
    """Multiply each share by a public value (mod PRIME)."""
    return [(s * pub_val) % PRIME for s in shares]

def secure_add(shares1: List[int], shares2: List[int]) -> List[int]:
    """Add two secret-shared values (mod PRIME)."""
    return [(x + y) % PRIME for x, y in zip(shares1, shares2)]

def secure_mult(x_shares: List[int], y_shares: List[int]) -> List[int]:
    """
    Multiply two secret-shared values in BGW. 
    (Naive approach for t=1, ignoring extra "degree-reduction" steps.)
    x[p], y[p] -> local multiplication. 
    Then we would do a round to reduce polynomial degree (omitted).
    """
    # Local multiply
    product = [(a * b) % PRIME for (a, b) in zip(x_shares, y_shares)]
    # In real BGW, we'd do another round to reduce degree from 2t to t.
    return product

# =====================================================
# 5. SECURE MATRIX-VECTOR & VECTOR-MATRIX MULTIPLICATION
# =====================================================
def secure_mat_vec_mult(A_shares: List[List[List[int]]],
                        w_public: List[int]) -> List[List[int]]:
    """
    Compute u = A * w, where A_shares is shape (m, N, P),
    and w_public is an N-dimensional vector known by all parties.
    Returns u_shares (length m), where each entry is a list of P shares.
    """
    m_local = len(A_shares)
    N_local = len(A_shares[0])
    u_shares = [[0]*P for _ in range(m_local)]
    
    for r in range(m_local):
        for c in range(N_local):
            # multiply A_shares[r][c] by w[c] (public)
            part_prod = public_mult(A_shares[r][c], w_public[c])
            # add to accumulator
            u_shares[r] = secure_add(u_shares[r], part_prod)
    return u_shares

def secure_vec_mat_mult(u_shares: List[List[int]],
                        A_shares: List[List[List[int]]]) -> List[List[int]]:
    """
    Compute z = u^T * A, where u_shares is shape (m, P),
    A_shares is shape (m, N, P).
    Returns z_shares (length N), each a list of P shares.
    """
    m_local = len(u_shares)
    N_local = len(A_shares[0])
    z_shares = [[0]*P for _ in range(N_local)]
    
    for c in range(N_local):
        for r in range(m_local):
            # multiply u_shares[r] * A_shares[r][c]
            prod = secure_mult(u_shares[r], A_shares[r][c])
            z_shares[c] = secure_add(z_shares[c], prod)
    return z_shares

def secure_compute_AwTA(A_shares: List[List[List[int]]],
                        w_public: List[int]) -> List[List[int]]:
    """
    Compute (A w)^T A in a secure manner.
      - Step 1: u = A * w
      - Step 2: z = u^T * A
    Return z_shares of length N, each a list of P shares.
    """
    u_shares = secure_mat_vec_mult(A_shares, w_public)
    z_shares = secure_vec_mat_mult(u_shares, A_shares)
    return z_shares

# =====================================================
# 6. MAIN SIMULATION
# =====================================================
if __name__ == "__main__":
    # -----------------------------
    # A) Generate random data (excluded from timing)
    # -----------------------------
    A = [[random.randint(0, max_val) for _ in range(N)] for _ in range(m)]
    w = [random.randint(0, max_val) for _ in range(N)]  # public vector w
    # v_i are private vectors of length N for each party
    v_list = []
    for i in range(P):
        v_i = [random.randint(0, max_val) for _ in range(N)]
        v_list.append(v_i)
    
    print(f"Parameters:\n  P={P}, N={N}, m={m}, gamma={gamma}\n")
    
    # -----------------------------
    # B) Time the main MPC steps to compute (Aw)^T A
    # -----------------------------
    start_time = time.time()
    
    # 1) Partition columns of A among parties & secret-share them
    #    (simulate one-to-all sending with MSG_PASS_DELAY once per party)
    A_shares = share_matrix_columns(A)
    
    # 2) Broadcast w (public)
    time.sleep(BROADCAST_DELAY)
    w_public = w[:]  # Everyone has it
    
    # 3) Securely compute z = (A w)^T A
    z_shares = secure_compute_AwTA(A_shares, w_public)
    
    # Done with computing (Aw)^T A
    mid_time = time.time()
    time_AwTA = mid_time - start_time
    
    print(f"Time for computing (A w)^T A (including sharing A, broadcast w): {time_AwTA:.3f} s")
    
    # -----------------------------
    # C) Time overhead for adding sum(v_i)
    #    i.e. compute final_z = z + sum_{i=1}^P v_i
    # -----------------------------
    overhead_start = time.time()
    
    # 1) Each party i secret-shares v_i to all parties
    for i in range(P):
        # share v_i
        _ = share_vector(v_list[i])  # we won't necessarily store it here
        # simulate sending shares one-to-all
        time.sleep(MSG_PASS_DELAY)
    
    # For demonstration, let's actually build the sum of v_i in shares.
    # Each v_i is length N => we do share_vector => shape (N, P)
    # Then add them up component-wise.
    v_sum_shares = [[0]*P for _ in range(N)]
    
    for i in range(P):
        vi_shares = share_vector(v_list[i])
        # we have to re-simulate the 0.1s share again for each i? 
        # Actually let's skip the second delay here for brevity. 
        # The code above has time.sleep for the actual "sending".
        for idx in range(N):
            v_sum_shares[idx] = secure_add(v_sum_shares[idx], vi_shares[idx])
    
    # 2) Now we add v_sum_shares to z_shares
    final_z_shares = []
    for idx in range(N):
        sum_share = secure_add(z_shares[idx], v_sum_shares[idx])
        final_z_shares.append(sum_share)
    
    overhead_end = time.time()
    overhead_time = overhead_end - overhead_start
    total_time = overhead_end - start_time
    
    print(f"Overhead time for adding sum(v_i): {overhead_time:.3f} s")
    print(f"Total time (Aw)^T A + sum(v_i): {total_time:.3f} s")
    
    # # -----------------------------
    # # D) (Optional) Reconstruct final result and verify
    # #     (excluded from the timed portion)
    # # -----------------------------
    # # Reconstruct final_z
    # final_z = [reconstruct_secret(final_z_shares[idx]) for idx in range(N)]
    
    # # Let's do a cleartext check:
    # # 1) u = A * w
    # u_clear = [0]*m
    # for row in range(m):
    #     for col in range(N):
    #         u_clear[row] = (u_clear[row] + A[row][col]*w[col]) % PRIME
    # # 2) z_clear = u^T * A => length N
    # z_clear = [0]*N
    # for col in range(N):
    #     tmp = 0
    #     for row in range(m):
    #         tmp = (tmp + u_clear[row]*A[row][col]) % PRIME
    #     z_clear[col] = tmp
    # # 3) Add sum_{i=1}^P v_i
    # for col in range(N):
    #     for i in range(P):
    #         z_clear[col] = (z_clear[col] + v_list[i][col]) % PRIME
    
    # # Quick consistency check for a few indices
    # print("\nVerification for a few random indices:")
    # indices_to_check = random.sample(range(N), min(5, N))  # up to 5 random indices
    # for idx in indices_to_check:
    #     matches = (final_z[idx] == z_clear[idx])
    #     print(f"  idx={idx}, final_z={final_z[idx]}, clear_z={z_clear[idx]}, match={matches}")
