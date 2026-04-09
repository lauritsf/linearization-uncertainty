# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language=c++

cimport numpy as np
cimport cython
import numpy as np
np.import_array()

from libcpp.algorithm cimport sort

ctypedef np.int64_t int64_t
ctypedef np.uint32_t uint32_t
ctypedef np.uint8_t uint8_t

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF

# --- RANDOM UTILS ---

cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)
    return seed[0] % (<uint32_t>RAND_R_MAX + 1)

cdef inline uint32_t rand_int(uint32_t end, uint32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef int sort_array(int64_t[::1] arr, int size) nogil:
    sort(&arr[0], (&arr[0]) + size)

# --- SCIENTIFIC SELECTION LOGIC ---

cdef inline int get_degree(int[:] indptr, int node) nogil:
    return indptr[node + 1] - indptr[node]

cdef int select_candidate(
    int[:] candidates, 
    int num_candidates, 
    int[:] indptr, 
    int bias, 
    uint32_t* rand_r_state,
    bint deterministic
) nogil:
    """
    Selects a node from candidates based on bias and tie-breaking rules.
    bias 0: random_order
    bias 1: max_degree_first
    bias 2: min_degree_first
    """
    # 0. random_order
    if num_candidates <= 0:
        return -1
    if bias == 0 or num_candidates == 1:
        return candidates[rand_int(<uint32_t> num_candidates, rand_r_state)]
    
    cdef int i, node, current_val
    cdef int best_node = candidates[0]
    cdef int best_val = get_degree(indptr, best_node)
    cdef int tie_count = 1
    
    # Scan candidates for best degree
    for i in range(1, num_candidates):
        node = candidates[i]
        current_val = get_degree(indptr, node)
        
        # Priority Check (max_degree_first or min_degree_first)
        if (bias == 1 and current_val > best_val) or (bias == 2 and current_val < best_val):
            best_val = current_val
            best_node = node
            tie_count = 1
            
        elif current_val == best_val:
            if deterministic:
                # Canonical tie-breaking: prioritize lower node ID
                if node < best_node:
                    best_node = node
            else:
                # Stochastic tie-breaking: reservoir sampling
                tie_count += 1
                if rand_int(<uint32_t> tie_count, rand_r_state) == 0:
                    best_node = node
                
    return best_node

# ==============================================================================
# 1. UNLABELED SAMPLER
# ==============================================================================

def sample_sent(
    csr_matrix,
    int seq_length=-1,
    int idx_offset=0,
    int reset_idx=-1,
    int left_bracket=-2,
    int right_bracket=-3,
    bint undirected=True,
    object rng=None,
    int start_bias=0,
    int jump_bias=0,
    int neighbor_bias=0,
    bint deterministic=False
):
    """Sample a SENT from an unattributed graph with descriptive bias strategies."""
    if rng is None:
        rng = np.random.RandomState(0)

    cdef:
        int[:] indices = csr_matrix.indices
        int[:] indptr = csr_matrix.indptr
        int num_nodes = csr_matrix.shape[0]
        int num_edges = indices.shape[0]

    if seq_length < 0:
        seq_length = 20 if num_nodes <= 1 else (num_edges + num_nodes) * 2
            
    cdef:
        uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
        uint32_t* rand_r_state = &rand_r_state_seed

        int num_unvisited = num_nodes
        uint8_t[:] unvisited = np.ones(num_nodes, dtype=np.uint8)

        int64_t[:] sent_seq = np.full(seq_length, reset_idx, dtype=np.int64)
        int64_t[:] node_index_map = np.full(num_nodes, -1, dtype=np.int64)

        int sent_seq_idx = 0
        int64_t node_index = 0
        int current_node, prev_node, i, k, neighbor
        int neighbors_num, num_candidates, neighborhood_set_size

        int[:] neighbors = np.empty(num_nodes, dtype=np.int32)
        int64_t[::1] neighborhood_set = np.empty(num_nodes, dtype=np.int64)
        int[:] candidates = np.empty(num_nodes, dtype=np.int32)

    if num_nodes <= 0:
        return np.asarray(sent_seq[:0]), np.asarray(node_index_map)

    with nogil:
        # --- INITIALIZATION ---
        num_candidates = 0
        for i in range(num_nodes):
            candidates[num_candidates] = i
            num_candidates += 1
        current_node = select_candidate(candidates, num_candidates, indptr, start_bias, rand_r_state, deterministic)
        
        if current_node < 0 or current_node >= num_nodes:
            with gil:
                return np.asarray(sent_seq[:0]), np.asarray(node_index_map)

        node_index_map[current_node] = node_index
        sent_seq[sent_seq_idx] = node_index + idx_offset
        node_index += 1
        sent_seq_idx += 1
        unvisited[current_node] = False
        num_unvisited -= 1
        
        while num_unvisited > 0 and sent_seq_idx < seq_length:
            prev_node = current_node
            
            # Collect unvisited neighbors
            neighbors_num = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if unvisited[neighbor]:
                    neighbors[neighbors_num] = neighbor
                    neighbors_num += 1
            
            if neighbors_num == 0: # Trail Stuck -> Jump
                sent_seq[sent_seq_idx] = reset_idx
                sent_seq_idx += 1 
                if sent_seq_idx >= seq_length: break

                num_candidates = 0
                for i in range(num_nodes):
                    if unvisited[i]:
                        candidates[num_candidates] = i
                        num_candidates += 1
                current_node = select_candidate(candidates, num_candidates, indptr, jump_bias, rand_r_state, deterministic)
                
            else: 
                current_node = select_candidate(neighbors, neighbors_num, indptr, neighbor_bias, rand_r_state, deterministic)

            if current_node < 0 or current_node >= num_nodes: break

            node_index_map[current_node] = node_index
            sent_seq[sent_seq_idx] = node_index + idx_offset
            node_index += 1
            sent_seq_idx += 1
            unvisited[current_node] = False
            num_unvisited -= 1
            if sent_seq_idx >= seq_length: break

            # Build Neighborhood (Chordal tokens)
            neighborhood_set_size = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if not unvisited[neighbor] and neighbor != prev_node:
                    neighborhood_set[neighborhood_set_size] = node_index_map[neighbor]
                    neighborhood_set_size += 1
            
            if neighborhood_set_size > 0:
                sort_array(neighborhood_set, neighborhood_set_size)
                sent_seq[sent_seq_idx] = left_bracket
                sent_seq_idx += 1
                if sent_seq_idx >= seq_length: break

                for i in range(neighborhood_set_size):
                    sent_seq[sent_seq_idx] = neighborhood_set[i] + idx_offset
                    sent_seq_idx += 1
                    if sent_seq_idx >= seq_length: break
                
                if sent_seq_idx < seq_length:
                    sent_seq[sent_seq_idx] = right_bracket
                    sent_seq_idx += 1

    return np.asarray(sent_seq[:sent_seq_idx]), np.asarray(node_index_map)


# ==============================================================================
# 2. LABELED SAMPLER
# ==============================================================================

def sample_labeled_sent(
    csr_matrix,
    int64_t[:] node_labels,
    int64_t[:] edge_labels,
    int node_idx_offset,
    int edge_idx_offset,
    int seq_length=-1,
    int idx_offset=0,
    int reset_idx=-1,
    int left_bracket=-2,
    int right_bracket=-3,
    bint undirected=True,
    object rng=None,
    int start_bias=0,
    int jump_bias=0,
    int neighbor_bias=0,
    bint deterministic=False
):
    """Sample a labeled SENT with scientific priority strategies."""
    if rng is None:
        rng = np.random.RandomState(0)

    cdef:
        int[:] indices = csr_matrix.indices
        int[:] indptr = csr_matrix.indptr
        int num_nodes = csr_matrix.shape[0]
        int num_edges = indices.shape[0]

    if seq_length < 0:
        seq_length = 20 if num_nodes <= 1 else 2 * (num_nodes * 2 + num_edges * 2)

    cdef:
        uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
        uint32_t* rand_r_state = &rand_r_state_seed

        int num_unvisited = num_nodes
        uint8_t[:] unvisited = np.ones(num_nodes, dtype=np.uint8)

        int64_t[:] sent_seq = np.full(seq_length, reset_idx, dtype=np.int64)
        int64_t[:] node_index_map = np.full(num_nodes, -1, dtype=np.int64)

        int sent_seq_idx = 0
        int64_t node_index = 0
        int current_node, prev_node, i, k, neighbor, sample_idx
        int neighbors_num, num_candidates, neighborhood_set_size
        int64_t edge_index_val

        int[:] neighbors = np.empty(num_nodes, dtype=np.int32)
        int64_t[::1] neighborhood_set = np.empty(num_nodes, dtype=np.int64)
        int[:] current_edge_indices = np.zeros(num_nodes, dtype=np.int32)
        int[:] candidates = np.empty(num_nodes, dtype=np.int32)

    if num_nodes <= 0:
        return np.asarray(sent_seq[:0]), np.asarray(node_index_map)

    with nogil:
        # --- INITIALIZATION ---
        num_candidates = 0
        for i in range(num_nodes):
            candidates[num_candidates] = i
            num_candidates += 1
        current_node = select_candidate(candidates, num_candidates, indptr, start_bias, rand_r_state, deterministic)
        
        if current_node < 0 or current_node >= num_nodes:
            with gil:
                return np.asarray(sent_seq[:0]), np.asarray(node_index_map)

        node_index_map[current_node] = node_index
        sent_seq[sent_seq_idx] = node_index + idx_offset
        node_index += 1
        sent_seq_idx += 1
        unvisited[current_node] = False
        num_unvisited -= 1
        
        # Node Label
        if sent_seq_idx < seq_length:
            sent_seq[sent_seq_idx] = node_labels[current_node] + node_idx_offset
            sent_seq_idx += 1
        
        while num_unvisited > 0 and sent_seq_idx < seq_length:
            prev_node = current_node
            
            # Collect Neighbors and their edge pointers
            neighbors_num = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if unvisited[neighbor]:
                    current_edge_indices[neighbors_num] = k
                    neighbors[neighbors_num] = neighbor
                    neighbors_num += 1
            
            if neighbors_num == 0: # Trail Stuck -> Jump
                sent_seq[sent_seq_idx] = reset_idx
                sent_seq_idx += 1
                if sent_seq_idx >= seq_length: break

                num_candidates = 0
                for i in range(num_nodes):
                    if unvisited[i]:
                        candidates[num_candidates] = i
                        num_candidates += 1
                current_node = select_candidate(candidates, num_candidates, indptr, jump_bias, rand_r_state, deterministic)
                
            else: 
                current_node = select_candidate(neighbors, neighbors_num, indptr, neighbor_bias, rand_r_state, deterministic)
                
                if current_node < 0 or current_node >= num_nodes: break

                # Locate edge index for the chosen neighbor
                sample_idx = -1
                for i in range(neighbors_num):
                    if neighbors[i] == current_node:
                        sample_idx = i
                        break
                
                if sample_idx >= 0:
                    edge_index_val = current_edge_indices[sample_idx]
                    sent_seq[sent_seq_idx] = edge_labels[edge_index_val] + edge_idx_offset
                    sent_seq_idx += 1
                    if sent_seq_idx >= seq_length: break

            if current_node < 0 or current_node >= num_nodes: break

            node_index_map[current_node] = node_index
            sent_seq[sent_seq_idx] = node_index + idx_offset
            node_index += 1
            sent_seq_idx += 1
            unvisited[current_node] = False
            num_unvisited -= 1
            if sent_seq_idx >= seq_length: break
            
            # Node Label
            sent_seq[sent_seq_idx] = node_labels[current_node] + node_idx_offset
            sent_seq_idx += 1
            if sent_seq_idx >= seq_length: break

            # Neighborhood (Chordal tokens)
            neighborhood_set_size = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if not unvisited[neighbor] and neighbor != prev_node:
                    neighborhood_set[neighborhood_set_size] = node_index_map[neighbor]
                    neighborhood_set_size += 1
                    current_edge_indices[node_index_map[neighbor]] = k # Map new index to CSR edge index
            
            if neighborhood_set_size > 0:
                sort_array(neighborhood_set, neighborhood_set_size)
                sent_seq[sent_seq_idx] = left_bracket
                sent_seq_idx += 1
                if sent_seq_idx >= seq_length: break

                for i in range(neighborhood_set_size):
                    # Edge Label then Node Index
                    sent_seq[sent_seq_idx] = edge_labels[current_edge_indices[neighborhood_set[i]]] + edge_idx_offset
                    sent_seq_idx += 1
                    if sent_seq_idx >= seq_length: break

                    sent_seq[sent_seq_idx] = neighborhood_set[i] + idx_offset
                    sent_seq_idx += 1
                    if sent_seq_idx >= seq_length: break

                if sent_seq_idx < seq_length:
                    sent_seq[sent_seq_idx] = right_bracket
                    sent_seq_idx += 1

    return np.asarray(sent_seq[:sent_seq_idx]), np.asarray(node_index_map)


# ==============================================================================
# 3. RECONSTRUCTION
# ==============================================================================

@cython.boundscheck(True)
@cython.wraparound(True)
def reconstruct_graph_from_sent(
    int64_t[:] sent_seq,
    int reset_idx,
    int left_bracket,
    int right_bracket,
    int idx_offset=0,
):
    """Reconstruct graph topology from a SENT sequence."""
    cdef:
        int i
        int walk_length = sent_seq.shape[0]
        int64_t[:, ::1] edge_index = np.zeros((2, walk_length * 2), dtype=np.int64)
        int idx = 0
        int64_t bracket_idx = 0
        bint start_bracket = False
        int64_t u, v

    with nogil:
        for i in range(walk_length - 1):
            if sent_seq[i] == reset_idx or sent_seq[i + 1] == reset_idx or sent_seq[i + 1] == left_bracket:
                start_bracket = False
                continue
            if sent_seq[i] == left_bracket:
                if i > 0:
                    start_bracket = True
                    bracket_idx = sent_seq[i - 1] - idx_offset
            elif sent_seq[i] == right_bracket and start_bracket:
                u, v = bracket_idx, sent_seq[i + 1] - idx_offset
                if u >= 0 and v >= 0:
                    edge_index[0, idx] = u
                    edge_index[1, idx] = v
                    idx += 1
                start_bracket = False
            elif start_bracket:
                u, v = bracket_idx, sent_seq[i] - idx_offset
                if u >= 0 and v >= 0:
                    edge_index[0, idx] = u
                    edge_index[1, idx] = v
                    idx += 1
            else:
                u, v = sent_seq[i] - idx_offset, sent_seq[i + 1] - idx_offset
                if u >= 0 and v >= 0:
                    edge_index[0, idx] = u
                    edge_index[1, idx] = v
                    idx += 1

    return np.asarray(edge_index[:, :idx])


@cython.boundscheck(True)
@cython.wraparound(True)
def reconstruct_graph_from_labeled_sent(
    int64_t[:] sent_seq,
    int reset_idx,
    int left_bracket,
    int right_bracket,
    int idx_offset=0,
    int max_nodes=1000,
):
    """Reconstruct graph topology and labels from a SENT sequence."""
    cdef:
        int i = 0
        int walk_length = sent_seq.shape[0]
        int64_t[:, ::1] edge_index = np.zeros((2, walk_length * 2), dtype=np.int64)
        int64_t[:] node_labels = np.full(max_nodes, -1, dtype=np.int64)
        int64_t[:] edge_labels = np.full(walk_length * 2, -1, dtype=np.int64)
        int idx = 0
        int64_t current_node_idx = 0
        int64_t bracket_idx = 0
        bint start_bracket = False
        int64_t u, v

    with nogil:
        while i < walk_length - 1:
            if sent_seq[i] == reset_idx or sent_seq[i + 1] == reset_idx:
                start_bracket = False
                i += 1
                continue
            
            if sent_seq[i] == left_bracket:
                if i >= 2:
                    start_bracket = True
                    bracket_idx = sent_seq[i - 2] - idx_offset
                i += 1
            elif sent_seq[i] == right_bracket and start_bracket:
                if i + 2 < walk_length:
                    u, v = bracket_idx, sent_seq[i + 2] - idx_offset
                    if u >= 0 and v >= 0 and u < max_nodes and v < max_nodes:
                        edge_index[0, idx] = u
                        edge_index[1, idx] = v
                        if edge_labels[idx] == -1:
                            edge_labels[idx] = sent_seq[i + 1]
                        idx += 1
                start_bracket = False
                i += 2
            elif start_bracket:
                if i + 1 < walk_length:
                    u, v = bracket_idx, sent_seq[i + 1] - idx_offset
                    if u >= 0 and v >= 0 and u < max_nodes and v < max_nodes:
                        edge_index[0, idx] = u
                        edge_index[1, idx] = v
                        if edge_labels[idx] == -1:
                            edge_labels[idx] = sent_seq[i]
                        idx += 1
                i += 2
            elif i + 2 < walk_length and sent_seq[i + 2] == reset_idx:
                current_node_idx = sent_seq[i] - idx_offset
                if current_node_idx >= 0 and current_node_idx < max_nodes:
                    if node_labels[current_node_idx] == -1:
                        node_labels[current_node_idx] = sent_seq[i + 1]
                i += 2
            elif i + 2 < walk_length and sent_seq[i + 2] == left_bracket:
                current_node_idx = sent_seq[i] - idx_offset
                if current_node_idx >= 0 and current_node_idx < max_nodes:
                    if node_labels[current_node_idx] == -1:
                        node_labels[current_node_idx] = sent_seq[i + 1]
                i += 2
            else:
                current_node_idx = sent_seq[i] - idx_offset
                if i + 3 < walk_length:
                    u, v = current_node_idx, sent_seq[i + 3] - idx_offset
                    if u >= 0 and v >= 0 and u < max_nodes and v < max_nodes:
                        edge_index[0, idx] = u
                        edge_index[1, idx] = v
                        if edge_labels[idx] == -1:
                            edge_labels[idx] = sent_seq[i + 2]
                        idx += 1
                if current_node_idx >= 0 and current_node_idx < max_nodes:
                    if node_labels[current_node_idx] == -1:
                        node_labels[current_node_idx] = sent_seq[i + 1]
                i += 3

    return np.asarray(edge_index[:, :idx]), np.asarray(node_labels), np.asarray(edge_labels[:idx])