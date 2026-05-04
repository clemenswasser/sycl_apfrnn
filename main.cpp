#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace sycl;

// =========================================================================
// DEVICE HELPERS: Morton Code Generation & Binary Search
// =========================================================================

// Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit
inline uint64_t expand_bits(uint32_t v) {
    uint64_t x = v & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8)  & 0x100f00f00f00f00fULL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2)  & 0x1249249249249249ULL;
    return x;
}

// Device binary search: Finds the first particle in the requested cell
inline int lower_bound(const uint64_t* arr, int n, uint64_t key) {
    int left = 0, right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < key) left = mid + 1;
        else right = mid;
    }
    return left;
}

// Device binary search: Finds the end boundary of the requested cell
inline int upper_bound(const uint64_t* arr, int n, uint64_t key) {
    int left = 0, right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= key) left = mid + 1;
        else right = mid;
    }
    return left;
}

// =========================================================================
// TIMER UTILITY
// =========================================================================
struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// =========================================================================
// MAIN PIPELINE
// =========================================================================

void run_gpu_neighbor_search(queue& q, const std::vector<sycl::float3>& h_pos, float radius) {
    int N = h_pos.size();
    Timer timer, total_timer;
    
    std::cout << "--- GPU Pipeline Starting (" << N << " particles) ---\n";
    total_timer.start();

    // 1. Allocate USM (Unified Shared Memory) Device buffers
    sycl::float3* d_pos       = malloc_device<sycl::float3>(N, q);
    uint64_t*     d_keys      = malloc_device<uint64_t>(N, q);
    uint32_t*     d_values    = malloc_device<uint32_t>(N, q);
    sycl::float3* d_sorted_pos= malloc_device<sycl::float3>(N, q);
    int*          d_counts    = malloc_device<int>(N, q);

    // Host to Device Copy
    q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float3)).wait();

    const float WORLD_OFFSET = 10000.0f; // Offset to ensure positive grid cells
    const float inv_radius = 1.0f / radius;
    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    // ---------------------------------------------------------------------
    // STEP 1: Compute Morton Codes
    // ---------------------------------------------------------------------
    timer.start();
    q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), 
    [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N) return;

        sycl::float3 p = d_pos[idx];
        
        uint32_t cx = sycl::floor((p.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = sycl::floor((p.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = sycl::floor((p.z() + WORLD_OFFSET) * inv_radius);

        d_keys[idx] = expand_bits(cx) | (expand_bits(cy) << 1) | (expand_bits(cz) << 2);
        d_values[idx] = idx; 
        d_counts[idx] = 0; // Initialize counts
    }).wait();
    std::cout << "1. Morton Codes Gen:   " << timer.stop() << " ms\n";

    // ---------------------------------------------------------------------
    // STEP 2: Radix Sort via oneDPL
    // ---------------------------------------------------------------------
    timer.start();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    oneapi::dpl::sort_by_key(policy, d_keys, d_keys + N, d_values);
    q.wait();
    std::cout << "2. Hardware Radix Sort:" << timer.stop() << " ms\n";

    // ---------------------------------------------------------------------
    // STEP 3: Reorder the Position Array
    // ---------------------------------------------------------------------
    timer.start();
    q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), 
    [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx < N) {
            d_sorted_pos[idx] = d_pos[d_values[idx]];
        }
    }).wait();
    std::cout << "3. Position Reorder:   " << timer.stop() << " ms\n";

    // ---------------------------------------------------------------------
    // STEP 4: Linear Cell List Search Kernel
    // ---------------------------------------------------------------------
    timer.start();
    q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), 
    [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N) return;

        sycl::float3 my_pos = d_sorted_pos[idx];
        float radius2 = radius * radius;
        
        uint32_t cx = sycl::floor((my_pos.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = sycl::floor((my_pos.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = sycl::floor((my_pos.z() + WORLD_OFFSET) * inv_radius);

        int neighbor_count = 0;

        #pragma unroll 3
        for(int dz = -1; dz <= 1; ++dz) {
            for(int dy = -1; dy <= 1; ++dy) {
                for(int dx = -1; dx <= 1; ++dx) {
                    
                    uint64_t neighbor_code = expand_bits(cx + dx) | 
                                            (expand_bits(cy + dy) << 1) | 
                                            (expand_bits(cz + dz) << 2);
                    
                    int start = lower_bound(d_keys, N, neighbor_code);
                    int end   = upper_bound(d_keys, N, neighbor_code);

                    for(int j = start; j < end; ++j) {
                        if (idx == j) continue; // Skip self

                        sycl::float3 n_pos = d_sorted_pos[j];
                        
                        sycl::float3 dist_vec = my_pos - n_pos;
                        float dist2 = dist_vec.x()*dist_vec.x() + 
                                      dist_vec.y()*dist_vec.y() + 
                                      dist_vec.z()*dist_vec.z();

                        if (dist2 <= radius2) {
                            neighbor_count++;
                        }
                    }
                }
            }
        }

        // Scatter back to original index to maintain mapping!
        int original_idx = d_values[idx];
        d_counts[original_idx] = neighbor_count;
    }).wait();
    std::cout << "4. Neighbor Search:    " << timer.stop() << " ms\n";
    std::cout << "------------------------------------------\n";
    std::cout << "TOTAL GPU PIPELINE:    " << total_timer.stop() << " ms\n\n";

    // Read back a few results to verify correctness
    std::vector<int> h_counts(N);
    q.memcpy(h_counts.data(), d_counts, N * sizeof(int)).wait();

    int sum_neighbors = 0;
    for(int i = 0; i < N; i++) sum_neighbors += h_counts[i];
    std::cout << "Average neighbors per particle: " << (float)sum_neighbors / N << "\n";
    std::cout << "Sample count for Particle 0: " << h_counts[0] << "\n";

    free(d_pos, q);
    free(d_keys, q);
    free(d_values, q);
    free(d_sorted_pos, q);
    free(d_counts, q);
}

int main() {
    int NUM_PARTICLES = 1000000;
    float SEARCH_RADIUS = 1.0f;
    float BOX_SIZE = 50.0f; // 1 million particles in a 50x50x50 box

    std::cout << "Generating " << NUM_PARTICLES << " random particles...\n";
    std::vector<sycl::float3> h_pos(NUM_PARTICLES);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, BOX_SIZE);

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_pos[i] = sycl::float3(dist(rng), dist(rng), dist(rng));
    }

    try {
        // Automatically select the best GPU available
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        std::cout << "\nRunning on Device: " 
                  << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

        run_gpu_neighbor_search(q, h_pos, SEARCH_RADIUS);
    } catch (sycl::exception const& e) {
        std::cout << "SYCL Exception: " << e.what() << "\n";
    }

    return 0;
}