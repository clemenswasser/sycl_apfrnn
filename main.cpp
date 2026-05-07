#include <chrono>
#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;

inline uint64_t expand_bits(uint32_t v) {
  uint64_t x = v & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffffULL;
  x = (x | x << 16) & 0x1f0000ff0000ffULL;
  x = (x | x << 8) & 0x100f00f00f00f00fULL;
  x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
  x = (x | x << 2) & 0x1249249249249249ULL;
  return x;
}

inline int lower_bound(const uint64_t *arr, int n, uint64_t key) {
  int left = 0, right = n;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] < key)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

// Device binary search: Finds the end boundary of the requested cell
inline int upper_bound(const uint64_t *arr, int n, uint64_t key) {
  int left = 0, right = n;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] <= key)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

double run_gpu_neighbor_search(queue &q, const std::vector<sycl::float3> &h_pos,
                               float radius, bool print) {
  int N = h_pos.size();

  if (print) {
    std::cout << "--- GPU Pipeline Starting (" << N << " particles) ---\n";
  }

  sycl::float3 *d_pos = malloc_device<sycl::float3>(N, q);
  uint64_t *d_keys = malloc_device<uint64_t>(N, q);
  uint32_t *d_values = malloc_device<uint32_t>(N, q);
  sycl::float3 *d_sorted_pos = malloc_device<sycl::float3>(N, q);

  int *d_counts = malloc_device<int>(N + 1, q);
  int *d_offsets = malloc_device<int>(N + 1, q);

  q.fill(d_counts, 0, N + 1).wait();

  q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float3)).wait();

  const float WORLD_OFFSET = 10000.0f; // Offset to ensure positive grid cells
  const float inv_radius = 1.0f / radius;
  size_t local_size = 256;
  size_t global_size = ((N + local_size - 1) / local_size) * local_size;

  sycl::event global_start_evt =
      q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });

  auto evt = q.parallel_for(
      nd_range<1>(range<1>(global_size), range<1>(local_size)),
      [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N)
          return;

        sycl::float3 p = d_pos[idx];

        uint32_t cx = sycl::floor((p.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = sycl::floor((p.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = sycl::floor((p.z() + WORLD_OFFSET) * inv_radius);

        d_keys[idx] =
            expand_bits(cx) | (expand_bits(cy) << 1) | (expand_bits(cz) << 2);
        d_values[idx] = idx;
      });
  evt.wait();
  uint64_t start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  double true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "1. Morton Codes Gen:   " << true_gpu_time_ms << " ms\n";

  auto policy = oneapi::dpl::execution::make_device_policy(q);
  sycl::event start_evt =
      q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });
  oneapi::dpl::sort_by_key(policy, d_keys, d_keys + N, d_values);
  sycl::event end_evt =
      q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });
  end_evt.wait();

  start_ns =
      start_evt
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns =
      end_evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "2. Hardware Radix Sort: " << true_gpu_time_ms << " ms\n";

  evt = q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)),
                       [=](nd_item<1> item) {
                         int idx = item.get_global_id(0);
                         if (idx < N) {
                           d_sorted_pos[idx] = d_pos[d_values[idx]];
                         }
                       });
  evt.wait();
  start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "3. Position Reorder:   " << true_gpu_time_ms << " ms\n";

  evt = q.parallel_for(
      nd_range<1>(range<1>(global_size), range<1>(local_size)),
      [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N)
          return;

        sycl::float3 my_pos = d_sorted_pos[idx];
        float radius2 = radius * radius;

        uint32_t cx = sycl::floor((my_pos.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = sycl::floor((my_pos.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = sycl::floor((my_pos.z() + WORLD_OFFSET) * inv_radius);

        int neighbor_count = 0;

#pragma unroll 3
        for (int dz = -1; dz <= 1; ++dz) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

              uint64_t neighbor_code = expand_bits(cx + dx) |
                                       (expand_bits(cy + dy) << 1) |
                                       (expand_bits(cz + dz) << 2);

              int start = lower_bound(d_keys, N, neighbor_code);
              int end = upper_bound(d_keys, N, neighbor_code);

              for (int j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                sycl::float3 n_pos = d_sorted_pos[j];
                sycl::float3 dist_vec = my_pos - n_pos;
                float dist2 = dist_vec.x() * dist_vec.x() +
                              dist_vec.y() * dist_vec.y() +
                              dist_vec.z() * dist_vec.z();

                if (dist2 <= radius2) {
                  neighbor_count++;
                }
              }
            }
          }
        }

        int original_idx = d_values[idx];
        d_counts[original_idx] = neighbor_count;
      });
  evt.wait();
  start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "4. Neighbor Count:     " << true_gpu_time_ms << " ms\n";

  start_evt = q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });
  oneapi::dpl::exclusive_scan(policy, d_counts, d_counts + N + 1, d_offsets, 0);
  end_evt = q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });
  end_evt.wait();

  start_ns =
      start_evt
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns =
      end_evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "5. Prefix Sum:         " << true_gpu_time_ms << " ms\n";

  // Read total neighbor sum to allocate exact flat array
  int total_neighbors = 0;
  q.memcpy(&total_neighbors, d_offsets + N, sizeof(int)).wait();

  int *d_neighbors = nullptr;
  if (total_neighbors > 0) {
    d_neighbors = malloc_device<int>(total_neighbors, q);
  }

  evt = q.parallel_for(
      nd_range<1>(range<1>(global_size), range<1>(local_size)),
      [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N)
          return;

        sycl::float3 my_pos = d_sorted_pos[idx];
        float radius2 = radius * radius;

        uint32_t cx = sycl::floor((my_pos.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = sycl::floor((my_pos.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = sycl::floor((my_pos.z() + WORLD_OFFSET) * inv_radius);

        int original_idx = d_values[idx];
        int write_offset = d_offsets[original_idx];
        int local_neighbor_count = 0;

#pragma unroll 3
        for (int dz = -1; dz <= 1; ++dz) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

              uint64_t neighbor_code = expand_bits(cx + dx) |
                                       (expand_bits(cy + dy) << 1) |
                                       (expand_bits(cz + dz) << 2);

              int start = lower_bound(d_keys, N, neighbor_code);
              int end = upper_bound(d_keys, N, neighbor_code);

              for (int j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                sycl::float3 n_pos = d_sorted_pos[j];
                sycl::float3 dist_vec = my_pos - n_pos;
                float dist2 = dist_vec.x() * dist_vec.x() +
                              dist_vec.y() * dist_vec.y() +
                              dist_vec.z() * dist_vec.z();

                if (dist2 <= radius2) {
                  // Write the **original** index of the neighbor into the list
                  d_neighbors[write_offset + local_neighbor_count] =
                      d_values[j];
                  local_neighbor_count++;
                }
              }
            }
          }
        }
      });
  evt.wait();
  start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "6. Populate Neighbors: " << true_gpu_time_ms << " ms\n";

  sycl::event global_end_evt =
      q.submit([&](sycl::handler &h) { h.single_task<>([]() {}); });
  global_end_evt.wait();
  start_ns =
      global_start_evt
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns = global_end_evt
               .get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;

  if (print) {
    std::cout << "------------------------------------------\n";
    std::cout << "TOTAL GPU PIPELINE:    " << true_gpu_time_ms << " ms\n\n";

    std::cout << "Total exact interactions found: " << total_neighbors << "\n";
    std::cout << "Average neighbors per particle: "
              << (float)total_neighbors / N << "\n";

    // Read back Particle 0's neighbor list to verify correctness
    std::vector<int> h_offsets(2);
    q.memcpy(h_offsets.data(), d_offsets, 2 * sizeof(int)).wait();
    int p0_count = h_offsets[1] - h_offsets[0];

    std::cout << "Sample count for Particle 0: " << p0_count << "\n";
    if (p0_count > 0 && d_neighbors) {
      std::vector<int> p0_neighbors(p0_count);
      q.memcpy(p0_neighbors.data(), d_neighbors + h_offsets[0],
               p0_count * sizeof(int))
          .wait();
      std::cout << "Neighbors of Particle 0 (original IDs): ";
      for (int i = 0; i < std::min(10, p0_count); ++i) {
        std::cout << p0_neighbors[i] << " ";
      }
      if (p0_count > 10)
        std::cout << "...";
      std::cout << "\n";
    }
  }

  // Free resources
  free(d_pos, q);
  free(d_keys, q);
  free(d_values, q);
  free(d_sorted_pos, q);
  free(d_counts, q);
  free(d_offsets, q);
  if (d_neighbors)
    free(d_neighbors, q);

  return true_gpu_time_ms;
}

int main() {
  int NUM_PARTICLES = 1000000;
  float SEARCH_RADIUS = 1.0f;
  float BOX_SIZE = 50.0f;

  std::cout << "Generating " << NUM_PARTICLES << " random particles...\n";
  std::vector<sycl::float3> h_pos(NUM_PARTICLES);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, BOX_SIZE);

  for (int i = 0; i < NUM_PARTICLES; ++i) {
    h_pos[i] = sycl::float3(dist(rng), dist(rng), dist(rng));
  }

  sycl::queue q(sycl::gpu_selector_v,
                {sycl::property::queue::in_order(),
                 sycl::property::queue::enable_profiling()});
  std::cout << "\nRunning on Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

  double avg_time_ms = 0.0;
  for (uint32_t i = 0; i < 10; ++i) {
    avg_time_ms += run_gpu_neighbor_search(q, h_pos, SEARCH_RADIUS, false);
  }
  avg_time_ms /= 10.0;
  std::cout << "Average GPU time over 10 runs: " << avg_time_ms << " ms\n";
  run_gpu_neighbor_search(q, h_pos, SEARCH_RADIUS, true);

  return 0;
}