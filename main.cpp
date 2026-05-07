#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;

inline uint32_t expand_bits(uint32_t v) {
  v = v & 0x3FF;
  v = (v | (v << 16)) & 0xFF0000FF;
  v = (v | (v << 8)) & 0x0300F00F;
  v = (v | (v << 4)) & 0x030C30C3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

double run_gpu_neighbor_search(queue &q, const std::vector<sycl::float3> &h_pos,
                               float radius, bool print) {
  int N = h_pos.size();

  if (print) {
    std::cout << "--- GPU Pipeline Starting (" << N << " particles) ---\n";
  }

  const uint32_t TABLE_BITS = 21;
  const uint32_t TABLE_SIZE = 1u << TABLE_BITS;
  const uint32_t TABLE_MASK = TABLE_SIZE - 1;
  sycl::float4 *d_pos = malloc_device<sycl::float4>(N, q);
  sycl::float4 *d_sorted_pos = malloc_device<sycl::float4>(N, q);
  uint32_t *d_keys = malloc_device<uint32_t>(N, q);
  uint32_t *d_values = malloc_device<uint32_t>(N, q);
  int *d_counts = malloc_device<int>(N + 1, q);
  int *d_offsets = malloc_device<int>(N + 1, q);
  uint64_t *d_cell_bounds = malloc_device<uint64_t>(TABLE_SIZE, q);

  q.fill(d_counts, 0, N + 1);
  q.fill(d_cell_bounds, 0ULL, TABLE_SIZE);
  q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float3)).wait();

  const float WORLD_OFFSET = 10.0f;
  const float inv_radius = 1.0f / radius;
  const float radius2 = radius * radius;
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

        sycl::float4 p = d_pos[idx];
        uint32_t cx = (uint32_t)((p.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((p.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((p.z() + WORLD_OFFSET) * inv_radius);

        d_keys[idx] = (expand_bits(cx) | (expand_bits(cy) << 1) |
                       (expand_bits(cz) << 2)) &
                      TABLE_MASK;
        d_values[idx] = idx;
      });
  evt.wait();
  uint64_t start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t end_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  double true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "1. Hash Codes Gen:     " << true_gpu_time_ms << " ms\n";

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

  uint32_t *d_cell_bounds_32 = reinterpret_cast<uint32_t *>(d_cell_bounds);

  evt = q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)),
                       [=](nd_item<1> item) {
                         int idx = item.get_global_id(0);
                         if (idx >= N)
                           return;

                         int orig_idx = d_values[idx];
                         sycl::float4 p = d_pos[orig_idx];
                         p.w() = sycl::bit_cast<float>(orig_idx);
                         d_sorted_pos[idx] = p;

                         uint32_t hash = d_keys[idx];
                         if (idx == 0) {
                           d_cell_bounds_32[hash * 2] = 0;
                         } else {
                           uint32_t prev_hash = d_keys[idx - 1];
                           if (hash != prev_hash) {
                             d_cell_bounds_32[hash * 2] = idx;
                             d_cell_bounds_32[prev_hash * 2 + 1] = idx;
                           }
                         }
                         if (idx == N - 1) {
                           d_cell_bounds_32[hash * 2 + 1] = N;
                         }
                       });
  evt.wait();
  start_ns =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  end_ns = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  true_gpu_time_ms = (end_ns - start_ns) / 1e6;
  if (print)
    std::cout << "3. Pos Reorder & Spans: " << true_gpu_time_ms << " ms\n";

  evt = q.parallel_for(
      nd_range<1>(range<1>(global_size), range<1>(local_size)),
      [=](nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx >= N)
          return;

        sycl::float4 my_pos = d_sorted_pos[idx];
        uint32_t cx = (uint32_t)((my_pos.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((my_pos.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((my_pos.z() + WORLD_OFFSET) * inv_radius);
        int neighbor_count = 0;
        uint32_t cx_m[3], cy_m[3], cz_m[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          cx_m[i] = expand_bits(cx + i - 1);
          cy_m[i] = expand_bits(cy + i - 1) << 1;
          cz_m[i] = expand_bits(cz + i - 1) << 2;
        }

#pragma unroll 1
        for (int z = 0; z < 3; ++z) {
#pragma unroll 1
          for (int y = 0; y < 3; ++y) {
#pragma unroll 1
            for (int x = 0; x < 3; ++x) {
              uint32_t neighbor_hash =
                  (cx_m[x] | cy_m[y] | cz_m[z]) & TABLE_MASK;

              uint64_t bounds = d_cell_bounds[neighbor_hash];
              uint32_t start = (uint32_t)(bounds & 0xFFFFFFFF);
              uint32_t end = (uint32_t)(bounds >> 32);

              for (uint32_t j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                sycl::float4 n_pos = d_sorted_pos[j];
                float dx = my_pos.x() - n_pos.x();
                float dy = my_pos.y() - n_pos.y();
                float dz = my_pos.z() - n_pos.z();

                float dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 <= radius2) {
                  neighbor_count++;
                }
              }
            }
          }
        }

        int original_idx = sycl::bit_cast<int>(my_pos.w());
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

        sycl::float4 my_pos = d_sorted_pos[idx];
        uint32_t cx = (uint32_t)((my_pos.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((my_pos.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((my_pos.z() + WORLD_OFFSET) * inv_radius);
        int original_idx = sycl::bit_cast<int>(my_pos.w());
        int write_offset = d_offsets[original_idx];
        int local_neighbor_count = 0;
        uint32_t cx_m[3], cy_m[3], cz_m[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          cx_m[i] = expand_bits(cx + i - 1);
          cy_m[i] = expand_bits(cy + i - 1) << 1;
          cz_m[i] = expand_bits(cz + i - 1) << 2;
        }

#pragma unroll 1
        for (int z = 0; z < 3; ++z) {
#pragma unroll 1
          for (int y = 0; y < 3; ++y) {
#pragma unroll 1
            for (int x = 0; x < 3; ++x) {
              uint32_t neighbor_hash =
                  (cx_m[x] | cy_m[y] | cz_m[z]) & TABLE_MASK;

              uint64_t bounds = d_cell_bounds[neighbor_hash];
              uint32_t start = (uint32_t)(bounds & 0xFFFFFFFF);
              uint32_t end = (uint32_t)(bounds >> 32);

              for (uint32_t j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                sycl::float4 n_pos = d_sorted_pos[j];
                float dx = my_pos.x() - n_pos.x();
                float dy = my_pos.y() - n_pos.y();
                float dz = my_pos.z() - n_pos.z();

                float dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 <= radius2) {
                  d_neighbors[write_offset + local_neighbor_count] =
                      sycl::bit_cast<int>(n_pos.w());
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

  free(d_pos, q);
  free(d_keys, q);
  free(d_values, q);
  free(d_sorted_pos, q);
  free(d_counts, q);
  free(d_offsets, q);
  free(d_cell_bounds, q);
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