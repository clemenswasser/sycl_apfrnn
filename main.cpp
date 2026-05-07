#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;

inline uint32_t get_hash(uint32_t cx, uint32_t cy, uint32_t cz, uint32_t mask) {
  const uint32_t p1 = 73856093;
  const uint32_t p2 = 19349663;
  const uint32_t p3 = 83492791;
  return ((cx * p1) ^ (cy * p2) ^ (cz * p3)) & mask;
}

double run_gpu_neighbor_search(queue &q, const std::vector<sycl::float3> &h_pos,
                               float radius, bool print) {
  int N = h_pos.size();

  if (print) {
    std::cout << "--- GPU Pipeline Starting (" << N << " particles) ---\n";
  }

  const uint32_t TABLE_BITS = 22;
  const uint32_t TABLE_SIZE = 1u << TABLE_BITS;
  const uint32_t TABLE_MASK = TABLE_SIZE - 1;
  sycl::float3 *d_pos = malloc_device<sycl::float3>(N, q);
  uint32_t *d_keys = malloc_device<uint32_t>(N, q);
  uint32_t *d_values = malloc_device<uint32_t>(N, q);
  float *d_sorted_pos_x = malloc_device<float>(N, q);
  float *d_sorted_pos_y = malloc_device<float>(N, q);
  float *d_sorted_pos_z = malloc_device<float>(N, q);
  int *d_counts = malloc_device<int>(N + 1, q);
  int *d_offsets = malloc_device<int>(N + 1, q);
  uint32_t *d_cell_start = malloc_device<uint32_t>(TABLE_SIZE, q);
  uint32_t *d_cell_end = malloc_device<uint32_t>(TABLE_SIZE, q);

  q.fill(d_counts, 0, N + 1);
  q.fill(d_cell_start, 0u, TABLE_SIZE);
  q.fill(d_cell_end, 0u, TABLE_SIZE);
  q.memcpy(d_pos, h_pos.data(), N * sizeof(sycl::float3)).wait();

  const float WORLD_OFFSET = 10000.0f;
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

        sycl::float3 p = d_pos[idx];

        uint32_t cx = (uint32_t)((p.x() + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((p.y() + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((p.z() + WORLD_OFFSET) * inv_radius);

        d_keys[idx] = get_hash(cx, cy, cz, TABLE_MASK);
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

  evt = q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)),
                       [=](nd_item<1> item) {
                         int idx = item.get_global_id(0);
                         if (idx >= N)
                           return;

                         int orig_idx = d_values[idx];
                         sycl::float3 p = d_pos[orig_idx];

                         d_sorted_pos_x[idx] = p.x();
                         d_sorted_pos_y[idx] = p.y();
                         d_sorted_pos_z[idx] = p.z();

                         uint32_t hash = d_keys[idx];
                         if (idx == 0) {
                           d_cell_start[hash] = 0;
                         } else {
                           uint32_t prev_hash = d_keys[idx - 1];
                           if (hash != prev_hash) {
                             d_cell_start[hash] = idx;
                             d_cell_end[prev_hash] = idx;
                           }
                         }
                         if (idx == N - 1) {
                           d_cell_end[hash] = N;
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

        float my_x = d_sorted_pos_x[idx];
        float my_y = d_sorted_pos_y[idx];
        float my_z = d_sorted_pos_z[idx];

        uint32_t cx = (uint32_t)((my_x + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((my_y + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((my_z + WORLD_OFFSET) * inv_radius);

        int neighbor_count = 0;

#pragma unroll 1
        for (int dz = -1; dz <= 1; ++dz) {
          uint32_t hash_z = (cz + dz) * 83492791;
#pragma unroll 1
          for (int dy = -1; dy <= 1; ++dy) {
            uint32_t hash_zy = hash_z ^ ((cy + dy) * 19349663);
#pragma unroll 1
            for (int dx = -1; dx <= 1; ++dx) {
              uint32_t neighbor_hash =
                  (hash_zy ^ ((cx + dx) * 73856093)) & TABLE_MASK;

              uint32_t start = d_cell_start[neighbor_hash];
              uint32_t end = d_cell_end[neighbor_hash];

              for (uint32_t j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                float dist_x = my_x - d_sorted_pos_x[j];
                float dist_y = my_y - d_sorted_pos_y[j];
                float dist_z = my_z - d_sorted_pos_z[j];

                float dist2 =
                    dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

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

        float my_x = d_sorted_pos_x[idx];
        float my_y = d_sorted_pos_y[idx];
        float my_z = d_sorted_pos_z[idx];
        uint32_t cx = (uint32_t)((my_x + WORLD_OFFSET) * inv_radius);
        uint32_t cy = (uint32_t)((my_y + WORLD_OFFSET) * inv_radius);
        uint32_t cz = (uint32_t)((my_z + WORLD_OFFSET) * inv_radius);
        int original_idx = d_values[idx];
        int write_offset = d_offsets[original_idx];
        int local_neighbor_count = 0;

#pragma unroll 1
        for (int dz = -1; dz <= 1; ++dz) {
          uint32_t hash_z = (cz + dz) * 83492791;
#pragma unroll 1
          for (int dy = -1; dy <= 1; ++dy) {
            uint32_t hash_zy = hash_z ^ ((cy + dy) * 19349663);
#pragma unroll 1
            for (int dx = -1; dx <= 1; ++dx) {
              uint32_t neighbor_hash =
                  (hash_zy ^ ((cx + dx) * 73856093)) & TABLE_MASK;

              uint32_t start = d_cell_start[neighbor_hash];
              uint32_t end = d_cell_end[neighbor_hash];

              for (uint32_t j = start; j < end; ++j) {
                if (idx == j)
                  continue;

                float dist_x = my_x - d_sorted_pos_x[j];
                float dist_y = my_y - d_sorted_pos_y[j];
                float dist_z = my_z - d_sorted_pos_z[j];
                float dist2 =
                    dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

                if (dist2 <= radius2) {
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
  free(d_sorted_pos_x, q);
  free(d_sorted_pos_y, q);
  free(d_sorted_pos_z, q);
  free(d_counts, q);
  free(d_offsets, q);
  free(d_cell_start, q);
  free(d_cell_end, q);
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