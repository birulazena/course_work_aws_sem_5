#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "topology.hpp"

struct Matrix {
  int n;
  float* data = nullptr;

  Matrix(int size) : n(size) {
    size_t bytes = sizeof(float) * static_cast<size_t>(n) * n;
    if (posix_memalign(reinterpret_cast<void**>(&data), 64, bytes) != 0) {
      data = nullptr;
    }
  }

  ~Matrix() {
    if (data) {
      free(data);
    }
  }

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  void init_a() {
    for (int i = 0; i < n * n; ++i) {
      data[i] = static_cast<float>((i % 97) + 1) * 0.01f;
    }
  }

  void init_b() {
    for (int i = 0; i < n * n; ++i) {
      data[i] = static_cast<float>(((i * 7) % 97) + 1) * 0.01f;
    }
  }

  void zero() {
    for (int i = 0; i < n * n; ++i) {
      data[i] = 0.0f;
    }
  }

  // Контрольная сумма диагонали для предотвращения оптимизации мертвого кода
  // компилятором
  float checksum() const {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += data[i * n + i];
    }
    return sum;
  }
};

struct BenchResult {
  double elapsed_ms;
  double gflops;
  float checksum;
};

class CpuBenchmark {
 public:
  static int get_iterations(int n) {
    if (n <= 64) return 200;
    if (n <= 256) return 30;
    if (n <= 1024) return 5;
    if (n <= 4096) return 2;
    return 1;
  }

  static BenchResult run(int n, const std::vector<int>& cpu_ids) {
    Matrix A(n);
    Matrix B(n);
    Matrix C(n);

    A.init_a();
    B.init_b();
    C.zero();

    const int num_threads = static_cast<int>(cpu_ids.size());
    const int iters = get_iterations(n);

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      int cpu_id = cpu_ids[t];
      workers.emplace_back([&, t, cpu_id]() {
        // Привязываем поток к целевому логическому CPU
        CpuTopology::pin_current_thread(cpu_id);

        // Ожидаем отмашки старта таймера
        while (!start_flag.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
          __builtin_ia32_pause();
#endif
        }

        int r_start = (t * n) / num_threads;
        int r_end = ((t + 1) * n) / num_threads;

        for (int it = 0; it < iters; ++it) {
          for (int i = r_start; i < r_end; ++i) {
            float* c_row = &C.data[i * n];
            for (int k = 0; k < n; ++k) {
              float a_val = A.data[i * n + k];
              const float* b_row = &B.data[k * n];

#pragma GCC ivdep
              for (int j = 0; j < n; ++j) {
                c_row[j] += a_val * b_row[j];
              }
            }
          }
        }
      });
    }

    // Запуск замера времени после готовности и привязки всех потоков
    auto t_start = std::chrono::high_resolution_clock::now();
    start_flag.store(true, std::memory_order_release);

    for (auto& w : workers) {
      w.join();
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff = t_end - t_start;
    double elapsed_ms = diff.count();

    // 2 * N^3 операций на одно перемножение (умножение + сложение)
    double total_ops = 2.0 * static_cast<double>(n) * n * n * iters;
    double gflops = (total_ops / (elapsed_ms * 1e-3)) / 1e9;

    return {elapsed_ms / iters, gflops, C.checksum()};
  }
};