#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_bench.hpp"
#include "gpu_bench.hpp"
#include "topology.hpp"

static std::vector<int> load_matrix_sizes(const std::string& filename) {
  std::vector<int> sizes;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Ошибка: не удалось открыть " << filename
              << ". Используются базовые размеры.\n";
    return {32, 64, 128, 256, 512, 1024, 2048, 4096};
  }

  std::string line;
  while (std::getline(file, line)) {
    size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    std::stringstream ss(line);
    int n = 0;
    if (ss >> n && n > 0) {
      sizes.push_back(n);
    }
  }
  return sizes;
}

int main() {
  std::cout << "=========================================================\n";
  std::cout << "         CPU & iGPU MATRIX MULTIPLICATION BENCHMARK      \n";
  std::cout << "=========================================================\n";

  // 1. Определение аппаратной конфигурации CPU
  CpuTopology topo = CpuTopology::detect();
  std::cout << "[CPU] Обнаружено логических потоков: "
            << topo.total_logical_cpus << "\n";
  std::cout << "[CPU] Обнаружено физических ядер:    " << topo.cores.size()
            << "\n";
  std::cout << "[CPU] Выделенное ядро для режимов 1-2: CoreID "
            << topo.cores[0].core_id << " (CPUs: ";
  for (size_t i = 0; i < topo.cores[0].logical_cpus.size(); ++i) {
    std::cout << topo.cores[0].logical_cpus[i]
              << (i + 1 < topo.cores[0].logical_cpus.size() ? ", " : "");
  }
  std::cout << ")\n";

  // 2. Инициализация iGPU
  GpuBenchmark gpu_bench;
  if (gpu_bench.available) {
    std::cout << "[iGPU] Найдено OpenCL устройство:  " << gpu_bench.device_name
              << "\n";
  } else {
    std::cout << "[iGPU] OpenCL GPU не найден или драйвер не установлен. Режим "
                 "5 будет пропущен.\n";
  }

  // 3. Загрузка размеров матриц
  std::vector<int> sizes = load_matrix_sizes("sizes.cfg");
  std::cout << "[CONFIG] Загружено размеров матриц:   " << sizes.size()
            << "\n\n";

  // 4. Подготовка CSV файла
  const std::string csv_filename = "benchmark_results.csv";
  std::ofstream csv(csv_filename);
  csv << "MatrixSize,Mode,Threads,Time_ms,GFLOPS\n";

  // Списки ядер для режимов 1-4
  auto mode1_cpus = topo.get_mode1_cpus();
  auto mode2_cpus = topo.get_mode2_cpus();
  auto mode3_cpus = topo.get_mode3_cpus();
  auto mode4_cpus = topo.get_mode4_cpus();

  // 5. Запуск тестов
  std::cout << std::left << std::setw(10) << "Size" << std::setw(28) << "Mode"
            << std::setw(10) << "Threads" << std::setw(14) << "Time (ms)"
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << std::string(74, '-') << "\n";

  for (int n : sizes) {
    // Режим 1: Single-core (SMT OFF)
    {
      BenchResult res = CpuBenchmark::run(n, mode1_cpus);
      std::cout << std::left << std::setw(10) << n << std::setw(28)
                << "Single-Core (SMT OFF)" << std::setw(10) << mode1_cpus.size()
                << std::setw(14) << std::fixed << std::setprecision(4)
                << res.elapsed_ms << std::setw(12) << std::fixed
                << std::setprecision(2) << res.gflops << "\n";
      csv << n << ",Single-Core (SMT OFF)," << mode1_cpus.size() << ","
          << res.elapsed_ms << "," << res.gflops << "\n";
    }

    // Режим 2: Single-core (SMT ON)
    {
      BenchResult res = CpuBenchmark::run(n, mode2_cpus);
      std::cout << std::left << std::setw(10) << n << std::setw(28)
                << "Single-Core (SMT ON)" << std::setw(10) << mode2_cpus.size()
                << std::setw(14) << std::fixed << std::setprecision(4)
                << res.elapsed_ms << std::setw(12) << std::fixed
                << std::setprecision(2) << res.gflops << "\n";
      csv << n << ",Single-Core (SMT ON)," << mode2_cpus.size() << ","
          << res.elapsed_ms << "," << res.gflops << "\n";
    }

    // Режим 3: Multi-core (SMT OFF)
    {
      BenchResult res = CpuBenchmark::run(n, mode3_cpus);
      std::cout << std::left << std::setw(10) << n << std::setw(28)
                << "Multi-Core (SMT OFF)" << std::setw(10) << mode3_cpus.size()
                << std::setw(14) << std::fixed << std::setprecision(4)
                << res.elapsed_ms << std::setw(12) << std::fixed
                << std::setprecision(2) << res.gflops << "\n";
      csv << n << ",Multi-Core (SMT OFF)," << mode3_cpus.size() << ","
          << res.elapsed_ms << "," << res.gflops << "\n";
    }

    // Режим 4: Multi-core (SMT ON)
    {
      BenchResult res = CpuBenchmark::run(n, mode4_cpus);
      std::cout << std::left << std::setw(10) << n << std::setw(28)
                << "Multi-Core (SMT ON)" << std::setw(10) << mode4_cpus.size()
                << std::setw(14) << std::fixed << std::setprecision(4)
                << res.elapsed_ms << std::setw(12) << std::fixed
                << std::setprecision(2) << res.gflops << "\n";
      csv << n << ",Multi-Core (SMT ON)," << mode4_cpus.size() << ","
          << res.elapsed_ms << "," << res.gflops << "\n";
    }

    // Режим 5: iGPU Compute
    if (gpu_bench.available) {
      BenchResult res = gpu_bench.run(n);
      std::cout << std::left << std::setw(10) << n << std::setw(28)
                << "iGPU Compute (OpenCL)" << std::setw(10) << "GPU"
                << std::setw(14) << std::fixed << std::setprecision(4)
                << res.elapsed_ms << std::setw(12) << std::fixed
                << std::setprecision(2) << res.gflops << "\n";
      csv << n << ",iGPU Compute (OpenCL),GPU," << res.elapsed_ms << ","
          << res.gflops << "\n";
    }

    csv.flush();
    std::cout << std::string(74, '-') << "\n";
  }

  csv.close();
  std::cout << "\nБенчмарк успешно завершен. Результаты сохранены в: "
            << csv_filename << "\n";
  return 0;
}