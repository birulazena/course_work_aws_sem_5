#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct PhysicalCore {
  int core_id;
  std::vector<int> logical_cpus;
  unsigned long max_freq_khz = 0;
};

class CpuTopology {
 public:
  std::vector<PhysicalCore> cores;
  int total_logical_cpus = 0;

  static CpuTopology detect() {
    CpuTopology topo;
    topo.total_logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (topo.total_logical_cpus <= 0) {
      topo.total_logical_cpus = 1;
    }

    std::set<int> visited_cpus;

    for (int cpu = 0; cpu < topo.total_logical_cpus; ++cpu) {
      if (visited_cpus.count(cpu)) {
        continue;
      }

      std::string sys_path =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu);

      // Читаем список потоков-собратьев (thread siblings)
      std::string siblings_file = sys_path + "/topology/thread_siblings_list";
      std::ifstream f_sib(siblings_file);
      if (!f_sib.is_open()) {
        // В ядре 6.x путь может быть core_cpus_list
        siblings_file = sys_path + "/topology/core_cpus_list";
        f_sib.open(siblings_file);
      }

      std::vector<int> siblings;
      if (f_sib.is_open()) {
        std::string line;
        std::getline(f_sib, line);
        siblings = parse_cpu_list(line);
      } else {
        siblings.push_back(cpu);
      }

      // Читаем core_id
      int core_id = cpu;
      std::ifstream f_cid(sys_path + "/topology/core_id");
      if (f_cid.is_open()) {
        f_cid >> core_id;
      }

      // Читаем максимальную частоту для различения P/E-ядер
      unsigned long max_freq = 0;
      std::ifstream f_freq(sys_path + "/cpufreq/cpuinfo_max_freq");
      if (f_freq.is_open()) {
        f_freq >> max_freq;
      }

      PhysicalCore pcore;
      pcore.core_id = core_id;
      pcore.logical_cpus = siblings;
      pcore.max_freq_khz = max_freq;

      for (int id : siblings) {
        visited_cpus.insert(id);
      }

      topo.cores.push_back(pcore);
    }

    // Если sysfs недоступен, фоллбэк на плоскую структуру
    if (topo.cores.empty()) {
      for (int i = 0; i < topo.total_logical_cpus; ++i) {
        PhysicalCore p;
        p.core_id = i;
        p.logical_cpus = {i};
        topo.cores.push_back(p);
      }
    }

    // Сортировка: производительные ядра (с SMT и высокой частотой) идут
    // первыми. Это гарантирует выбор P-core на Intel i7-12700H и валидного ядра
    // на AMD Ryzen.
    std::sort(topo.cores.begin(), topo.cores.end(),
              [](const PhysicalCore& a, const PhysicalCore& b) {
                if (a.logical_cpus.size() != b.logical_cpus.size()) {
                  return a.logical_cpus.size() > b.logical_cpus.size();
                }
                return a.max_freq_khz > b.max_freq_khz;
              });

    return topo;
  }

  // Режим 1: 1 поток на 1 физическое ядро (P-core на Intel)
  std::vector<int> get_mode1_cpus() const { return {cores[0].logical_cpus[0]}; }

  // Режим 2: 2 потока на то же физическое ядро (оба его SMT потока)
  std::vector<int> get_mode2_cpus() const {
    if (cores[0].logical_cpus.size() >= 2) {
      return {cores[0].logical_cpus[0], cores[0].logical_cpus[1]};
    }
    // Фоллбэк, если SMT аппаратно отключен
    return {cores[0].logical_cpus[0]};
  }

  // Режим 3: строго по 1 потоку на каждое физическое ядро (SMT OFF)
  std::vector<int> get_mode3_cpus() const {
    std::vector<int> cpus;
    cpus.reserve(cores.size());
    for (const auto& c : cores) {
      cpus.push_back(c.logical_cpus[0]);
    }
    return cpus;
  }

  // Режим 4: все логические ядра системы (все физические + гиперпотоки)
  std::vector<int> get_mode4_cpus() const {
    std::vector<int> cpus;
    cpus.reserve(total_logical_cpus);
    for (const auto& c : cores) {
      for (int id : c.logical_cpus) {
        cpus.push_back(id);
      }
    }
    return cpus;
  }

  static bool pin_current_thread(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) ==
           0;
  }

 private:
  static std::vector<int> parse_cpu_list(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (token.empty()) continue;
      auto dash_pos = token.find('-');
      if (dash_pos != std::string::npos) {
        int start = std::stoi(token.substr(0, dash_pos));
        int end = std::stoi(token.substr(dash_pos + 1));
        for (int i = start; i <= end; ++i) {
          result.push_back(i);
        }
      } else {
        result.push_back(std::stoi(token));
      }
    }
    return result;
  }
};