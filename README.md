# 🚀 Matrix Multiplication Hardware Benchmark (CPU vs iGPU)

Сравнительный бенчмарк алгоритма матричного умножения ($N \times N$, тип `float`) для двух мобильных платформ на базе Linux Ubuntu:

- **AMD Ryzen 7 7840HS** (8 ядер / 16 потоков, iGPU Radeon 780M, Ubuntu 26).
- **Intel Core i7-12700H** (6 P-ядер + 8 E-ядер / 20 потоков, iGPU Iris Xe, Ubuntu 24).

Программа последовательно тестирует вычислительную подсистему и фиксирует переходы между уровнями памяти: **L1 Cache $\to$ L2 Cache $\to$ L3 Cache $\to$ DDR5 RAM**.

---

## 📋 5 Режимов тестирования

1. **Single-Core (SMT OFF):** 1 поток, жестко зафиксированный на одном физическом ядре (на Intel — строго на P-core).
2. **Single-Core (SMT ON):** 2 потока, жестко привязанные к двум логическим потокам одного и того же физического ядра.
3. **Multi-Core (SMT OFF):** Потоки строго по числу физических ядер (1 поток на каждое физическое ядро, гиперпотоки отключены).
4. **Multi-Core (SMT ON):** Все доступные логические процессоры (все ядра + все виртуальные потоки).
5. **iGPU Compute (OpenCL):** Вычисления на встроенном графическом ядре с блочным кэшированием в локальной памяти (LDS/Shared Memory).

---

## 🛠️ 1. Установка зависимостей

### 1.1. Базовые библиотеки и компилятор (на обеих системах)

```bash
sudo apt update
sudo apt install -y build-essential ocl-icd-opencl-dev opencl-headers clinfo python3-pandas python3-matplotlib
```

### 1.2. Конфигурация для Intel Core i7-12700H (Iris Xe, Ubuntu 24)

Установка официального OpenCL runtime драйвера:

```bash
sudo apt install -y intel-opencl-icd
```

_Проверка распознавания iGPU:_

```bash
clinfo | grep "Device Name"
```

### 1.3. Конфигурация для AMD Ryzen 7 7840HS (Radeon 780M, Ubuntu 26)

Установка OpenCL драйвера Mesa RustiCL:

```bash
sudo apt install -y mesa-opencl-icd
```

_Проверка доступности iGPU:_

```bash
RUSTICL_ENABLE=radeonsi clinfo | grep "Device Name"
```

---

## ⚙️ 2. Сборка проекта

Сборка осуществляется компилятором `g++` с флагами глубокой оптимизации `-O3`, архитектурной векторизации `-march=native` (AVX2 + FMA) и линковкой OpenCL:

```bash
g++ -O3 -march=native -pthread main.cpp -lOpenCL -o benchmark
```

---

## 🏎️ 3. Запуск стандартного тестирования

Убедитесь, что в рабочей папке присутствуют конфигурационный файл `sizes.cfg` и файл с ядром `cl_kernel.cl`.

- **На платформе Intel (Ubuntu 24):**

  ```bash
  ./benchmark
  ```

- **На платформе AMD (Ubuntu 26):**
  ```bash
  RUSTICL_ENABLE=radeonsi ./benchmark
  ```

---

## 📊 5. Построение отчетов и графиков

После выполнения тестов программа сформирует файл `benchmark_results.csv`. Для создания сводного сравнительного графика выполните:

```bash
python3 plot_results.py
```

Готовый файл графика высокого разрешения будет сохранен в:  
`benchmark_results.png`
