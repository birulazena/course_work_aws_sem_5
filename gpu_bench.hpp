#pragma once

#define CL_SILENCE_DEPRECATION
#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_bench.hpp"

class GpuBenchmark {
 public:
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  std::string device_name;
  bool available = false;

  GpuBenchmark() { init_opencl(); }

  ~GpuBenchmark() { cleanup(); }

  BenchResult run(int n) {
    if (!available) {
      return {0.0, 0.0, 0.0f};
    }

    Matrix A(n);
    Matrix B(n);
    Matrix C(n);

    A.init_a();
    B.init_b();
    C.zero();

    size_t bytes = sizeof(float) * static_cast<size_t>(n) * n;
    cl_int err;

    cl_mem d_a = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data, &err);
    cl_mem d_b = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data, &err);
    cl_mem d_c =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);

    // Локальный размер рабочей группы 16x16
    const size_t local_size[2] = {16, 16};
    const size_t total_n = ((static_cast<size_t>(n) + 15) / 16) * 16;

    // Максимальное количество строк за один dispatch (кратно 16).
    // 1024 строки гарантируют, что ни один запуск не превысит watchdog-таймаут
    // amdgpu.
    const size_t CHUNK_ROWS = 1024;

    // Вспомогательная лямбда для запуска вычислений по порциям
    auto dispatch_all = [&]() {
      for (size_t row_offset = 0; row_offset < total_n;
           row_offset += CHUNK_ROWS) {
        size_t current_chunk = std::min(CHUNK_ROWS, total_n - row_offset);

        size_t global_offset[2] = {0, row_offset};
        size_t global_size[2] = {total_n, current_chunk};

        clEnqueueNDRangeKernel(queue, kernel, 2, global_offset, global_size,
                               local_size, 0, nullptr, nullptr);
      }
    };

    // Прогревочный прогон
    if (n < 4096) {
      dispatch_all();
      clFinish(queue);
    }

    const int iters = CpuBenchmark::get_iterations(n);

    // Замер времени чистого выполнения на GPU
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      dispatch_all();
    }
    clFinish(queue);
    auto t_end = std::chrono::high_resolution_clock::now();

    // Чтение результата
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, C.data, 0, nullptr,
                        nullptr);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);

    std::chrono::duration<double, std::milli> diff = t_end - t_start;
    double elapsed_ms = diff.count();
    double total_ops = 2.0 * static_cast<double>(n) * n * n * iters;
    double gflops = (total_ops / (elapsed_ms * 1e-3)) / 1e9;

    return {elapsed_ms / iters, gflops, C.checksum()};
  }

 private:
  void init_opencl() {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // Поиск первого доступного GPU устройства
    for (auto p : platforms) {
      cl_uint num_devices = 0;
      if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) ==
              CL_SUCCESS &&
          num_devices > 0) {
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devices.data(),
                       nullptr);
        device = devices[0];
        break;
      }
    }

    if (!device) return;

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    device_name = std::string(name);

    cl_int err;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return;

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) return;

    // Загрузка исходного кода кернела
    std::ifstream k_file("cl_kernel.cl");
    if (!k_file.is_open()) return;

    std::stringstream ss;
    ss << k_file.rdbuf();
    std::string src = ss.str();
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) return;

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr,
                         nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
      std::cerr << "OpenCL Build Error:\n" << log.data() << std::endl;
      return;
    }

    kernel = clCreateKernel(program, "matmul", &err);
    if (err == CL_SUCCESS) {
      available = true;
    }
  }

  void cleanup() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};