#define TILE_SIZE 16

__kernel void matmul(
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    // Идентификаторы потока внутри рабочей группы
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    // Глобальные координаты элемента матрицы C
    const int global_col = get_global_id(0);
    const int global_row = get_global_id(1);

    // Локальная разделяемая память для тайлов матриц A и B
    __local float a_tile[TILE_SIZE][TILE_SIZE];
    __local float b_tile[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col = t * TILE_SIZE + local_col;
        int tiled_row = t * TILE_SIZE + local_row;

        // Копирование тайла A из глобальной памяти в локальную
        if (global_row < N && tiled_col < N) {
            a_tile[local_row][local_col] = A[global_row * N + tiled_col];
        } else {
            a_tile[local_row][local_col] = 0.0f;
        }

        // Копирование тайла B из глобальной памяти в локальную
        if (tiled_row < N && global_col < N) {
            b_tile[local_row][local_col] = B[tiled_row * N + global_col];
        } else {
            b_tile[local_row][local_col] = 0.0f;
        }

        // Ожидание завершения загрузки тайлов всеми потоками группы
        barrier(CLK_LOCAL_MEM_FENCE);

        // Накопление частичного произведения по загруженному тайлу
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += a_tile[local_row][k] * b_tile[k][local_col];
        }

        // Ожидание завершения вычислений перед загрузкой следующего тайла
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Запись итогового значения в глобальную память
    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = acc;
    }
}