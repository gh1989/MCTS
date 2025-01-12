#ifndef GPU_H_
#define GPU_H_

#include <torch/torch.h>
#include <cuda_runtime.h>

void LogGPUStats() {
    if (torch::cuda::is_available()) {
        for (int64_t device = 0; device < torch::cuda::device_count(); ++device) {
            // Get basic device properties
            cudaDeviceProp properties;
            cudaGetDeviceProperties(&properties, device);
            
            // Get current memory usage
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            
            Logger::Log(LogLevel::INFO, "GPU " + std::to_string(device) + " (" + properties.name + "):");
            Logger::Log(LogLevel::INFO, "  - Total Memory: " + 
                std::to_string(total_memory / 1024 / 1024) + " MB");
            Logger::Log(LogLevel::INFO, "  - Free Memory: " + 
                std::to_string(free_memory / 1024 / 1024) + " MB");
            Logger::Log(LogLevel::INFO, "  - Used Memory: " + 
                std::to_string((total_memory - free_memory) / 1024 / 1024) + " MB");
        }
    } else {
        Logger::Log(LogLevel::INFO, "No GPU available, running on CPU");
    }
}

#endif  // GPU_H_