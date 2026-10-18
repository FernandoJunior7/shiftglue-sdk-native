#pragma once

#include <cstdint>
#include <string>

namespace rex::graphics::vulkan {

bool StartMetalGpuTiming(bool encoder_profile);
uint64_t MetalGpuBusyNs();
std::string TakeMetalPresentationStats();
std::string TakeMetalEncoderProfile(uint64_t frames, size_t top);

}  // namespace rex::graphics::vulkan
