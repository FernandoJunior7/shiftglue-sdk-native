#pragma once

#include <cstdint>
#include <memory>

#include <rex/ui/vulkan/api.h>

namespace rex::ui::vulkan {

class VulkanDevice;

class MetalFxUpscaler {
 public:
  static std::unique_ptr<MetalFxUpscaler> Create(const VulkanDevice* vulkan_device);
  ~MetalFxUpscaler();

  bool Encode(VkImage source, uint32_t source_width, uint32_t source_height, VkImage output,
              uint32_t output_width, uint32_t output_height);
  void Commit();

 private:
  struct State;
  explicit MetalFxUpscaler(std::unique_ptr<State> state);
  bool EncodeInPool(VkImage source, uint32_t source_width, uint32_t source_height, VkImage output,
                    uint32_t output_width, uint32_t output_height);
  std::unique_ptr<State> state_;
};

}  // namespace rex::ui::vulkan
