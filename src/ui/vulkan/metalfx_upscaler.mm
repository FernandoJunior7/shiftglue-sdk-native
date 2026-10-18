#include <rex/ui/vulkan/metalfx_upscaler.h>

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <rex/logging.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/instance.h>

namespace rex::ui::vulkan {

struct MetalFxUpscaler::State {
  const VulkanDevice* vulkan_device = nullptr;
  PFN_vkExportMetalObjectsEXT export_objects = nullptr;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLFXSpatialScaler> scaler = nil;
  id<MTLCommandBuffer> pending = nil;
  NSUInteger input_width = 0, input_height = 0, output_width = 0, output_height = 0;
  MTLPixelFormat input_format = MTLPixelFormatInvalid, output_format = MTLPixelFormatInvalid;

  id<MTLTexture> Texture(VkImage image) const {
    VkExportMetalTextureInfoEXT texture = {};
    texture.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT;
    texture.image = image;
    texture.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT objects = {};
    objects.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    objects.pNext = &texture;
    export_objects(vulkan_device->device(), &objects);
    return texture.mtlTexture;
  }
};

MetalFxUpscaler::MetalFxUpscaler(std::unique_ptr<State> state) : state_(std::move(state)) {}

MetalFxUpscaler::~MetalFxUpscaler() {
  [state_->pending release];
  [state_->scaler release];
  [state_->queue release];
  [state_->device release];
}

std::unique_ptr<MetalFxUpscaler> MetalFxUpscaler::Create(const VulkanDevice* vulkan_device) {
  if (!vulkan_device->extensions().ext_EXT_metal_objects) return nullptr;
  auto state = std::make_unique<State>();
  state->vulkan_device = vulkan_device;
  state->export_objects = PFN_vkExportMetalObjectsEXT(
      vulkan_device->vulkan_instance()->functions().vkGetDeviceProcAddr(
          vulkan_device->device(), "vkExportMetalObjectsEXT"));
  if (!state->export_objects) return nullptr;
  VkExportMetalCommandQueueInfoEXT queue = {};
  queue.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT;
  {
    const VulkanDevice::Queue::Acquisition acquisition =
        vulkan_device->AcquireQueue(vulkan_device->queue_family_graphics_compute(), 0);
    queue.queue = acquisition.queue();
  }
  VkExportMetalDeviceInfoEXT device = {};
  device.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT;
  device.pNext = &queue;
  VkExportMetalObjectsInfoEXT objects = {};
  objects.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
  objects.pNext = &device;
  state->export_objects(vulkan_device->device(), &objects);
  state->device = [device.mtlDevice retain];
  state->queue = [queue.mtlCommandQueue retain];
  if (!state->device || !state->queue ||
      ![MTLFXSpatialScalerDescriptor supportsDevice:state->device]) {
    [state->queue release];
    [state->device release];
    return nullptr;
  }
  @autoreleasepool {
    REXLOG_INFO("VulkanPresenter: MetalFX spatial upscaling available on {}",
                state->device.name.UTF8String);
  }
  return std::unique_ptr<MetalFxUpscaler>(new MetalFxUpscaler(std::move(state)));
}

bool MetalFxUpscaler::Encode(VkImage source, uint32_t source_width, uint32_t source_height,
                             VkImage output, uint32_t output_width, uint32_t output_height) {
  @autoreleasepool {
    return EncodeInPool(source, source_width, source_height, output, output_width,
                        output_height);
  }
}

bool MetalFxUpscaler::EncodeInPool(VkImage source, uint32_t source_width,
                                   uint32_t source_height, VkImage output,
                                   uint32_t output_width, uint32_t output_height) {
  State& state = *state_;
  id<MTLTexture> input = state.Texture(source);
  id<MTLTexture> target = state.Texture(output);
  if (!input || !target) return false;
  if (!state.scaler || state.input_width != input.width ||
      state.input_height != input.height || state.output_width != output_width ||
      state.output_height != output_height || state.input_format != input.pixelFormat ||
      state.output_format != target.pixelFormat) {
    MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
    descriptor.inputWidth = input.width;
    descriptor.inputHeight = input.height;
    descriptor.outputWidth = output_width;
    descriptor.outputHeight = output_height;
    descriptor.colorTextureFormat = input.pixelFormat;
    descriptor.outputTextureFormat = target.pixelFormat;
    descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
    [state.scaler release];
    state.scaler = [descriptor newSpatialScalerWithDevice:state.device];
    [descriptor release];
    if (!state.scaler) {
      REXLOG_ERROR("VulkanPresenter: MetalFX could not scale {}x{} to {}x{}", source_width,
                   source_height, output_width, output_height);
      return false;
    }
    state.input_width = input.width;
    state.input_height = input.height;
    state.output_width = output_width;
    state.output_height = output_height;
    state.input_format = input.pixelFormat;
    state.output_format = target.pixelFormat;
  }
  state.scaler.colorTexture = input;
  state.scaler.outputTexture = target;
  state.scaler.inputContentWidth = source_width;
  state.scaler.inputContentHeight = source_height;
  if (!state.pending) state.pending = [[state.queue commandBuffer] retain];
  [state.scaler encodeToCommandBuffer:state.pending];
  return true;
}

void MetalFxUpscaler::Commit() {
  if (!state_->pending) return;
  [state_->pending commit];
  [state_->pending release];
  state_->pending = nil;
}

}  // namespace rex::ui::vulkan
