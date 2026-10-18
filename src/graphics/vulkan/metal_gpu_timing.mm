#include <rex/graphics/vulkan/metal_gpu_timing.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#import <Metal/Metal.h>
#include <objc/runtime.h>

#include <fmt/format.h>
#include <rex/logging.h>
#include <rex/perf/counter.h>

namespace rex::graphics::vulkan {

namespace {

std::atomic<uint64_t> busy_ns{0};
std::mutex busy_mutex;
double busy_end = 0;
bool encoder_profile = false;
id<MTLCounterSampleBuffer> sample_buffer = nil;
NSUInteger sample_count = 0;
std::atomic<uint64_t> next_sample{0};
double ns_per_tick = 1.0;

IMP original_commit = nullptr;
IMP original_render = nullptr;
IMP original_compute = nullptr;
IMP original_blit = nullptr;

struct EncoderSamples {
  id encoder;
  uint32_t first_sample;
  bool render;
  std::string key;
};
struct EncoderStat {
  uint64_t ns = 0;
  uint64_t fragment_ns = 0;
  uint64_t count = 0;
};
std::mutex encoders_mutex;
std::unordered_map<void*, std::vector<EncoderSamples>> pending_encoders;
std::map<std::string, EncoderStat> encoder_stats;

uint32_t AllocateSamples(uint32_t count) {
  return uint32_t(next_sample.fetch_add(count, std::memory_order_relaxed) % (sample_count / 4)) * 4;
}

void Track(id buffer, id encoder, uint32_t first_sample, bool render, std::string key) {
  std::lock_guard<std::mutex> lock(encoders_mutex);
  pending_encoders[(__bridge void*)buffer].push_back(
      {encoder, first_sample, render, std::move(key)});
}

std::string RenderPassKey(MTLRenderPassDescriptor* descriptor) {
  std::string key;
  auto attachment = [&](MTLRenderPassAttachmentDescriptor* a, const char* name) {
    if (!a.texture) return;
    if (key.empty()) {
      key = fmt::format("{}x{}", a.texture.width, a.texture.height);
    }
    key += fmt::format(" {}{}:{}{}", name, uint32_t(a.texture.pixelFormat),
                       "DLC"[std::min<uint32_t>(uint32_t(a.loadAction), 2)],
                       "DSRRU?"[std::min<uint32_t>(uint32_t(a.storeAction), 5)]);
  };
  for (uint32_t i = 0; i < 8; ++i) {
    attachment(descriptor.colorAttachments[i], "c");
  }
  attachment(descriptor.depthAttachment, "d");
  if (descriptor.stencilAttachment.texture != descriptor.depthAttachment.texture) {
    attachment(descriptor.stencilAttachment, "s");
  }
  return key.empty() ? std::string("no attachments") : key;
}

void Commit(id buffer, SEL selector) {
  std::shared_ptr<std::vector<EncoderSamples>> encoders;
  if (encoder_profile) {
    std::lock_guard<std::mutex> lock(encoders_mutex);
    auto it = pending_encoders.find((__bridge void*)buffer);
    if (it != pending_encoders.end()) {
      encoders = std::make_shared<std::vector<EncoderSamples>>(std::move(it->second));
      pending_encoders.erase(it);
    }
  }
  [(id<MTLCommandBuffer>)buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    const double start = completed.GPUStartTime, end = completed.GPUEndTime;
    if (start > 0 && end > start) {
      std::lock_guard<std::mutex> lock(busy_mutex);
      const double from = std::max(start, busy_end);
      if (end > from) {
        const uint64_t ns = uint64_t((end - from) * 1e9);
        busy_ns.fetch_add(ns, std::memory_order_relaxed);
        PERF_counter_add(kMetalGpuBusyNs, int64_t(ns));
      }
      busy_end = std::max(busy_end, end);
    }
    if (!encoders || encoders->empty()) return;
    std::vector<std::tuple<std::string, uint64_t, uint64_t>> results;
    for (EncoderSamples& samples : *encoders) {
      NSData* data =
          [sample_buffer resolveCounterRange:NSMakeRange(samples.first_sample, 4)];
      if (!data || data.length < 4 * sizeof(MTLCounterResultTimestamp)) continue;
      const auto* t = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
      auto valid = [](uint64_t v) { return v && v != MTLCounterErrorValue; };
      uint64_t first = UINT64_MAX, last = 0, fragment = 0;
      const uint32_t used = samples.render ? 4 : 2;
      for (uint32_t i = 0; i < used; ++i) {
        if (!valid(t[i].timestamp)) continue;
        first = std::min(first, t[i].timestamp);
        last = std::max(last, t[i].timestamp);
      }
      if (samples.render && valid(t[2].timestamp) && valid(t[3].timestamp) &&
          t[3].timestamp > t[2].timestamp) {
        fragment = t[3].timestamp - t[2].timestamp;
      }
      std::string key = samples.render ? "R " + samples.key : samples.key;
      NSString* label = [samples.encoder label];
      if (!samples.render && label) key += std::string(" ") + label.UTF8String;
      results.emplace_back(std::move(key), last > first ? last - first : 0, fragment);
    }
    std::lock_guard<std::mutex> lock(encoders_mutex);
    for (auto& [key, ticks, fragment] : results) {
      EncoderStat& stat = encoder_stats[key];
      stat.ns += uint64_t(double(ticks) * ns_per_tick);
      stat.fragment_ns += uint64_t(double(fragment) * ns_per_tick);
      ++stat.count;
    }
  }];
  reinterpret_cast<void (*)(id, SEL)>(original_commit)(buffer, selector);
}

id RenderEncoder(id buffer, SEL selector, MTLRenderPassDescriptor* descriptor) {
  const uint32_t first = AllocateSamples(4);
  MTLRenderPassSampleBufferAttachmentDescriptor* attachment =
      descriptor.sampleBufferAttachments[0];
  attachment.sampleBuffer = sample_buffer;
  attachment.startOfVertexSampleIndex = first;
  attachment.endOfVertexSampleIndex = first + 1;
  attachment.startOfFragmentSampleIndex = first + 2;
  attachment.endOfFragmentSampleIndex = first + 3;
  id encoder =
      reinterpret_cast<id (*)(id, SEL, MTLRenderPassDescriptor*)>(original_render)(
          buffer, selector, descriptor);
  if (encoder) Track(buffer, encoder, first, true, RenderPassKey(descriptor));
  return encoder;
}

id ComputeEncoder(id buffer, SEL, MTLDispatchType type) {
  const uint32_t first = AllocateSamples(2);
  MTLComputePassDescriptor* descriptor = [MTLComputePassDescriptor computePassDescriptor];
  descriptor.dispatchType = type;
  descriptor.sampleBufferAttachments[0].sampleBuffer = sample_buffer;
  descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = first;
  descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = first + 1;
  id encoder = [(id<MTLCommandBuffer>)buffer computeCommandEncoderWithDescriptor:descriptor];
  if (encoder) Track(buffer, encoder, first, false, "C");
  return encoder;
}

id BlitEncoder(id buffer, SEL) {
  const uint32_t first = AllocateSamples(2);
  MTLBlitPassDescriptor* descriptor = [MTLBlitPassDescriptor blitPassDescriptor];
  descriptor.sampleBufferAttachments[0].sampleBuffer = sample_buffer;
  descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = first;
  descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = first + 1;
  id encoder = [(id<MTLCommandBuffer>)buffer blitCommandEncoderWithDescriptor:descriptor];
  if (encoder) Track(buffer, encoder, first, false, "B");
  return encoder;
}

IMP Replace(Class buffer_class, SEL selector, IMP replacement) {
  Method method = class_getInstanceMethod(buffer_class, selector);
  if (!method) return nullptr;
  IMP original = method_getImplementation(method);
  class_replaceMethod(buffer_class, selector, replacement, method_getTypeEncoding(method));
  return original;
}

bool StartEncoderProfile(id<MTLDevice> device, Class buffer_class) {
  if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return false;
  id<MTLCounterSet> timestamps = nil;
  for (id<MTLCounterSet> set in device.counterSets) {
    if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
  }
  if (!timestamps) return false;
  for (NSUInteger count : {NSUInteger(32768), NSUInteger(4096)}) {
    MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = timestamps;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = count;
    sample_buffer = [device newCounterSampleBufferWithDescriptor:descriptor error:nil];
    if (sample_buffer) {
      sample_count = count;
      break;
    }
  }
  if (!sample_buffer) return false;
  MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
  const auto wall0 = std::chrono::steady_clock::now();
  [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
  const double wall =
      double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - wall0)
                 .count());
  if (gpu1 > gpu0) ns_per_tick = wall / double(gpu1 - gpu0);
  original_render = Replace(buffer_class, @selector(renderCommandEncoderWithDescriptor:),
                            reinterpret_cast<IMP>(RenderEncoder));
  original_compute = Replace(buffer_class, @selector(computeCommandEncoderWithDispatchType:),
                             reinterpret_cast<IMP>(ComputeEncoder));
  original_blit =
      Replace(buffer_class, @selector(blitCommandEncoder), reinterpret_cast<IMP>(BlitEncoder));
  encoder_profile = true;
  return true;
}

}  // namespace

bool StartMetalGpuTiming(bool profile_encoders) {
  static std::once_flag once;
  static bool started = false;
  std::call_once(once, [&] {
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> buffer = [queue commandBuffer];
      if (!buffer) return;
      Class buffer_class = object_getClass(buffer);
      original_commit = Replace(buffer_class, @selector(commit), reinterpret_cast<IMP>(Commit));
      if (!original_commit) return;
      started = true;
      const bool profiling = profile_encoders && StartEncoderProfile(device, buffer_class);
      REXGPU_INFO("Metal GPU timing on {}{}", class_getName(buffer_class),
                  profiling ? fmt::format(", encoder profile with {} samples, {:.3f} ns per tick",
                                          sample_count, ns_per_tick)
                            : std::string());
    }
  });
  return started;
}

uint64_t MetalGpuBusyNs() { return busy_ns.load(std::memory_order_relaxed); }

std::string TakeMetalEncoderProfile(uint64_t frames, size_t top) {
  std::map<std::string, EncoderStat> stats;
  {
    std::lock_guard<std::mutex> lock(encoders_mutex);
    stats.swap(encoder_stats);
  }
  if (stats.empty() || !frames) return {};
  std::vector<std::pair<uint64_t, const std::string*>> order;
  uint64_t render = 0, other = 0, encoders = 0;
  for (const auto& [key, stat] : stats) {
    order.emplace_back(stat.ns, &key);
    (key[0] == 'R' ? render : other) += stat.ns;
    encoders += stat.count;
  }
  std::sort(order.begin(), order.end(), std::greater<>());
  const double per_frame = 1.0 / double(frames);
  std::string text = fmt::format(
      "render {:.2f} ms, compute and blit {:.2f} ms, {:.1f} encoders;",
      double(render) / 1e6 * per_frame, double(other) / 1e6 * per_frame,
      double(encoders) * per_frame);
  for (size_t i = 0; i < order.size() && i < top; ++i) {
    const EncoderStat& stat = stats[*order[i].second];
    text += fmt::format(" [{}] {:.3f} ms (fragment {:.3f}) x{:.1f};", *order[i].second,
                        double(stat.ns) / 1e6 * per_frame,
                        double(stat.fragment_ns) / 1e6 * per_frame, double(stat.count) * per_frame);
  }
  return text;
}

}  // namespace rex::graphics::vulkan
