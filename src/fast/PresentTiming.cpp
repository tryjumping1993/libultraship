#include "fast/PresentTiming.h"

#include <atomic>

namespace Fast {

static std::atomic<PresentTimeCallback> sPresentTimeCallback{ nullptr };

void SetPresentTimeCallback(PresentTimeCallback callback) {
    sPresentTimeCallback.store(callback, std::memory_order_release);
}

bool HasPresentTimeCallback() {
    return sPresentTimeCallback.load(std::memory_order_acquire) != nullptr;
}

void ReportPresentTime(int64_t steadyClockNs) {
    PresentTimeCallback callback = sPresentTimeCallback.load(std::memory_order_acquire);
    if (callback != nullptr) {
        callback(steadyClockNs);
    }
}

} // namespace Fast
