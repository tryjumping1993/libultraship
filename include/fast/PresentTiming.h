//
//  PresentTiming.h
//  libultraship
//
//  When frames actually reach the screen, for renderers that can find out
//  (Vulkan with VK_GOOGLE_display_timing or VK_KHR_present_wait). A game that
//  paces itself to the display registers a callback here and gets one call per
//  presented frame, in present order, usually a frame or more after the present.
//  Renderers that cannot tell never call it, so the game keeps measuring the
//  present calls itself.
//
#pragma once

#include <cstdint>

// Lets a game test for this interface with the preprocessor.
#define LUS_HAS_PRESENT_TIMING 1

namespace Fast {

// steadyClockNs: std::chrono::steady_clock time, in nanoseconds since that
// clock's epoch, at which the frame began to show. Called on the render thread.
using PresentTimeCallback = void (*)(int64_t steadyClockNs);

void SetPresentTimeCallback(PresentTimeCallback callback);

// For renderers: true while someone listens, so timing work can be skipped otherwise.
bool HasPresentTimeCallback();
void ReportPresentTime(int64_t steadyClockNs);

} // namespace Fast
