<!-- mdformat off(b/169948621#comment2) -->
<!--ts-->
* [Offline Memory Plan](#offline-memory-plan)
   * [Background and Motivation](#background-and-motivation)
   * [Usage](#usage)

<!-- Added by: deqiangc, at: Tue 28 Sep 2021 02:36:28 PM PDT -->

<!--te-->

# Offline Memory Plan Via NonpersistentMemoryPlannerShim

This doc outline how to use the NonPersistentMemoryPlannerShim class to work
with a external tooling that can plan the offset of each non persistent buffer
for the Model within the TFLM arena.

This is an experimental feature right now and subjected to change. Comments are
welcome!

## Background and Motivation

The
[(memory management page)](memory_management.md#offline-planned-tensor-allocations)
describes a way to specify the offset of each non persistent buffer in a
flatbuffer model file. This document describe an alternative that allows the
offset of each non persistent buffer for the Model within the TFLM arena to be
specified by a C++ struct. The approach in this document is an early stage
exploration of what the next version of offline memory planning in TFLM might
look like.

If the NonPersistentMemoryPlannerShim is used, then the final binary does not
have any of the symbols associated with the GreedyMemoryPlanner which results in
a reduced memory footprint.

Additionally, the offline planning of the non-persistent buffers can be used to
have a more efficient utilization compared to the GreedyMemoryPlanner.

## Usage

The more effecient memory plan above can be represented by the following C++
struct

```cc
const struct BufferPlan kOfflineNonPersistentBufferPlan = {
  .buffer_count = 9,
  .buffer_plan_entries = {
    [0] = { .offset = 0 },
    [1] = { .offset = 400 },
    [2] = { .offset = 801 },
    [3] = { .offset = 400 },
    [4] = { .offset = 811 },
    [5] = { .offset = 601 },
    [6] = { .offset = 814 },
    [7] = { .offset = 601 },
    [8] = { .offset = 801 },
   }
};
```

Then you can create a NonPersistentBufferPlannerShim and provide it to the
Interpreter such as below

```cc
// The arena includes both persistent buffers and non-persistent buffers. 
constexpr int kArenaSize = 2*1048;
uint8_t tensor_arena[kArenaSize];

tflite::NonPersistentMemoryPlannerShim planner(&kOfflineNonPersistentBufferPlan);

tflite::MicroAllocator * allocator = tflite::MicroAllocator::Create(
  tensor_arena, arena_size, &planner);

tflite::MicroInterpreter interpreter(model, op_resolver, allocator);
```

