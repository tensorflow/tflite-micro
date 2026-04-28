# Error Handling & Defensive Programming Guide

Balancing defensiveness with the extreme constraints of microcontrollers (where
every byte of Flash/RAM and every clock cycle matters) is a core design tension
in TFLM. Since TFLM targets environments that often lack memory protection (no
MMU), don't support C++ exceptions, and run on bare metal or simple RTOSs, a
"standard" defensive posture would result in unacceptable binary bloat and
performance degradation.

This guide provides an architectural framework for writing safe kernels and
offers recommendations for code reviews for contributors.

--------------------------------------------------------------------------------

## 1. Trust Boundaries: Where Data Comes From

To achieve high performance without sacrificing critical stability, TFLM
distinguishes between trusted and untrusted data sources.

### The C++ API (Trusted)

When a firmware developer uses the public TFLM C++ API (e.g., instantiating a
`MicroInterpreter` or calling `Invoke()`), we treat the developer as trusted.

*   **Recommendation:** The core API should not spend runtime cycles or binary
    space checking for invalid parameters (e.g., passing `nullptr`) or incorrect
    state-machine sequences.
*   **Mechanism:** Rely entirely on `TFLITE_DCHECK`. The firmware engineer gets
    an immediate assertion failure during local debugging if they misuse the
    API. In release builds, these compile out, leaving zero overhead.

### The FlatBuffer Model (Partially Trusted)

When dealing with the `.tflite` FlatBuffer model, we distinguish between two
types of malformation:

*   **Corrupted FlatBuffer Files (Structural Malformation):** The core
    `MicroInterpreter` assumes the FlatBuffer is structurally valid. We *do not*
    perform standard bounds checking on FlatBuffer schema offsets or operator
    tensor indices (e.g., if an operator requests tensor index `999` but the
    model only has `10` tensors, a production TFLM build will read out of
    bounds). To aid local debugging, structural checks are considered a
    "good-to-have" feature but **should exclusively use `TFLITE_DCHECK`**. If
    models are provided via an untrusted channel (e.g., OTA), the application
    layer is entirely responsible for validating the integrity of the model
    before passing it to TFLM.
*   **Invalid Model Topologies (Semantic Malformation):** The runtime *should*
    defend against TFLite Converter bugs or unsupported configurations (e.g.,
    wrong tensor shapes, unsupported types). This validation happens entirely in
    the Setup phase. Because developers can use `TF_LITE_STRIP_ERROR_STRINGS` to
    remove the string bloat in production, checking topologies in `Prepare`
    provides safe fallback (e.g., rejecting a bad OTA update) without the severe
    ROM penalty of embedded strings.

--------------------------------------------------------------------------------

## 2. Execution Phases: When Code Runs

TFLM distinguishes between setup (which runs once) and execution (which runs
continuously).

### Phase 1: Setup & Initialization (`Prepare`)

During the `Prepare` phase, kernels should validate their inputs and parameters
to ensure `Eval` can run blindly and safely. We prioritize clear error messages
here (relying on `TF_LITE_STRIP_ERROR_STRINGS` to mitigate the ROM cost in
production) and we can afford to spend CPU cycles on validation.

**What to Validate (Use `TF_LITE_ENSURE`):**

*   **Model-provided parameters:** Check the number of inputs/outputs, tensor
    types, and tensor shapes.
*   **Quantization parameters:** Explicitly check for invalid quantization
    parameters (e.g., a `scale` of `0.0` or a `zero_point` out of bounds) that
    could cause a divide-by-zero or overflow later in `Eval`.
*   **Resource allocations:** Check if memory allocations (like
    `AllocateTempInputTensor`) return `nullptr`.

### Phase 2: Execution (`Eval`)

During the `Eval` phase, the runtime is vulnerable to data that causes hardware
traps or memory corruption. We should avoid spending cycles or ROM on redundant
checks.

**What to Defend Against (Recommended Validation):**

Kernels should actively defend against three specific runtime threats during
`Eval`:

1.  **Out-of-bounds Memory Access:** If an input tensor contains indices or
    offsets generated at runtime (e.g., in `GATHER`, `STRIDED_SLICE`), these
    **should** be bounds-checked at runtime using a raw `if (!valid) return
    kTfLiteError;`.
2.  **Hardware Faults (Divide by Zero):** Kernels should mathematically protect
    against divide-by-zero (e.g., if a divisor comes from an input tensor) or
    explicitly check and return an error.
3.  **Infinite Loops:** All loops inside `Eval` *should* have a guaranteed
    maximum bound to prevent data-dependent infinite hangs.

**What to Ignore (GIGO for Signal Data):**

*   TFLM kernels are **strongly discouraged** from explicitly scanning general
    input signal data (like images or audio) for `NaN`/`Inf`. For standard math
    operations, garbage data simply results in garbage output (GIGO).

--------------------------------------------------------------------------------

## 3. Macro Selection Guide

To strike the right balance between debuggability, code size (ROM), and
performance, follow this decision tree.

### The Decision Tree

1.  **Are you writing a Unit Test?**
    *   Use GoogleTest-style macros like `EXPECT_EQ`, `EXPECT_NEAR`, etc. for
        verifying test conditions.
2.  **Are you in `Init` or `Prepare`?**
    *   Use `TF_LITE_ENSURE`. We want to fail early with a clear log message if
        the model is incompatible.
3.  **Are you in `Eval` (or a helper called by `Eval`)?**
    *   *Is it an invariant guaranteed by `Prepare`?* (e.g., "this pointer
        cannot be null"). Use `TFLITE_DCHECK`. It costs nothing in production.
    *   *Is it validating control data from an input tensor?* (e.g., a dynamic
        index). Use a raw `if (!condition) return kTfLiteError;`. This safely
        prevents memory corruption without paying the ROM cost of a string
        literal.
    *   *Are you propagating an error from a helper function?* Use
        `TF_LITE_ENSURE_STATUS` or `TF_LITE_ENSURE_OK`.

### Macro Cheat Sheet

Macro                                          | Cost in Release                                          | When to Use
:--------------------------------------------- | :------------------------------------------------------- | :----------
`TFLITE_DCHECK`<br>`TFLITE_DCHECK_EQ`          | **Zero** (Optimized out)                                 | **Default for `Eval` invariants and FlatBuffer structural bounds checking.**
`if (!cond) return kTfLiteError;`              | **Low** (Branch only)                                    | **Default for validating control data in `Eval`.**
`TF_LITE_ENSURE_OK`<br>`TF_LITE_ENSURE_STATUS` | **Low** (Branch only)                                    | **Default for propagating errors from helper functions.**
`TF_LITE_ENSURE`<br>`TF_LITE_ENSURE_EQ`        | **High** (Branch + logs `__FILE__`, `__LINE__`, `#cond`) | **Default for `Prepare` / Setup.** Can be used in `Eval` *only* if the failure indicates an unrecoverable state-machine corruption; avoid for normal signal data out-of-bounds.
`TF_LITE_ENSURE_MSG`                           | **Highest** (Branch + custom string)                     | **Use Sparingly.** Only when the default `TF_LITE_ENSURE` error is too cryptic.
`TFLITE_CHECK`                                 | **Fatal** (Calls `Abort()`)                              | **Avoid in core TFLM.** Halts the microcontroller.

> **The Hidden Cost of `TF_LITE_ENSURE`:** This macro expands to an `if` branch
> that calls `TF_LITE_KERNEL_LOG`. By default, this embeds `__FILE__`,
> `__LINE__`, and the stringified condition directly into the `.rodata` section
> of the binary. Every single invocation permanently consumes precious ROM. If
> you have multiple preconditions, combine them into a single
> `TF_LITE_ENSURE(context, a != nullptr && b != nullptr)`. **Note:** For
> production builds where ROM is severely constrained, firmware developers
> should define the `TF_LITE_STRIP_ERROR_STRINGS` macro to compile out these
> strings, reducing `TF_LITE_ENSURE` to a simple low-cost branch.

--------------------------------------------------------------------------------

## 4. Code Review Guidelines & Concrete Examples

How to address common scenarios in PRs:

### Validating Framework Pointers

**Discouraged.** Please avoid wasting ROM checking the validity of the TFLM
framework itself. The `context` and `node` pointers passed by the runtime are
typically guaranteed to be valid. Notably, `TF_LITE_ENSURE(context, context !=
nullptr)` is logically flawed: if `context` is actually `nullptr`, the macro
will dereference it to log the error, causing an immediate hardware fault.

### Validating Public API Parameters

**Discouraged.** This penalizes production deployments for bugs that the
firmware developer should have caught during local testing. We recommend asking
the contributor to use `TFLITE_DCHECK(op_resolver != nullptr);` instead of
adding `if (op_resolver == nullptr) return kTfLiteError;` to the
`MicroInterpreter` API.

### Preventing Null Pointer Dereferences

*   **If in `Prepare` (Recommended):** It is standard practice to check for
    nulls when pulling tensors from the context.

```cpp
TfLiteTensor* operand = micro_context->AllocateTempInputTensor(node, kOperandTensor);
TF_LITE_ENSURE(context, operand != nullptr);
```

*   **If in `Eval` (Discouraged):** Production builds shouldn't pay the cycle
    cost for checks that can't fail at runtime if `Prepare` succeeded. Ask the
    author to change it to `TFLITE_DCHECK`.

```cpp
const TfLiteEvalTensor* input_id = tflite::micro::GetEvalInput(context, node, 0);
TFLITE_DCHECK(input_id != nullptr);
```

### Preventing Buffer Overflows

*   **If in `Prepare` (Recommended):** Use `TF_LITE_ENSURE` to validate that
    tensor sizes are compatible.
*   **If in `Eval`:** If the overflow comes from invalid *control data* (like an
    input tensor providing indices), it *should* be validated in `Eval` to
    prevent memory corruption. Use raw returns:

```cpp
// e.g., third_party/tflite_micro/tensorflow/lite/micro/kernels/gather_nd.cc
// Note: use subtraction to prevent integer overflow!
if (from_pos < 0 || from_pos > params_flat_size - slice_size) {
  return kTfLiteError; // Halts execution to prevent out-of-bounds memory read
}
```

```
If it's just an invariant guaranteed by `Prepare`, ask to hoist the check or
use `TFLITE_DCHECK`.
```

### Fixing Fuzzer Crashes

Fuzzer fixes should align with the core philosophy. We evaluate them based on
the type of crash:

1.  **Corrupted FlatBuffer Files (Working As Intended):** If the fuzzer mutates
    the FlatBuffer byte array so a `Tensor` offset points beyond the end of the
    file (causing a segfault), **request changes if the PR uses `TF_LITE_ENSURE`
    or raw `if` statements**. To keep the binary size as small as possible, we
    prefer to omit the FlatBuffer verifier. However, **it is recommended to
    accept the PR if it exclusively uses `TFLITE_DCHECK`** to catch the
    out-of-bounds index. This treats the structural check purely as a zero-cost
    developer aid during debugging.
2.  **Invalid Model Topologies (Fix in Prepare):** If the fuzzer generates a
    valid FlatBuffer but modifies a `CONV_2D` operator to have 0 inputs instead
    of the expected 3, **this is a good fix in `Prepare`** using
    `TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);`.
3.  **Static Math Crashes (Fix in Prepare):** If the fuzzer sets a quantization
    `scale` parameter to `0.0` (causing a hardware trap during inference),
    **this is a good fix in `Prepare`** using `TF_LITE_ENSURE(context, scale !=
    0.0);`.
4.  **Dynamic Math/Data Crashes (Fix in Eval):** If the fuzzer provides input
    data to a `GATHER` op with an index of `999` (causing out-of-bounds
    corruption), or a divisor tensor evaluates to `0` (causing a divide-by-zero
    trap), **this is a good fix in `Eval`**. Prefer a raw `if (index >= 10)
    return kTfLiteError;` to save ROM (avoid using `TF_LITE_ENSURE` here just to
    log the error; save the ROM).
