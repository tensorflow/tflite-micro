# Security Policy & Threat Model

TensorFlow Lite for Microcontrollers (TFLM) is an embedded inference library
targeting deeply resource-constrained bare-metal microcontrollers (typically
with a single address space and no MMU) where every byte of Flash/ROM, every
byte of RAM, and every CPU cycle matters.

Because TFLM is linked directly into device firmware as a local library rather
than operating as a multi-tenant service or privilege boundary, its defensive
programming model differs fundamentally from desktop, mobile, and server-side
ML runtimes. Before reporting an issue or submitting a defensive-coding pull
request, please read our
[Error Handling & Defensive Programming Guide](docs/error_handling_guide.md).

## Trust Boundaries

1. **The C++ API (Trusted):** TFLM treats the firmware engineer calling the
   public C++ API as trusted.
2. **The `.tflite` FlatBuffer Model (Trusted Structure):** TFLM assumes that the
   `.tflite` model loaded into the interpreter is structurally valid and was
   produced by a trusted converter pipeline. To minimize binary footprint, TFLM
   intentionally omits a runtime FlatBuffer verifier as well as runtime bounds
   and integer-overflow checks on static model parameters. If a deployment
   receives models over an untrusted channel (such as OTA updates), the
   application firmware is responsible for authenticating and validating the
   integrity of the model before passing it to TFLM.

## Out of Scope (Not Security Vulnerabilities)

The following categories are **working as intended** under TFLM's design
constraints and are **not security vulnerabilities**:

* **Corrupted, mutated, or maliciously crafted `.tflite` FlatBuffer files:**
  Crashes, aborts (`TFLITE_CHECK` / `TFLITE_DCHECK`), division-by-zero faults,
  integer overflows on static tensor dimensions, or out-of-bounds memory
  accesses that require modifying or crafting a `.tflite` model file (such as
  invalid FlatBuffer offsets, mismatched static tensor shapes, out-of-range
  operator attributes, or missing quantization parameters).
* **Misuse of the C++ API:** Passing invalid pointers, uninitialized buffers, or
  violating interpreter state-machine ordering from application code.
* **Speculative or AI-scanned hardening reports:** Automated static-analysis,
  fuzzer, or LLM-generated reports and pull requests that flag theoretical
  checks without a reproducible bug in a real TFLM deployment.
