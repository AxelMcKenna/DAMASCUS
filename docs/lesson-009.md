
  # Lesson 009 — Compartment 2: Tensor PIMPL split (MTLBuffer-backed)

  **Date:** 2026-06-23

  ## The PIMPL split — why MTLBuffer can't live in the header
MTLBuffer is a Metal / Objective C object that cannot cleanly live in a normal C++ header. If tensor.hpp exposed 
id<MTLBuffer> then every file that includes the tensor header file would need to be compiled as Obj-C. The
PIMPL split allows Tensor to kepp a normal C++ interface while hiding the metal-backed storage inside TensorImpl
  
  ## Incomplete type forces ~Tensor() into the .mm
In the header, TensorImpl is only forward declared with struct TensorImpl; this is enough for the Tensor to hold a 
TensorImpl* but not enough for the compiler to know how to destroy the actual object. Because Tensor owns impl, ~Tensor() must be defined in tensor.mm where the full TensorImlp definition is visible.

  
  ## The internal Obj-C++ header — keeping id<> out of pure C++
The public headers should expose public opaque C++ types like MetalContext, not Obj-C types like id<MTLDevice>.
The internal Obj-C header is the bridge - .mm files can include it to recover the real Metal Device when they need to allocate buffers. This keeps #import <Metal/Metal.h>  and id<> syntax out of the pure C++ side of codebase.

  ## Two lifetimes, two mechanisms — ARC vs C++ delete
There are two liftetimes involved with the memory management. C++ owns TensorImpl wrapper with new and delete. ARC owns the Obj-C Metal object stored inside of it. So delete impl destroys the C++ wrapper, and ARC handles releasing the id<MTLBuffer> member. 


  ## Header/impl signatures must match exactly
The header and implementation have to describe the exact same functions. E.g forgot to update void* data() -> std::byte* Tensor::data()

  ## What's not closed — the ASan run is still pending
  <!-- compiles clean ≠ verified; allocate/fill/readback under ASan blocked on IT access -->
Need to sus M1 perms on work laptop as I haven't been able to run ASan yet for the test suite.


  ## What tripped me up
  The main trap was thinking PIMPL only hides the Metal field, when it also determines where destruction is legal. Since the full TensorImpl type only exists in the .mm, ownership logic has to live there too. The other trap was letting tensor.hpp and tensor.mm drift apart, especially around data() returning void* versus std::byte*.


  ## Open / next session
  Next session is the runtime smoke test. A {4} f32 tensor should be 4 * 4 = 16 bytes, so the Metal buffer length should be 16 bytes. The test should write 0, 1, 2, 3 through as_f32(), read them back, move the tensor, read them again through the moved-to object, then run the whole thing under ASan.