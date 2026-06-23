 # Lesson 010 — Compartment 3: GGUF parsing & model loading

  Date: 2026-06-23

  ## What I set out to do
Goal was to turn GGUF into a legit model loader - DoD was to load the target model, report shape and hand back the tensor list.

  ## The GGUF file layout
GGUF always starts with a fixed header, magic u32, version u32, tensor count u64 and metadata key-value count u64. After that is the metadata entries, then the tensor info
table. THEN the actual tensor data blobs. 


  ## Metadata: why std::variant
  <!-- Why couldn't a plain struct or a switch hold metadata values? What problem
       does std::variant solve here? What's the type of a metadata value in our code? -->
Metadata values are not one-fixed type: one key might hold a u32, another a float, another a string. std::variant is best here because the type needs to be stored. The metadata value
represents a GGUF value in this context.       

  ## The recursive ARRAY type
  <!-- An ARRAY value can contain other values. How did we express that in the type
       without infinite size? Why does std::vector tolerate an incomplete type? -->
A GGUF array is "many GGUF vals" so the representation is std::vector<GGUFVALUE> inside GGUF. This required GGUFValue to become a struct with an Array arm.
The recursion works because the vector stores its elements indirectly thru allocated storage so the struct can contain a vector of its own type without allocating infinite size.

  ## The read_primitive-into-std::string bug
  <!-- Describe the bug in your own words: what does read_primitive<T> do mechanically,
       and why is it catastrophic when T is std::string? What did we use instead, and why? -->
read_primitive<T> mechanically copies sizeof(T) raw bytes from the file into an object using memcpy. That is alg for numeric types but doesn't work for std::string because 
it would overwrite the strings object internal pointer, size, capacity, etc. GGUF must use read_string() instead.


  ## std::get vs std::get_if
  <!-- When do you reach for each? Tie it to required vs optional config fields and what
       happens on a type mismatch for each. -->
std::get<T> is for required fields where the type must be correct. If the wrong variant arm is active, it throws loudly. std::get_it<T> is optional for probe style reads.
For required config fields, get_metadata<T> uses std::get<T> so missing keys and wrong types become explicit loader errors instead of a silent failure.


  ## Alignment & offset relativity (Q1 + Q2)
  <!-- Why is hardcoding alignment = 32 a latent bug, and why fine for THIS model?
       Write the exact pointer expression for a tensor's first byte from a GGUFModel
       and a TensorInfo, and explain each of the three terms. -->
Is fine for this model because it uses the default alignment which is 32, however should move this to general.alignment when extending to other models so it doesn't shift the computed tensor data blob start.


  ## Wiring the entry point
  <!-- What does main now do? Why argc >= 3 before argv[2]? Why try/catch around
       load_gguf when --device-info needed no such thing? -->
main.cpp is now a command dispatcher instead of the parser itself. It keeps the existing version and device-info commands whilst adding the load gguf one.
Wrapped in a try / catch block because there are many recoverable failures. e.g model not in the correct path, so hard failure isn't necessary.

  ## What's verified vs. not
  <!-- Be honest: what did we actually prove (compiles? links? warnings?) and what
       is still unverified, and WHY can't we verify it on this machine? -->
Hardware can't run on work laptop (gay) 


  ## What I'd revisit
  <!-- The general.alignment TODO, the vocab_size caveat, anything that felt shaky. -->
The first thing to revisit is general.alignment, because hardcoding 32 is only safe while the model happens to use the default. I would also tighten the vocab_size handling, since some GGUF files expose it directly while others require deriving it from tokenizer.ggml.tokens. The other shaky area is deciding how much validation belongs in load_gguf() now versus later, especially around tensor type support and whether gaps in the tensor blob are acceptable alignment padding or something suspicious.