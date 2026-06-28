# Single Makefile. Migrate to CMake when the project outgrows this.

CXX       = clang++
CXXFLAGS  = -std=c++20 -Wall -Wextra -Wpedantic -O2 -fno-rtti
INCLUDES  = -Ithird_party/metal-cpp -Iinclude -Isrc
FRAMEWORKS = -framework Foundation -framework Metal -framework QuartzCore

# Debug build adds sanitizers. `make debug` to invoke.
DEBUG_FLAGS = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer

BUILD_DIR = build
SRC_DIR   = src

CXX_SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/gguf/gguf.cpp \
           $(SRC_DIR)/tokenizer/tokenizer.cpp $(SRC_DIR)/kernels/kernels_cpu.cpp \
           $(SRC_DIR)/quant/quant_cpu.cpp $(SRC_DIR)/model/model.cpp \
           $(SRC_DIR)/forward/forward.cpp
MM_SRCS  = $(SRC_DIR)/metal/metal_ctx.mm $(SRC_DIR)/tensor/tensor.mm

CXX_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CXX_SRCS))
MM_OBJS  = $(patsubst $(SRC_DIR)/%.mm,$(BUILD_DIR)/%.o,$(MM_SRCS))
OBJS     = $(CXX_OBJS) $(MM_OBJS)

TEST_BIN = $(BUILD_DIR)/tensor_test

.PHONY: test
test: CXXFLAGS += $(DEBUG_FLAGS)
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): tests/tensor_test.cpp $(MM_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc $(FRAMEWORKS) $^ -o $@

TARGET = $(BUILD_DIR)/damascus

.PHONY: all clean debug compile_commands

.DEFAULT_GOAL := all

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: all

# Generate compile_commands.json for clangd. Requires `bear`.
compile_commands:
	$(MAKE) clean
	bear -- $(MAKE) all

clean:
	rm -rf $(BUILD_DIR)

# ---- Oracle (Milestone-A correctness reference) ----
# Runs HF Llama in a Linux container, escaping the host EDR block. CPU only.
# The Llama model is gated: `hf auth login` on the host first (cache is mounted in).
ORACLE_IMG    = damascus-oracle
ORACLE_DIR    = tools/oracle
ORACLE_PROMPT ?= The capital of France is
DOCKER_RUN    = docker run --rm -v $(PWD)/$(ORACLE_DIR):/work \
                  -v $$HOME/.cache/huggingface:/root/.cache/huggingface \
                  -e HF_TOKEN $(ORACLE_IMG)

ORACLE_GGUF ?= models/llama-3.2-3b-instruct.Q4_K_M.gguf
DOCKER_RUN_GGUF = docker run --rm -v $(PWD)/$(ORACLE_DIR):/work \
                  -v $(PWD)/models:/models:ro $(ORACLE_IMG)

.PHONY: oracle-image oracle oracle-gguf oracle-compare

oracle-image:
	docker build -t $(ORACLE_IMG) $(ORACLE_DIR)

# Ungated, same-weights oracle: reference from the LOCAL GGUF (no HF token).
# Dequantizes the same Q4_K_M blocks the engine uses -> 1e-3 absolute is meaningful.
oracle-gguf: oracle-image
	$(DOCKER_RUN_GGUF) dump_reference.py --gguf /models/$(notdir $(ORACLE_GGUF)) \
	  --prompt "$(ORACLE_PROMPT)" --out /work/reference

# Gated route: original fp32 HF weights (needs HF_TOKEN; logits differ by quant noise).
oracle: oracle-image
	$(DOCKER_RUN) dump_reference.py --prompt "$(ORACLE_PROMPT)" --out /work/reference

# Score engine activations vs the reference: make oracle-compare ENGINE=tools/oracle/engine
oracle-compare:
	$(DOCKER_RUN) compare.py /work/reference /work/$(notdir $(ENGINE))

# ---- CPU-verified compartments (C4 tokenizer, C5 kernels) ----
# Compiled+run under gcc/Linux in a container, escaping the host EDR block.
# No Metal needed: these are the scalar references the Metal ports validate
# against later. `make cpu-check` runs both gates.
GCC_RUN = docker run --rm -v "$(PWD):/src:ro" -v "$(PWD)/models:/models:ro" \
            -w /src gcc:latest bash -c

.PHONY: tok-check kernels-check cpu-check

# Compartment 4: tokenizer vs HF oracle ids + round-trip invariants.
tok-check:
	$(GCC_RUN) 'g++ -std=c++20 -Wall -Wextra -Iinclude \
	  src/gguf/gguf.cpp src/tokenizer/tokenizer.cpp tools/oracle/tokenizer_check.cpp \
	  -O2 -o /tmp/tok && /tmp/tok /models/$(notdir $(ORACLE_GGUF))'

# Compartment 5: CPU reference kernels vs hand-computed values.
kernels-check:
	$(GCC_RUN) 'g++ -std=c++20 -Wall -Wextra -Iinclude \
	  src/kernels/kernels_cpu.cpp tools/oracle/kernels_check.cpp \
	  -O2 -o /tmp/k && /tmp/k'

cpu-check: kernels-check tok-check

# Compartment 6: full forward pass vs the SAME-WEIGHTS GGUF oracle (Milestone A).
# Build the same-weights reference first with `make oracle-gguf` (it lands in
# tools/oracle/reference_gguf). Source is mounted read-only for the compile;
# tools/oracle is mounted writable so the engine can dump its npy arrays.
forward-check:
	mkdir -p $(ORACLE_DIR)/engine_gguf
	docker run --rm -v "$(PWD):/src:ro" -v "$(PWD)/$(ORACLE_DIR):/work" \
	  -v "$(PWD)/models:/models:ro" -w /src gcc:latest bash -c \
	  'g++ -std=c++20 -Wall -Wextra -Iinclude -Itools/oracle \
	    src/gguf/gguf.cpp src/quant/quant_cpu.cpp src/model/model.cpp \
	    src/kernels/kernels_cpu.cpp src/forward/forward.cpp tools/oracle/forward_dump.cpp \
	    -O2 -o /tmp/fwd && /tmp/fwd /models/$(notdir $(ORACLE_GGUF)) \
	    /work/reference_gguf /work/engine_gguf'
	$(DOCKER_RUN) compare.py /work/reference_gguf /work/engine_gguf
