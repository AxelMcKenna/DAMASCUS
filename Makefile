# Single Makefile. Migrate to CMake when the project outgrows this.

CXX       = clang++
CXXFLAGS  = -std=c++20 -Wall -Wextra -Wpedantic -O2 -fno-rtti
INCLUDES  = -Ithird_party/metal-cpp -Isrc
FRAMEWORKS = -framework Foundation -framework Metal -framework QuartzCore

# Debug build adds sanitizers. `make debug` to invoke.
DEBUG_FLAGS = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer

BUILD_DIR = build
SRC_DIR   = src

CXX_SRCS = $(SRC_DIR)/main.cpp
MM_SRCS  = $(SRC_DIR)/metal_ctx.mm

CXX_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CXX_SRCS))
MM_OBJS  = $(patsubst $(SRC_DIR)/%.mm,$(BUILD_DIR)/%.o,$(MM_SRCS))
OBJS     = $(CXX_OBJS) $(MM_OBJS)

TARGET = $(BUILD_DIR)/damascus

.PHONY: all clean debug compile_commands

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm | $(BUILD_DIR)
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
