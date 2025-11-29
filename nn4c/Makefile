CFLAGS := -O3 -march=native -ffast-math
CFLAGS += -Wall -Wextra -pedantic
CFLAGS += -Iinclude -MMD -MP

ifeq ($(OS),Windows_NT)
	# MinGW gcc performs much better than LLVM clang in optimization on Windows
	CC     := gcc
	EXT    := .exe
	LDLIBS :=
else
	CC     := clang
	EXT    := .out
	LDLIBS := -lm
endif

BUILD_DIR  := build
C_TARGET   := $(BUILD_DIR)/main$(EXT)
C_SRCS     := $(wildcard *.c)
C_OBJS     := $(patsubst %.c,$(BUILD_DIR)/%.o,$(C_SRCS))
DEPS       := $(C_OBJS:.o=.d)

.PHONY: all run clean

all: $(C_TARGET)

$(C_TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(DEPS)

run: $(C_TARGET)
	./$(C_TARGET)

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(C_TARGET)
