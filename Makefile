CFLAGS := -O3 -march=native -ffast-math
CFLAGS += -Wall -Wextra -pedantic
CFLAGS += -Inn4c/include

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

C_TARGET := nn4c/main$(EXT)

C_SRCS := $(wildcard nn4c/*.c)
C_OBJS := $(patsubst nn4c/%.c,nn4c/%.o,$(C_SRCS))

.PHONY: all

all: $(C_TARGET)

$(C_TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

nn4c/%.o: nn4c/%.c
	$(CC) $(CFLAGS) -c -o $@ $<
