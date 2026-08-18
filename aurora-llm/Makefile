AS=gcc
LD=ld
CFLAGS=-c -nostdlib -fno-asynchronous-unwind-tables -fno-stack-protector -mavx2 -mfma -O2

all: aurora-llm

aurora-llm: src/aurora_llm.S
	$(AS) $(CFLAGS) -o build.o $<
	$(LD) -static -z noexecstack -e _start -o $@ build.o
	strip -s $@

clean:
	rm -f build.o aurora-llm
