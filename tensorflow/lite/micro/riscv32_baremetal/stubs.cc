extern "C" {

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

// Need _trap as it is used inside void _exit 
// trap handler
static void _trap(void) {
  while (1) { __asm__("wfi"); }
}

//_sbrk() grows the heap by moving its end pointer and gives the old address to store data there.
// heap grows upwards and stack growd downwards
void* _sbrk(ptrdiff_t incr) {
  extern char _end;  // Symbol from linker script = end of program (.text + .data + .bss)
  static char* heap_end;  // Pointer that tracks the current end of the heap
  if (!heap_end) {
    heap_end = &_end;  // Initialize heap to start just after _end
  }
  char* prev_heap_end = heap_end;  // Save current heap end to return later
  heap_end += incr;        // Move heap pointer forward by requested amount
  return (void*)prev_heap_end;  // Return old end as start of allocated block
}

// Minimal syscall stubs for bare-metal (no OS).
// These dummy functions satisfy newlib/stdio dependencies so code using
// printf, malloc, etc. can link. Some stubs redirect output to UART.
// while others just return defaults or errors since there is no filesystem or OS. 
int _close(int file) { return -1; }
int _fstat(int file, struct stat* st) { st->st_mode = S_IFCHR; return 0; }
int _isatty(int file) { return 1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _read(int file, char* ptr, int len) { errno = EINVAL; return -1; }
int _write(int file, const char* ptr, int len) { return len; }
void _exit(int status) { _trap(); }
int _kill(int pid, int sig) { errno = EINVAL; return -1; }
int _getpid(void) { return 1; }

} 
