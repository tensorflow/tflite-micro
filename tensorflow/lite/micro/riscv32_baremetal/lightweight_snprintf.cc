#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>  

// mini_vsnprintf()
// A minimal implementation of vsnprintf() supporting %s, %d, and %p.
// Writes formatted output into 'buf' up to 'size' bytes.

int mini_vsnprintf(char *buf, size_t size, const char *fmt, va_list args) {
    char *out = buf;
    size_t remaining = size;

    #define PUTCHAR(c) \
        do { if (remaining > 1) { *out++ = (c); remaining--; } } while (0)

    while (*fmt) {
        if (*fmt == '%' && *(fmt+1) == 's') {
            fmt += 2;
            const char *s = va_arg(args, const char *);
            if (!s) s = "(null)";
            while (*s) PUTCHAR(*s++);
        } else if (*fmt == '%' && *(fmt+1) == 'd') {
            fmt += 2;
            int v = va_arg(args, int);
            if (v < 0) { PUTCHAR('-'); v = -v; }
            char tmp[16]; int i=0;
            do { tmp[i++] = '0' + (v % 10); v /= 10; } while (v && i < 16);
            while (i--) PUTCHAR(tmp[i]);
        } else if (*fmt == '%' && *(fmt+1) == 'p') {
            fmt += 2;
            void *ptr = va_arg(args, void*);
            uintptr_t val = (uintptr_t)ptr;
            char tmp[2*sizeof(void*)+1]; int i=0;
            if (!val) {
                const char *null_str = "(nil)";
                while (*null_str) PUTCHAR(*null_str++);
            } else {
                do { 
                    tmp[i++] = "0123456789abcdef"[val % 16]; 
                    val /= 16; 
                } while(val && i < (int)sizeof(tmp));
                PUTCHAR('0'); PUTCHAR('x');
                while(i--) PUTCHAR(tmp[i]);
            }
        } else {
            PUTCHAR(*fmt++);
        }
    }

    if (remaining > 0) *out = '\0';
    return (int)(size - remaining);
}

// mini_snprintf()
// Simple wrapper around mini_vsnprintf() that takes a variable number of args.
int mini_snprintf(char *buf, size_t size, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int ret = mini_vsnprintf(buf, size, fmt, args);
    va_end(args);
    return ret;
}

#ifdef __cplusplus
}
#endif