// LD_PRELOAD malloc profiler: counts allocator calls per call stack.
//
//   gcc -O1 -g -shared -fPIC -o mallocprof.so mallocprof.c -ldl
//   MALLOCPROF_OUT=out.txt LD_PRELOAD=./mallocprof.so oximond ...
//
// Output, one line per distinct stack:
//   <kind> <thread> <calls> <bytes> <ip0> <ip1> ...   (hex return addresses)
// kind is m (malloc, calloc, realloc, memalign) or f (free). thread is 0 for the
// main thread and 1 for any other thread. The first line "base <hex> <path>" gives
// the load address of the executable, to subtract before addr2line.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define DEPTH 32
#define SLOTS (1u << 17)

typedef struct {
    uint64_t hash;
    uint64_t calls, bytes;
    uint32_t kind, thread, depth;
    void *ip[DEPTH];
} Slot;

static Slot *slots;
static atomic_flag lock = ATOMIC_FLAG_INIT;
static __thread int busy;
static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void (*real_free)(void *);
static int (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);
static char boot[1 << 16];
static size_t boot_used;
static int ready;

__attribute__((noinline)) static void record(uint32_t kind, size_t bytes) {
    if (!ready || busy) return;
    busy = 1;
    void *ip[DEPTH + 2];
    int n = backtrace(ip, DEPTH + 2);
    // Drop this function and the hook.
    int skip = 2;
    uint32_t thread = syscall(SYS_gettid) != getpid();
    uint64_t h = 1469598103934665603ull ^ kind ^ ((uint64_t)thread << 8);
    for (int i = skip; i < n; i++) h = (h ^ (uint64_t)ip[i]) * 1099511628211ull;
    while (atomic_flag_test_and_set(&lock)) {}
    uint32_t i = (uint32_t)h & (SLOTS - 1);
    for (;;) {
        Slot *s = &slots[i];
        if (s->calls == 0) {
            s->hash = h; s->kind = kind; s->thread = thread; s->depth = n - skip;
            memcpy(s->ip, ip + skip, (n - skip) * sizeof(void *));
            s->calls = 1; s->bytes = bytes;
            break;
        }
        if (s->hash == h) { s->calls++; s->bytes += bytes; break; }
        i = (i + 1) & (SLOTS - 1);
    }
    atomic_flag_clear(&lock);
    busy = 0;
}

static Slot *mmap_slots(void) {
    void *p = mmap(0, sizeof(Slot) * SLOTS, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? 0 : p;
}

__attribute__((constructor)) static void init(void) {
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    slots = mmap_slots();
    void *warm[4];
    backtrace(warm, 4);
    ready = 1;
}

static int in_boot(void *p) { return (char *)p >= boot && (char *)p < boot + sizeof boot; }

void *malloc(size_t n) {
    if (!real_malloc) { void *p = boot + boot_used; boot_used += (n + 15) & ~15ul; return p; }
    record(0, n);
    return real_malloc(n);
}
void *calloc(size_t a, size_t b) {
    if (!real_calloc) { void *p = boot + boot_used; boot_used += (a * b + 15) & ~15ul; return p; }
    record(0, a * b);
    return real_calloc(a, b);
}
void *realloc(void *p, size_t n) {
    if (in_boot(p)) { void *q = malloc(n); memcpy(q, p, n); return q; }
    record(0, n);
    return real_realloc(p, n);
}
void free(void *p) {
    if (!p || in_boot(p)) return;
    record(1, 0);
    real_free(p);
}
int posix_memalign(void **out, size_t al, size_t n) {
    record(0, n);
    return real_posix_memalign(out, al, n);
}
void *aligned_alloc(size_t al, size_t n) {
    record(0, n);
    return real_aligned_alloc(al, n);
}

__attribute__((destructor)) static void dump(void) {
    const char *path = getenv("MALLOCPROF_OUT");
    if (!path || !slots) return;
    ready = 0;
    FILE *f = fopen(path, "w");
    if (!f) return;
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[512], exe[512] = {0};
    ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (len > 0) exe[len] = 0;
    while (maps && fgets(line, sizeof line, maps)) {
        if (strstr(line, exe)) {
            unsigned long lo;
            sscanf(line, "%lx", &lo);
            fprintf(f, "base %lx %s\n", lo, exe);
            break;
        }
    }
    if (maps) fclose(maps);
    for (uint32_t i = 0; i < SLOTS; i++) {
        Slot *s = &slots[i];
        if (!s->calls) continue;
        fprintf(f, "%c %u %lu %lu", s->kind ? 'f' : 'm', s->thread, s->calls, s->bytes);
        for (uint32_t k = 0; k < s->depth; k++) fprintf(f, " %lx", (unsigned long)s->ip[k]);
        fputc('\n', f);
    }
    fclose(f);
}
