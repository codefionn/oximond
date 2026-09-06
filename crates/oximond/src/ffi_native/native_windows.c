/* The Windows x64 native backend. libffi is not part of a Windows toolchain,
 * and the Win64 convention is regular enough to call through one trampoline:
 * every argument takes one 8-byte slot, the first four travel in RCX, RDX, R8
 * and R9 or in XMM0 to XMM3 by position, and the rest go on the stack above
 * 32 bytes of shadow space. The trampoline loads each of the first four slots
 * into both registers. The callee reads the one its prototype names, and a
 * variadic callee finds a floating argument in both, as the convention asks.
 * None of the supported types is an aggregate, so no slot holds a pointer to
 * a copy. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    TY_VOID,
    TY_SINT32,
    TY_UINT32,
    TY_SLONG,
    TY_ULONG,
    TY_SINT64,
    TY_UINT64,
    TY_FLOAT,
    TY_DOUBLE,
    TY_POINTER,
    TY_SINT8,
    TY_UINT8,
    TY_SINT16,
    TY_UINT16
};

/* Keeps the stack area below one page, so the trampoline needs no probe. */
#define MAX_ARGS 256

/* oximond_win64_invoke(function, slots, count, result) copies `count` slots
 * into the outgoing argument area, calls `function`, and stores RAX at
 * result[0] and XMM0 at result[1]. */
void oximond_win64_invoke(void *function, const uint64_t *slots, size_t count,
                          uint64_t *result);
__asm__(
    ".text\n"
    ".globl oximond_win64_invoke\n"
    ".def oximond_win64_invoke; .scl 2; .type 32; .endef\n"
    ".seh_proc oximond_win64_invoke\n"
    "oximond_win64_invoke:\n"
    "    pushq %rbp\n"
    "    .seh_pushreg %rbp\n"
    "    pushq %rsi\n"
    "    .seh_pushreg %rsi\n"
    "    pushq %rdi\n"
    "    .seh_pushreg %rdi\n"
    "    pushq %r12\n"
    "    .seh_pushreg %r12\n"
    "    pushq %r13\n"
    "    .seh_pushreg %r13\n"
    "    movq %rsp, %rbp\n"
    "    .seh_setframe %rbp, 0\n"
    "    .seh_endprologue\n"
    "    movq %rcx, %r12\n"
    "    movq %rdx, %rsi\n"
    "    movq %r8, %rdi\n"
    "    movq %r9, %r13\n"
    /* Five pushes realigned RSP to 16; the area is a multiple of 16 holding
     * at least the four shadow slots. */
    "    movq %r8, %rax\n"
    "    cmpq $4, %rax\n"
    "    jae 1f\n"
    "    movl $4, %eax\n"
    "1:\n"
    "    leaq 15(,%rax,8), %rax\n"
    "    andq $-16, %rax\n"
    "    subq %rax, %rsp\n"
    "    xorl %ecx, %ecx\n"
    "2:\n"
    "    cmpq %rdi, %rcx\n"
    "    jae 3f\n"
    "    movq (%rsi,%rcx,8), %rax\n"
    "    movq %rax, (%rsp,%rcx,8)\n"
    "    incq %rcx\n"
    "    jmp 2b\n"
    "3:\n"
    "    movq 0(%rsp), %rcx\n"
    "    movq 8(%rsp), %rdx\n"
    "    movq 16(%rsp), %r8\n"
    "    movq 24(%rsp), %r9\n"
    "    movq 0(%rsp), %xmm0\n"
    "    movq 8(%rsp), %xmm1\n"
    "    movq 16(%rsp), %xmm2\n"
    "    movq 24(%rsp), %xmm3\n"
    "    callq *%r12\n"
    "    movq %rax, 0(%r13)\n"
    "    movq %xmm0, 8(%r13)\n"
    "    leaq 0(%rbp), %rsp\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rdi\n"
    "    popq %rsi\n"
    "    popq %rbp\n"
    "    ret\n"
    ".seh_endproc\n");

static _Thread_local char last_error[512];

static void remember_error(const char *context) {
    DWORD code = GetLastError();
    size_t used = 0;
    if (context) {
        used = strlen(context);
        if (used > sizeof(last_error) - 3) used = sizeof(last_error) - 3;
        memcpy(last_error, context, used);
        last_error[used++] = ':';
        last_error[used++] = ' ';
    }
    DWORD length = FormatMessageA(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM,
                                  NULL, code, 0, last_error + used,
                                  (DWORD)(sizeof(last_error) - used), NULL);
    used += length;
    /* FormatMessage ends its text with CRLF. */
    while (used > 0 && (last_error[used - 1] == '\n' || last_error[used - 1] == '\r'))
        used--;
    if (length == 0) {
        static const char fallback[] = "Windows error";
        size_t room = sizeof(last_error) - used - 1;
        size_t take = sizeof(fallback) - 1 < room ? sizeof(fallback) - 1 : room;
        memcpy(last_error + used, fallback, take);
        used += take;
    }
    last_error[used] = '\0';
}

static HMODULE executable(void) { return GetModuleHandleW(NULL); }

void *oximond_ffi_open(const char *name) {
    if (!name) return executable();
    HMODULE handle = LoadLibraryExA(name, NULL, 0);
    if (!handle) remember_error(name);
    return handle;
}

void oximond_ffi_close(void *handle) {
    if (handle && handle != executable()) FreeLibrary((HMODULE)handle);
}

/* ffi.C searches what LuaJIT's does on Windows: the executable, then the C
 * runtime, kernel32, user32 and gdi32. The universal CRT defines printf and
 * its family inline in its headers, so msvcrt.dll, which exports them, comes
 * after it. The on-demand libraries load on first use and stay loaded. */
static void *process_symbol(const char *name) {
    static const char *const loaded[] = {"ucrtbase.dll", "kernel32.dll"};
    for (size_t i = 0; i < sizeof(loaded) / sizeof(loaded[0]); i++) {
        HMODULE module = GetModuleHandleA(loaded[i]);
        FARPROC symbol = module ? GetProcAddress(module, name) : NULL;
        if (symbol) return (void *)symbol;
    }
    static const char *const on_demand[] = {"msvcrt.dll", "user32.dll", "gdi32.dll"};
    for (size_t i = 0; i < sizeof(on_demand) / sizeof(on_demand[0]); i++) {
        HMODULE module = LoadLibraryExA(on_demand[i], NULL, 0);
        FARPROC symbol = module ? GetProcAddress(module, name) : NULL;
        if (symbol) return (void *)symbol;
    }
    return NULL;
}

void *oximond_ffi_symbol(void *handle, const char *name) {
    void *symbol = (void *)GetProcAddress((HMODULE)handle, name);
    if (!symbol && handle == executable()) symbol = process_symbol(name);
    if (!symbol) remember_error(name);
    return symbol;
}

const char *oximond_ffi_error(void) {
    return last_error[0] ? last_error : NULL;
}

int oximond_ffi_errno(void) {
    return errno;
}

void oximond_ffi_set_errno(int value) {
    errno = value;
}

/* Widens one argument to its 8-byte slot. Integers narrower than a slot are
 * extended by their signedness, a float stays in the low four bytes. */
static int encode(unsigned char type, const void *value, uint64_t *slot) {
    switch (type) {
    case TY_SINT8: *slot = (uint64_t)(int64_t)*(const int8_t *)value; return 0;
    case TY_UINT8: *slot = *(const uint8_t *)value; return 0;
    case TY_SINT16: *slot = (uint64_t)(int64_t)*(const int16_t *)value; return 0;
    case TY_UINT16: *slot = *(const uint16_t *)value; return 0;
    case TY_SINT32:
    case TY_SLONG: *slot = (uint64_t)(int64_t)*(const int32_t *)value; return 0;
    case TY_UINT32:
    case TY_ULONG: *slot = *(const uint32_t *)value; return 0;
    case TY_SINT64:
    case TY_UINT64: *slot = *(const uint64_t *)value; return 0;
    case TY_POINTER: *slot = (uint64_t)(uintptr_t)*(void *const *)value; return 0;
    case TY_FLOAT: *slot = 0; memcpy(slot, value, 4); return 0;
    case TY_DOUBLE: memcpy(slot, value, 8); return 0;
    default: return -1;
    }
}

int oximond_ffi_call(void *function, unsigned char result_type,
                     const unsigned char *arg_types, void **arg_values,
                     size_t arg_count, size_t fixed_count, void *result) {
    uint64_t slots[MAX_ARGS];
    uint64_t registers[2];
    if (!function || result_type > TY_UINT16 || arg_count > MAX_ARGS || fixed_count > arg_count)
        return -1;
    for (size_t i = 0; i < arg_count; i++) {
        if (encode(arg_types[i], arg_values[i], &slots[i]) != 0) return -1;
    }
    oximond_win64_invoke(function, slots, arg_count, registers);
    switch (result_type) {
    case TY_VOID: break;
    case TY_FLOAT: memcpy(result, &registers[1], 4); break;
    case TY_DOUBLE: memcpy(result, &registers[1], 8); break;
    default: memcpy(result, &registers[0], 8); break;
    }
    return 0;
}

#ifdef __MINGW32__
/* zig's compiler_rt, which it links into every windows-gnu executable,
 * defines a weak `fma` that rounds twice: fma(-1, -1e300, -1e300) gives
 * 2^-78 instead of 0. Rust's `f64::mul_add` calls `fma` wherever the code is
 * not compiled with the FMA instructions, so the Lua `%` of an unoptimized
 * build would round wrong. This strong definition overrides the weak one. It
 * lives in this file because the linker always loads it (the ffi module
 * references it), and a definition in an object nothing else pulls in would
 * lose to the weak one. It forwards to the universal C runtime's `fma`, which
 * the executable imports already. */
typedef double (*fma_fn)(double, double, double);

double fma(double x, double y, double z) {
    static fma_fn ucrt_fma;
    fma_fn f = ucrt_fma;
    if (!f) {
        HMODULE ucrt = GetModuleHandleA("ucrtbase.dll");
        if (!ucrt) ucrt = LoadLibraryA("ucrtbase.dll");
        f = (fma_fn)(void (*)(void))GetProcAddress(ucrt, "fma");
        ucrt_fma = f;
    }
    return f(x, y, z);
}
#endif
