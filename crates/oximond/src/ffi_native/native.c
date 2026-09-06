#include <ffi.h>
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>

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

#define TYPE_LIST(X) \
    X(TY_VOID, ffi_type_void) \
    X(TY_SINT32, ffi_type_sint32) \
    X(TY_UINT32, ffi_type_uint32) \
    X(TY_SLONG, ffi_type_slong) \
    X(TY_ULONG, ffi_type_ulong) \
    X(TY_SINT64, ffi_type_sint64) \
    X(TY_UINT64, ffi_type_uint64) \
    X(TY_FLOAT, ffi_type_float) \
    X(TY_DOUBLE, ffi_type_double) \
    X(TY_POINTER, ffi_type_pointer) \
    X(TY_SINT8, ffi_type_sint8) \
    X(TY_UINT8, ffi_type_uint8) \
    X(TY_SINT16, ffi_type_sint16) \
    X(TY_UINT16, ffi_type_uint16)

#ifdef OXIMOND_LAZY_LIBFFI
/* libffi is opened on the first call instead of at process start. Every
 * start paid for mapping and binding libffi.so (about 30 us and 8 page
 * faults), and most runs never call into C. */
#define STR2(x) #x
#define STR(x) STR2(x)

static struct {
    ffi_status (*prep_cif)(ffi_cif *, ffi_abi, unsigned, ffi_type *, ffi_type **);
    ffi_status (*prep_cif_var)(ffi_cif *, ffi_abi, unsigned, unsigned, ffi_type *,
                               ffi_type **);
    void (*call)(ffi_cif *, void (*)(void), void *, void **);
    ffi_type *types[TY_UINT16 + 1];
    int ready;
} libffi;
static pthread_once_t libffi_once = PTHREAD_ONCE_INIT;

/* build.rs defines OXIMOND_LIBFFI_SONAME as the soname of the libffi that
 * -lffi resolves to. Only that one matches the ABI of the ffi.h this file
 * is compiled against. */
static void libffi_load(void) {
    void *handle = dlopen(OXIMOND_LIBFFI_SONAME, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return;
    libffi.prep_cif = (__typeof__(libffi.prep_cif))dlsym(handle, "ffi_prep_cif");
    libffi.prep_cif_var = (__typeof__(libffi.prep_cif_var))dlsym(handle, "ffi_prep_cif_var");
    libffi.call = (__typeof__(libffi.call))dlsym(handle, "ffi_call");
#define LOAD(code, name) libffi.types[code] = (ffi_type *)dlsym(handle, STR(name));
    TYPE_LIST(LOAD)
#undef LOAD
    int ok = libffi.prep_cif && libffi.prep_cif_var && libffi.call;
    for (int i = 0; i <= TY_UINT16; i++) ok = ok && libffi.types[i];
    libffi.ready = ok;
}

static int libffi_available(void) {
    pthread_once(&libffi_once, libffi_load);
    return libffi.ready;
}

static ffi_type *type_for(unsigned char code) {
    return code <= TY_UINT16 ? libffi.types[code] : NULL;
}
#define VOID_TYPE (libffi.types[TY_VOID])
#define PREP_CIF libffi.prep_cif
#define PREP_CIF_VAR libffi.prep_cif_var
#define CALL libffi.call
#else
static int libffi_available(void) { return 1; }

static ffi_type *type_for(unsigned char code) {
    switch (code) {
#define CASE(code, name) case code: return &name;
    TYPE_LIST(CASE)
#undef CASE
    default: return NULL;
    }
}
#define VOID_TYPE (&ffi_type_void)
#define PREP_CIF ffi_prep_cif
#define PREP_CIF_VAR ffi_prep_cif_var
#define CALL ffi_call
#endif

void *oximond_ffi_open(const char *name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

int oximond_ffi_errno(void) {
    return errno;
}

void oximond_ffi_set_errno(int value) {
    errno = value;
}

void oximond_ffi_close(void *handle) {
    dlclose(handle);
}

/* The executable does not link libm (see crates/oximond-cli/src/libm.rs), so
 * a lookup in the process namespace also searches it, as it did when libm was
 * a needed library. Both handles are opened once and kept: each dlopen of
 * libm takes a reference that a later lookup would otherwise leak. */
static void *process_handle;
static void *libm_handle;
static pthread_once_t process_once = PTHREAD_ONCE_INIT;
static pthread_once_t libm_once = PTHREAD_ONCE_INIT;

static void process_load(void) { process_handle = dlopen(NULL, RTLD_LAZY); }

static void libm_load(void) { libm_handle = dlopen("libm.so.6", RTLD_LAZY | RTLD_LOCAL); }

void *oximond_ffi_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    if (!symbol) {
        pthread_once(&process_once, process_load);
        if (handle == process_handle) {
            pthread_once(&libm_once, libm_load);
            if (libm_handle) symbol = dlsym(libm_handle, name);
            if (symbol) dlerror();
        }
    }
    return symbol;
}

const char *oximond_ffi_error(void) {
    return dlerror();
}

int oximond_ffi_call(void *function, unsigned char result_type,
                     const unsigned char *arg_types, void **arg_values,
                     size_t arg_count, size_t fixed_count, void *result) {
    ffi_cif cif;
    if (!libffi_available()) return -4;
    ffi_type *return_type = type_for(result_type);
    if (!function || !return_type || arg_count > UINT32_MAX || fixed_count > arg_count)
        return -1;

    ffi_type **types = calloc(arg_count ? arg_count : 1, sizeof(*types));
    if (!types) return -2;
    for (size_t i = 0; i < arg_count; i++) {
        types[i] = type_for(arg_types[i]);
        if (!types[i] || types[i] == VOID_TYPE) {
            free(types);
            return -1;
        }
    }

    ffi_status status;
    if (fixed_count < arg_count)
        status = PREP_CIF_VAR(&cif, FFI_DEFAULT_ABI, (unsigned)fixed_count,
                                  (unsigned)arg_count, return_type, types);
    else
        status = PREP_CIF(&cif, FFI_DEFAULT_ABI, (unsigned)arg_count,
                              return_type, types);
    if (status == FFI_OK) CALL(&cif, FFI_FN(function), result, arg_values);
    free(types);
    return status == FFI_OK ? 0 : -3;
}
