#include <stddef.h>
#include <stdint.h>

/* The Rust API remains linkable on targets without this native backend. */
void *oximond_ffi_open(const char *name) {
    (void)name;
    return NULL;
}

void oximond_ffi_close(void *handle) { (void)handle; }
void *oximond_ffi_symbol(void *handle, const char *name) {
    (void)handle;
    (void)name;
    return NULL;
}
const char *oximond_ffi_error(void) {
    return "native FFI is unsupported on this target";
}
int oximond_ffi_errno(void) { return 0; }
void oximond_ffi_set_errno(int value) { (void)value; }
int oximond_ffi_call(void *function, unsigned char result_type,
                     const unsigned char *arg_types, void **arg_values,
                     size_t arg_count, size_t fixed_count, void *result) {
    (void)function;
    (void)result_type;
    (void)arg_types;
    (void)arg_values;
    (void)arg_count;
    (void)fixed_count;
    (void)result;
    return -1;
}
