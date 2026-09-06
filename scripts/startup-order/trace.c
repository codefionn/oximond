/* Single-steps a program and prints every instruction address it executes,
 * once each, as an offset from the load base. usage:
 *   trace OUT BASE_HEX PROGRAM [ARGS...]
 * ASLR is off for the child, so a PIE loads at 0x555555554000. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/personality.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    uint64_t base = strtoull(argv[2], 0, 16);
    pid_t pid = fork();
    if (pid == 0) {
        personality(ADDR_NO_RANDOMIZE);
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        execv(argv[3], argv + 3);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    const uint64_t span = 0x2000000;
    unsigned char *seen = calloc(span, 1);
    FILE *out = fopen(argv[1], "w");
    for (;;) {
        struct user_regs_struct regs;
        ptrace(PTRACE_GETREGS, pid, 0, &regs);
        uint64_t offset = regs.rip - base;
        if (offset < span && !seen[offset]) {
            seen[offset] = 1;
            fprintf(out, "%lx\n", offset);
        }
        if (ptrace(PTRACE_SINGLESTEP, pid, 0, 0) < 0) break;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) || WIFSIGNALED(status)) break;
    }
    fclose(out);
    return 0;
}
