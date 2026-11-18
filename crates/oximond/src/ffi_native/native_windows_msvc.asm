; oximond_win64_invoke for MSVC, which has no inline assembly on x64. It is the
; trampoline in native_windows.c, instruction for instruction; keep them the same.
;
; oximond_win64_invoke(function, slots, count, result) copies `count` slots
; into the outgoing argument area, calls `function`, and stores RAX at
; result[0] and XMM0 at result[1].

.code

oximond_win64_invoke PROC FRAME
    push rbp
    .pushreg rbp
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    push r12
    .pushreg r12
    push r13
    .pushreg r13
    mov rbp, rsp
    .setframe rbp, 0
    .endprolog
    mov r12, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r13, r9
    ; Five pushes realigned RSP to 16; the area is a multiple of 16 holding
    ; at least the four shadow slots.
    mov rax, r8
    cmp rax, 4
    jae have_count
    mov eax, 4
have_count:
    lea rax, [rax*8 + 15]
    and rax, -16
    sub rsp, rax
    xor ecx, ecx
copy_slot:
    cmp rcx, rdi
    jae copied
    mov rax, [rsi + rcx*8]
    mov [rsp + rcx*8], rax
    inc rcx
    jmp copy_slot
copied:
    mov rcx, [rsp]
    mov rdx, [rsp + 8]
    mov r8, [rsp + 16]
    mov r9, [rsp + 24]
    movq xmm0, qword ptr [rsp]
    movq xmm1, qword ptr [rsp + 8]
    movq xmm2, qword ptr [rsp + 16]
    movq xmm3, qword ptr [rsp + 24]
    call r12
    mov [r13], rax
    movq qword ptr [r13 + 8], xmm0
    lea rsp, [rbp]
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbp
    ret
oximond_win64_invoke ENDP

END
