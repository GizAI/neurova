.intel_syntax noprefix

/* AURORA-LLM v0.1: freestanding x86-64 assembly Llama-family inference server.
 * Runtime dependencies: Linux syscall ABI only.
 * Model format: ALI v1, row-wise Q8 weights + F32 norms/RoPE tables.
 */

/* Linux syscall numbers */
.equ SYS_read,              0
.equ SYS_write,             1
.equ SYS_close,             3
.equ SYS_lseek,             8
.equ SYS_mmap,              9
.equ SYS_munmap,           11
.equ SYS_socket,           41
.equ SYS_accept,           43
.equ SYS_bind,             49
.equ SYS_listen,           50
.equ SYS_setsockopt,       54
.equ SYS_fork,             57
.equ SYS_exit,             60
.equ SYS_sched_setaffinity,203
.equ SYS_sched_getaffinity,204
.equ SYS_openat,          257

.equ AT_FDCWD,            -100
.equ O_RDONLY,              0
.equ SEEK_END,              2
.equ PROT_READ,             1
.equ PROT_WRITE,            2
.equ MAP_SHARED,            1
.equ MAP_PRIVATE,           2
.equ MAP_ANONYMOUS,      0x20
.equ AF_INET,               2
.equ SOCK_STREAM,           1
.equ SOL_SOCKET,            1
.equ SO_REUSEADDR,          2

.equ HEADER_SIZE,          64
.equ LAYER_DESC_SIZE,      80
.equ MAX_LAYERS,          256
.equ MAX_PROMPT_TOKENS,   512
.equ MAX_GEN_TOKENS,      128
.equ HTTP_BUF_SIZE,     65536

/* Shared job layout */
.equ JOB_SEQ,               0
.equ JOB_W,                 8
.equ JOB_X,                16
.equ JOB_Y,                24
.equ JOB_OUT,              32
.equ JOB_IN,               40
.equ JOB_STRIDE,           48
.equ JOB_NWORK,            56
.equ JOB_DONE,             64
.equ JOB_STOP,             72
.equ JOB_READY,            80
.equ JOB_SIZE,            128

/* Layer descriptor */
.equ LD_ATTN_NORM,          0
.equ LD_WQ,                 8
.equ LD_WK,                16
.equ LD_WV,                24
.equ LD_WO,                32
.equ LD_FFN_NORM,          40
.equ LD_W1,                48
.equ LD_W3,                56
.equ LD_W2,                64

.section .rodata
.align 16
banner: .ascii "AURORA-LLM asm inference server\n"
banner_end:
usage: .ascii "usage: aurora-llm MODEL.ali [workers] [port]\n"
default_model: .asciz "/model.ali"
usage_end:
err_open: .ascii "error: cannot open model\n"
err_open_end:
err_magic: .ascii "error: invalid ALI model\n"
err_magic_end:
err_mmap: .ascii "error: mmap failed\n"
err_mmap_end:
err_socket: .ascii "error: socket/bind/listen failed\n"
err_socket_end:
ready1: .ascii "ready: port "
ready1_end:
ready2: .ascii ", workers "
ready2_end:
nl: .ascii "\n"

http_ok: .ascii "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"tokens\":["
http_ok_end:
http_tail: .ascii "],\"engine\":\"aurora-llm-asm\"}\n"
http_tail_end:
http_bad: .ascii "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"expected JSON tokens array\"}\n"
http_bad_end:
comma: .ascii ","

key_tokens: .ascii "\"tokens\""
key_tokens_end:
key_max: .ascii "\"max_tokens\""
key_max_end:

.align 4
f_eps:       .float 0.00001
f_one:       .float 1.0
f_half:      .float 0.5
f_log2e:     .float 1.4426950408889634
f_ln2:       .float 0.6931471805599453
f_ln2_2:     .float 0.2402265069591007
f_ln2_3:     .float 0.05550410866482158
f_ln2_4:     .float 0.009618129107628477
f_ln2_5:     .float 0.0013333558146428442
f_minexp:    .float -80.0
f_maxexp:    .float 80.0

.section .bss
.align 64
g_model_base:      .quad 0
g_model_len:       .quad 0
g_model_version:   .quad 0
g_dim:             .quad 0
g_hidden:          .quad 0
g_layers_n:        .quad 0
g_heads:           .quad 0
g_kv_heads:        .quad 0
g_vocab:           .quad 0
g_max_seq:         .quad 0
g_head_dim:        .quad 0
g_q_dim:           .quad 0
g_kv_dim:          .quad 0
g_stride_dim:      .quad 0
g_stride_q:        .quad 0
g_stride_hidden:   .quad 0
g_att_scale:       .long 0
.align 8
g_rms_norm_eps:    .long 0
.align 8
g_eos1:            .quad 0
g_eos2:            .quad 0
g_eos_count:       .long 0
g_embeddings:      .quad 0
g_final_norm:      .quad 0
g_output:          .quad 0
g_rope_cos:        .quad 0
g_rope_sin:        .quad 0

g_arena:           .quad 0
g_arena_len:       .quad 0
g_job:             .quad 0
g_x:               .quad 0
g_xb:              .quad 0
g_xb2:             .quad 0
g_q:               .quad 0
g_k:               .quad 0
g_v:               .quad 0
g_att:             .quad 0
g_h1:              .quad 0
g_h2:              .quad 0
g_logits:           .quad 0
g_scores:           .quad 0
g_kcache:           .quad 0
g_vcache:           .quad 0

g_workers:          .quad 0
g_worker_id:        .quad 0
g_allowed_workers:  .quad 0
g_physical_workers: .quad 0
g_port:             .quad 0
.align 64
g_cpu_order:        .skip 128
g_smt_order:        .skip 128
g_core_keys:        .skip 256
g_use_avx512:       .long 0
.align 8

g_listen_fd:        .quad 0
g_http_buf:         .quad 0
g_prompt_buf:       .quad 0
g_output_tokens:    .quad 0

.align 64
layer_descs:        .skip MAX_LAYERS*LAYER_DESC_SIZE

.section .text
.global _start

/* -------------------------------------------------------------------------- */
/* Syscall / utility helpers                                                  */
/* -------------------------------------------------------------------------- */

write_all:
    /* rdi=fd, rsi=buf, rdx=len */
    push rbx
    mov rbx, rdx
.wa_loop:
    test rbx, rbx
    jz .wa_done
    mov eax, SYS_write
    mov rdx, rbx
    syscall
    test rax, rax
    jle .wa_done
    add rsi, rax
    sub rbx, rax
    jmp .wa_loop
.wa_done:
    pop rbx
    ret

print_stderr:
    /* rsi ptr, rdx len */
    mov edi, 2
    jmp write_all

exit_now:
    mov eax, SYS_exit
    syscall
    ud2

parse_u64:
    /* rdi = NUL terminated ascii, returns rax, 0 if absent/invalid */
    xor eax, eax
    xor ecx, ecx
.pu_loop:
    movzx edx, byte ptr [rdi+rcx]
    cmp dl, '0'
    jb .pu_done
    cmp dl, '9'
    ja .pu_done
    imul rax, rax, 10
    sub edx, '0'
    add rax, rdx
    inc rcx
    jmp .pu_loop
.pu_done:
    ret

u64_to_ascii:
    /* rdi=value, rsi=dest; returns rax=len */
    push rbx
    mov rax, rdi
    mov rbx, 10
    xor ecx, ecx
    test rax, rax
    jnz .uta_digits
    mov byte ptr [rsi], '0'
    mov eax, 1
    pop rbx
    ret
.uta_digits:
    sub rsp, 32
.uta_loop:
    xor edx, edx
    div rbx
    add dl, '0'
    mov byte ptr [rsp+rcx], dl
    inc ecx
    test rax, rax
    jnz .uta_loop
    xor edx, edx
.uta_rev:
    mov al, byte ptr [rsp+rcx-1]
    mov byte ptr [rsi+rdx], al
    inc edx
    dec ecx
    jnz .uta_rev
    mov eax, edx
    add rsp, 32
    pop rbx
    ret

align64_rax:
    add rax, 63
    and rax, -64
    ret

memcpy_bytes:
    /* rdi dst, rsi src, rdx n */
    mov rcx, rdx
    rep movsb
    ret

memzero_bytes:
    /* rdi dst, rdx n */
    xor eax, eax
    mov rcx, rdx
    rep stosb
    ret

find_bytes:
    /* rdi hay, rsi haylen, rdx needle, rcx needlelen => rax ptr or 0 */
    push rbx
    push r12
    mov r12, rcx
    xor rax, rax
    cmp rsi, r12
    jb .fb_none
    mov rbx, rsi
    sub rbx, r12
    inc rbx
.fb_outer:
    test rbx, rbx
    jz .fb_none
    push rdi
    push rsi
    push rcx
    mov r8, rdi
    mov r9, rdx
    mov rcx, r12
.fb_inner:
    mov al, byte ptr [r8]
    cmp al, byte ptr [r9]
    jne .fb_miss
    inc r8
    inc r9
    dec rcx
    jnz .fb_inner
    pop rcx
    pop rsi
    pop rdi
    mov rax, rdi
    jmp .fb_done
.fb_miss:
    pop rcx
    pop rsi
    pop rdi
    inc rdi
    dec rbx
    jmp .fb_outer
.fb_none:
    xor eax, eax
.fb_done:
    pop r12
    pop rbx
    ret

/* -------------------------------------------------------------------------- */
/* Model mapping and memory arena                                             */
/* -------------------------------------------------------------------------- */

map_model:
    /* rdi = path. returns rax=0 success, nonzero error */
    push rbx
    mov rsi, rdi
    mov eax, SYS_openat
    mov edi, AT_FDCWD
    xor edx, edx
    xor r10d, r10d
    syscall
    test rax, rax
    js .mm_open_fail
    mov rbx, rax

    mov eax, SYS_lseek
    mov rdi, rbx
    xor esi, esi
    mov edx, SEEK_END
    syscall
    test rax, rax
    jle .mm_open_fail2
    mov [rip+g_model_len], rax

    mov rsi, rax
    xor edi, edi
    mov edx, PROT_READ
    mov r10d, MAP_PRIVATE
    mov r8, rbx
    xor r9d, r9d
    mov eax, SYS_mmap
    syscall
    cmp rax, -4095
    jae .mm_map_fail
    mov [rip+g_model_base], rax

    mov eax, SYS_close
    mov rdi, rbx
    syscall

    mov rdi, [rip+g_model_base]
    mov rax, qword ptr [rdi]
    mov rdx, 0x3130494c41525541 /* "AURALI01" little endian */
    cmp rax, rdx
    jne .mm_magic_fail
    mov eax, dword ptr [rdi+8]
    cmp eax, 1
    je .mm_version_ok
    cmp eax, 2
    jne .mm_magic_fail
.mm_version_ok:
    mov [rip+g_model_version], rax

    mov eax, dword ptr [rdi+12]
    mov [rip+g_dim], rax
    mov eax, dword ptr [rdi+16]
    mov [rip+g_hidden], rax
    mov eax, dword ptr [rdi+20]
    mov [rip+g_layers_n], rax
    mov eax, dword ptr [rdi+24]
    mov [rip+g_heads], rax
    mov eax, dword ptr [rdi+28]
    mov [rip+g_kv_heads], rax
    mov eax, dword ptr [rdi+32]
    mov [rip+g_vocab], rax
    mov eax, dword ptr [rdi+36]
    mov [rip+g_max_seq], rax
    movss xmm0, dword ptr [rip+f_eps]
    movss xmm1, dword ptr [rdi+44]
    vxorps xmm2, xmm2, xmm2
    vucomiss xmm1, xmm2
    jbe .mm_rms_eps_default
    movss xmm0, xmm1
.mm_rms_eps_default:
    movss dword ptr [rip+g_rms_norm_eps], xmm0
    mov eax, dword ptr [rdi+52]
    mov [rip+g_eos1], rax
    mov eax, dword ptr [rdi+56]
    mov [rip+g_eos2], rax
    mov eax, dword ptr [rdi+60]
    mov [rip+g_eos_count], rax

    mov rax, [rip+g_layers_n]
    cmp rax, MAX_LAYERS
    ja .mm_magic_fail
    mov rax, [rip+g_dim]
    test rax, 7
    jnz .mm_magic_fail
    mov rcx, [rip+g_heads]
    test rcx, rcx
    jz .mm_magic_fail
    xor edx, edx
    div rcx
    mov [rip+g_head_dim], rax
    mov eax, dword ptr [rdi+48]
    test eax, eax
    jz .mm_head_dim_ready
    mov [rip+g_head_dim], rax
.mm_head_dim_ready:
    mov rax, [rip+g_head_dim]
    test rax, 1
    jnz .mm_magic_fail
    imul rax, [rip+g_heads]
    mov [rip+g_q_dim], rax
    mov rax, [rip+g_head_dim]
    mov rcx, [rip+g_kv_heads]
    imul rax, rcx
    mov [rip+g_kv_dim], rax

    /* row strides = align64(4 + input_dim) */
    mov rax, [rip+g_dim]
    add rax, 4
    call align64_rax
    mov [rip+g_stride_dim], rax
    mov rax, [rip+g_q_dim]
    add rax, 4
    call align64_rax
    mov [rip+g_stride_q], rax
    mov rax, [rip+g_hidden]
    add rax, 4
    call align64_rax
    mov [rip+g_stride_hidden], rax

    /* attention scale = 1/sqrt(head_dim) */
    mov rax, [rip+g_head_dim]
    cvtsi2ss xmm0, rax
    sqrtss xmm0, xmm0
    movss xmm1, dword ptr [rip+f_one]
    divss xmm1, xmm0
    movss dword ptr [rip+g_att_scale], xmm1

    call parse_model_layout
    test rax, rax
    jnz .mm_magic_fail
    xor eax, eax
    pop rbx
    ret

.mm_open_fail2:
    mov eax, SYS_close
    mov rdi, rbx
    syscall
.mm_open_fail:
    mov eax, 1
    pop rbx
    ret
.mm_map_fail:
    mov eax, SYS_close
    mov rdi, rbx
    syscall
    mov eax, 2
    pop rbx
    ret
.mm_magic_fail:
    mov eax, 3
    pop rbx
    ret

q8_matrix_bytes:
    /* rdi=output rows, rsi=input cols. v1=row-major; v2=8x16 interleaved. */
    cmp qword ptr [rip+g_model_version], 2
    jne .qmb_v1
    mov rax, rsi
    shl rax, 3                     /* 8 * input cols */
    add rax, 64                    /* scale/header cache line per 8 rows */
    mov rcx, rdi
    shr rcx, 3                     /* output groups of 8 */
    imul rax, rcx
    ret
.qmb_v1:
    mov rax, rsi
    add rax, 67                    /* align64(4 + input cols) */
    and rax, -64
    imul rax, rdi
    ret

parse_model_layout:
    /* compute pointers from the fixed ALI v1 layout. rax=0 ok */
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rbx, [rip+g_model_base]
    lea r12, [rbx+HEADER_SIZE]
    add r12, 63
    and r12, -64

    mov [rip+g_embeddings], r12
    mov rax, [rip+g_vocab]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    xor r13d, r13d
    lea r14, [rip+layer_descs]
.pml_layer:
    cmp r13, [rip+g_layers_n]
    jae .pml_layers_done

    /* attn norm */
    mov [r14+LD_ATTN_NORM], r12
    mov rax, [rip+g_dim]
    shl rax, 2
    add r12, rax
    add r12, 63
    and r12, -64

    /* wq */
    mov [r14+LD_WQ], r12
    mov rdi, [rip+g_q_dim]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    /* wk */
    mov [r14+LD_WK], r12
    mov rdi, [rip+g_kv_dim]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    /* wv */
    mov [r14+LD_WV], r12
    mov rdi, [rip+g_kv_dim]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    /* wo */
    mov [r14+LD_WO], r12
    mov rdi, [rip+g_dim]
    mov rsi, [rip+g_q_dim]
    call q8_matrix_bytes
    add r12, rax

    /* ffn norm */
    mov [r14+LD_FFN_NORM], r12
    mov rax, [rip+g_dim]
    shl rax, 2
    add r12, rax
    add r12, 63
    and r12, -64

    /* w1 gate */
    mov [r14+LD_W1], r12
    mov rdi, [rip+g_hidden]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    /* w3 up */
    mov [r14+LD_W3], r12
    mov rdi, [rip+g_hidden]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    /* w2 down */
    mov [r14+LD_W2], r12
    mov rdi, [rip+g_dim]
    mov rsi, [rip+g_hidden]
    call q8_matrix_bytes
    add r12, rax

    add r14, LAYER_DESC_SIZE
    inc r13
    jmp .pml_layer

.pml_layers_done:
    mov [rip+g_final_norm], r12
    mov rax, [rip+g_dim]
    shl rax, 2
    add r12, rax
    add r12, 63
    and r12, -64

    mov [rip+g_output], r12
    mov rdi, [rip+g_vocab]
    mov rsi, [rip+g_dim]
    call q8_matrix_bytes
    add r12, rax

    add r12, 63
    and r12, -64
    mov [rip+g_rope_cos], r12
    mov rax, [rip+g_max_seq]
    mov rcx, [rip+g_head_dim]
    shr rcx, 1
    imul rax, rcx
    shl rax, 2
    add r12, rax
    add r12, 63
    and r12, -64
    mov [rip+g_rope_sin], r12
    mov rax, [rip+g_max_seq]
    mov rcx, [rip+g_head_dim]
    shr rcx, 1
    imul rax, rcx
    shl rax, 2
    add r12, rax

    mov rax, [rip+g_model_base]
    add rax, [rip+g_model_len]
    cmp r12, rax
    ja .pml_bad
    xor eax, eax
    jmp .pml_ret
.pml_bad:
    mov eax, 1
.pml_ret:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

alloc_arena:
    /* compute shared arena size and map it */
    push rbx
    push r12
    xor r12d, r12d
    add r12, JOB_SIZE
    add r12, 63
    and r12, -64

    /* vectors: 3*dim + 2*q_dim + 2*kv_dim + 2*hidden + vocab + maxseq floats */
    mov rax, [rip+g_dim]
    imul rax, 3
    add r12, rax
    mov rax, [rip+g_q_dim]
    shl rax, 1
    add r12, rax
    mov rax, [rip+g_kv_dim]
    shl rax, 1
    add r12, rax
    mov rax, [rip+g_hidden]
    shl rax, 1
    add r12, rax
    add r12, [rip+g_vocab]
    add r12, [rip+g_max_seq]
    shl r12, 2
    add r12, 4096

    /* prompt/output/http buffers */
    add r12, HTTP_BUF_SIZE
    add r12, MAX_PROMPT_TOKENS*8
    add r12, MAX_GEN_TOKENS*8

    /* KV caches: layers * max_seq * kv_dim * 4 * 2 */
    mov rax, [rip+g_layers_n]
    imul rax, [rip+g_max_seq]
    imul rax, [rip+g_kv_dim]
    shl rax, 3
    add r12, rax

    add r12, 4095
    and r12, -4096
    mov [rip+g_arena_len], r12

    xor edi, edi
    mov rsi, r12
    mov edx, PROT_READ|PROT_WRITE
    mov r10d, MAP_SHARED|MAP_ANONYMOUS
    mov r8, -1
    xor r9d, r9d
    mov eax, SYS_mmap
    syscall
    cmp rax, -4095
    jae .aa_fail
    mov [rip+g_arena], rax
    mov rbx, rax

    /* assign pointers, always 64-byte aligned */
    mov [rip+g_job], rbx
    add rbx, JOB_SIZE
    add rbx, 63
    and rbx, -64

    mov [rip+g_x], rbx
    mov rax, [rip+g_dim]
    shl rax, 2
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_xb], rbx
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_xb2], rbx
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_q], rbx
    mov rax, [rip+g_q_dim]
    shl rax, 2
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_att], rbx
    add rbx, rax
    add rbx,63; and rbx,-64

    mov [rip+g_k], rbx
    mov rax, [rip+g_kv_dim]
    shl rax, 2
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_v], rbx
    add rbx, rax
    add rbx,63; and rbx,-64

    mov [rip+g_h1], rbx
    mov rax, [rip+g_hidden]
    shl rax, 2
    add rbx, rax
    add rbx,63; and rbx,-64
    mov [rip+g_h2], rbx
    add rbx, rax
    add rbx,63; and rbx,-64

    mov [rip+g_logits], rbx
    mov rax, [rip+g_vocab]
    shl rax,2
    add rbx,rax
    add rbx,63; and rbx,-64

    mov [rip+g_scores], rbx
    mov rax, [rip+g_max_seq]
    shl rax,2
    add rbx,rax
    add rbx,63; and rbx,-64

    mov [rip+g_http_buf], rbx
    add rbx, HTTP_BUF_SIZE
    add rbx,63; and rbx,-64
    mov [rip+g_prompt_buf], rbx
    add rbx, MAX_PROMPT_TOKENS*8
    add rbx,63; and rbx,-64
    mov [rip+g_output_tokens], rbx
    add rbx, MAX_GEN_TOKENS*8
    add rbx,63; and rbx,-64

    mov [rip+g_kcache], rbx
    mov rax, [rip+g_layers_n]
    imul rax, [rip+g_max_seq]
    imul rax, [rip+g_kv_dim]
    shl rax,2
    add rbx,rax
    add rbx,63; and rbx,-64
    mov [rip+g_vcache], rbx

    /* zero job structure */
    mov rdi, [rip+g_job]
    mov edx, JOB_SIZE
    call memzero_bytes
    xor eax, eax
    pop r12
    pop rbx
    ret
.aa_fail:
    mov eax, 1
    pop r12
    pop rbx
    ret

/* -------------------------------------------------------------------------- */
/* Multicore Q8 matrix-vector engine                                          */
/* -------------------------------------------------------------------------- */

spawn_workers:
    /* g_workers total includes coordinator. children IDs 1..N-1 */
    push rbx
    mov rbx, 1
.sw_loop:
    cmp rbx, [rip+g_workers]
    jae .sw_parent_done
    mov eax, SYS_fork
    syscall
    test rax, rax
    js .sw_fork_failed
    jz .sw_child
    inc rbx
    jmp .sw_loop
.sw_child:
    mov [rip+g_worker_id], rbx
    mov rdi, rbx
    call pin_current_cpu
    mov rdi, rbx
    call worker_loop
    mov edi, 0
    jmp exit_now
.sw_fork_failed:
    /* rbx equals the number of workers that actually exist: IDs 0..rbx-1. */
    mov [rip+g_workers], rbx
.sw_parent_done:
    /* Do not publish a first matvec until every child has captured JOB_SEQ. */
    mov rdx, [rip+g_job]
    mov rax, [rip+g_workers]
    dec rax
.sw_wait_ready:
    cmp qword ptr [rdx+JOB_READY], rax
    jae .sw_all_ready
    pause
    jmp .sw_wait_ready
.sw_all_ready:
    xor edi, edi
    call pin_current_cpu
    pop rbx
    ret

detect_avx512:
    /* Return 1 when AVX-512F/DQ and full ZMM OS state are available. */
    push rbx
    xor r8d, r8d
    xor eax, eax
    cpuid
    cmp eax, 7
    jb .dav_done
    mov eax, 1
    cpuid
    mov eax, ecx
    and eax, (1<<27) | (1<<28)      /* OSXSAVE + AVX */
    cmp eax, (1<<27) | (1<<28)
    jne .dav_done
    xor ecx, ecx
    xgetbv
    and eax, 0xe6                   /* XMM,YMM,opmask,ZMM_hi256,hi16_ZMM */
    cmp eax, 0xe6
    jne .dav_done
    mov eax, 7
    xor ecx, ecx
    cpuid
    bt ebx, 16                      /* AVX512F */
    jnc .dav_done
    bt ebx, 17                      /* AVX512DQ (vextractf32x8) */
    jnc .dav_done
    mov r8d, 1
.dav_done:
    mov eax, r8d
    pop rbx
    ret

detect_workers:
    /* Build a physical-core-first CPU order from the current affinity mask.
       AMD 0x8000001e exposes core IDs directly. Generic x86 falls back to
       affinity order. Return the preferred auto-worker count. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32                     /* original mask + temporary one-CPU mask */
    pxor xmm0, xmm0
    movdqu xmmword ptr [rsp], xmm0
    mov eax, SYS_sched_getaffinity
    xor edi, edi
    mov esi, 16
    lea rdx, [rsp]
    syscall
    test rax, rax
    js .dw_one
    mov r12, [rsp]
    mov r13, [rsp+8]

    /* Prefer AMD's topology leaf when available. */
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x8000001e
    jb .dw_flat_order

    xor r14d, r14d                 /* unique physical cores */
    xor r15d, r15d                 /* SMT duplicates */
    xor ebp, ebp                    /* candidate CPU 0..127 */
.dw_topo_scan:
    cmp ebp, 128
    jae .dw_topo_done
    cmp ebp, 64
    jae .dw_test_hi
    bt r12, rbp
    jnc .dw_topo_next
    jmp .dw_candidate
.dw_test_hi:
    mov ecx, ebp
    sub ecx, 64
    bt r13, rcx
    jnc .dw_topo_next

.dw_candidate:
    mov qword ptr [rsp+16], 0
    mov qword ptr [rsp+24], 0
    mov ecx, ebp
    cmp ecx, 64
    jb .dw_mask_low
    sub ecx, 64
    mov rax, 1
    shl rax, cl
    mov [rsp+24], rax
    jmp .dw_pin_probe
.dw_mask_low:
    mov rax, 1
    shl rax, cl
    mov [rsp+16], rax
.dw_pin_probe:
    mov eax, SYS_sched_setaffinity
    xor edi, edi
    mov esi, 16
    lea rdx, [rsp+16]
    syscall
    test rax, rax
    js .dw_topo_next

    mov eax, 0x8000001e
    xor ecx, ecx
    cpuid
    /* key = node_id:core_id; enough to distinguish package-local cores. */
    and ebx, 0xff
    and ecx, 0xff
    shl ecx, 8
    or ebx, ecx

    xor ecx, ecx
.dw_key_find:
    cmp rcx, r14
    jae .dw_new_core
    lea rax, [rip+g_core_keys]
    cmp bx, word ptr [rax+rcx*2]
    je .dw_smt_cpu
    inc ecx
    jmp .dw_key_find
.dw_new_core:
    lea rax, [rip+g_core_keys]
    mov word ptr [rax+r14*2], bx
    lea rax, [rip+g_cpu_order]
    mov byte ptr [rax+r14], bpl
    inc r14
    jmp .dw_topo_next
.dw_smt_cpu:
    lea rax, [rip+g_smt_order]
    mov byte ptr [rax+r15], bpl
    inc r15
.dw_topo_next:
    inc ebp
    jmp .dw_topo_scan

.dw_topo_done:
    /* Append SMT siblings only after every unique core representative. */
    xor ecx, ecx
.dw_append_smt:
    cmp rcx, r15
    jae .dw_order_ready
    lea rax, [rip+g_smt_order]
    movzx edx, byte ptr [rax+rcx]
    lea rax, [rip+g_cpu_order]
    lea r8, [r14+rcx]
    mov byte ptr [rax+r8], dl
    inc rcx
    jmp .dw_append_smt
.dw_order_ready:
    lea rax, [r14+r15]
    mov [rip+g_allowed_workers], rax
    mov [rip+g_physical_workers], r14
    test r14, r14
    jz .dw_use_total
    mov rax, r14                    /* auto: one worker per physical core */
    jmp .dw_restore
.dw_use_total:
    mov rax, [rip+g_allowed_workers]
    jmp .dw_restore

.dw_flat_order:
    /* Generic fallback: preserve the kernel affinity-mask ordering. */
    xor r14d, r14d
    xor ebp, ebp
.dw_flat_scan:
    cmp ebp, 128
    jae .dw_flat_done
    cmp ebp, 64
    jae .dw_flat_hi
    bt r12, rbp
    jnc .dw_flat_next
    jmp .dw_flat_add
.dw_flat_hi:
    mov ecx, ebp
    sub ecx, 64
    bt r13, rcx
    jnc .dw_flat_next
.dw_flat_add:
    lea rax, [rip+g_cpu_order]
    mov byte ptr [rax+r14], bpl
    inc r14
.dw_flat_next:
    inc ebp
    jmp .dw_flat_scan
.dw_flat_done:
    mov [rip+g_allowed_workers], r14
    mov [rip+g_physical_workers], r14
    mov rax, r14

.dw_restore:
    push rax
    mov eax, SYS_sched_setaffinity
    xor edi, edi
    mov esi, 16
    lea rdx, [rsp+8]                /* original mask moved by push */
    syscall
    pop rax
    test rax, rax
    jnz .dw_ret
.dw_one:
    mov eax, 1
    mov qword ptr [rip+g_allowed_workers], 1
    mov qword ptr [rip+g_physical_workers], 1
    mov byte ptr [rip+g_cpu_order], 0
.dw_ret:
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

pin_current_cpu:
    /* Pin logical worker ID using the physical-core-first order built above. */
    cmp rdi, [rip+g_allowed_workers]
    jae .pcc_done
    sub rsp, 16
    pxor xmm0, xmm0
    movdqu xmmword ptr [rsp], xmm0
    lea rax, [rip+g_cpu_order]
    movzx ecx, byte ptr [rax+rdi]
    cmp ecx, 64
    jae .pcc_high
    mov rax, 1
    shl rax, cl
    mov [rsp], rax
    jmp .pcc_call
.pcc_high:
    sub ecx, 64
    mov rax, 1
    shl rax, cl
    mov [rsp+8], rax
.pcc_call:
    mov eax, SYS_sched_setaffinity
    xor edi, edi
    mov esi, 16
    mov rdx, rsp
    syscall
    add rsp, 16
.pcc_done:
    ret

tail_rows_cost:
    /* rdi=output rows, rsi=workers, rdx=row batch -> sum of scalar-tail rows
       after the same floor-partitioning used by q8_worker_do. */
    push r12
    push r13
    push r14
    mov r12, rdi                    /* output rows */
    mov r13, rdx                    /* row batch */
    xor r14d, r14d                  /* total tails */
    xor r8d, r8d                    /* previous boundary */
    mov r9, 1
.trc_loop:
    cmp r9, rsi
    ja .trc_done
    mov rax, r12
    mul r9
    div rsi                         /* boundary=floor(rows*i/workers) */
    mov r10, rax
    sub rax, r8                     /* shard length */
    xor edx, edx
    div r13
    add r14, rdx                    /* shard_len % row_batch */
    mov r8, r10
    inc r9
    jmp .trc_loop
.trc_done:
    mov rax, r14
    pop r14
    pop r13
    pop r12
    ret

projection_tail_score:
    /* rdi=workers, rsi=row batch -> weighted scalar-tail rows per token.
       Per transformer layer: 2*dim + 2*hidden + q_dim + 2*kv_dim; plus LM head. */
    push rbx
    push rbp
    push r12
    push r13
    mov r12, rdi
    mov r13, rsi
    xor ebp, ebp

    mov rdi, [rip+g_dim]
    mov rsi, r12
    mov rdx, r13
    call tail_rows_cost
    lea rbp, [rbp+rax*2]

    mov rdi, [rip+g_hidden]
    mov rsi, r12
    mov rdx, r13
    call tail_rows_cost
    lea rbp, [rbp+rax*2]

    mov rdi, [rip+g_q_dim]
    mov rsi, r12
    mov rdx, r13
    call tail_rows_cost
    add rbp, rax

    mov rdi, [rip+g_kv_dim]
    mov rsi, r12
    mov rdx, r13
    call tail_rows_cost
    lea rbp, [rbp+rax*2]

    imul rbp, [rip+g_layers_n]
    mov rdi, [rip+g_vocab]
    mov rsi, r12
    mov rdx, r13
    call tail_rows_cost
    add rbp, rax
    mov rax, rbp
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

tune_auto_workers:
    /* ALI v2 removes row tails entirely. On the target Zen 5 memory system,
       six physical workers are the measured bandwidth/synchronization sweet
       spot; additional workers reduce decode throughput. */
    cmp qword ptr [rip+g_model_version], 2
    jne .taw_v1
    cmp dword ptr [rip+g_use_avx512], 0
    je .taw_ret
    /* The measured six-worker cap is specific to small Zen 5 cpusets.  Do
       not impose it on larger servers where memory channels/core count differ. */
    cmp rax, 8
    ja .taw_ret
    cmp rax, 6
    jbe .taw_ret
    mov eax, 6
    ret
.taw_v1:
    /* v1: compare weighted scalar-tail work at the 6/7-worker boundary. */
    cmp rax, 7
    jne .taw_ret
    cmp dword ptr [rip+g_use_avx512], 0
    je .taw_ret
    push rbx
    mov edi, 7
    mov esi, 6
    call projection_tail_score
    mov rbx, rax                    /* score(7 workers, 6-row) */
    mov edi, 6
    mov esi, 8
    call projection_tail_score
    cmp rax, rbx
    jae .taw_keep7
    mov eax, 6
    pop rbx
    ret
.taw_keep7:
    mov eax, 7
    pop rbx
.taw_ret:
    ret

worker_loop:
    /* rdi=worker id */
    push r12
    push r13
    push r14
    mov r12, rdi
    mov r13, [rip+g_job]
    mov r14, [r13+JOB_SEQ]
    lock inc qword ptr [r13+JOB_READY]
.wl_spin:
    pause
    cmp qword ptr [r13+JOB_STOP], 0
    jne .wl_exit
    mov rax, [r13+JOB_SEQ]
    cmp rax, r14
    je .wl_spin
    mov r14, rax
    mov rdi, r12
    call q8_worker_do
    lock inc qword ptr [r13+JOB_DONE]
    jmp .wl_spin
.wl_exit:
    pop r14
    pop r13
    pop r12
    ret

q8_matvec:
    /* rdi=W, rsi=x, rdx=y, rcx=out, r8=in, r9=rowstride */
    push rbx
    push r12
    mov r12, [rip+g_job]
    mov [r12+JOB_W], rdi
    mov [r12+JOB_X], rsi
    mov [r12+JOB_Y], rdx
    mov [r12+JOB_OUT], rcx
    mov [r12+JOB_IN], r8
    mov [r12+JOB_STRIDE], r9
    mov rax, [rip+g_workers]
    mov [r12+JOB_NWORK], rax
    mov qword ptr [r12+JOB_DONE], 0
    /* x86 TSO orders descriptor stores before publishing JOB_SEQ. */
    inc qword ptr [r12+JOB_SEQ]

    xor edi, edi
    call q8_worker_do
    lock inc qword ptr [r12+JOB_DONE]
.qm_wait:
    mov rax, [r12+JOB_DONE]
    cmp rax, [r12+JOB_NWORK]
    jae .qm_done
    pause
    jmp .qm_wait
.qm_done:
    pop r12
    pop rbx
    ret

q8_worker_do:
    /* rdi=worker ID; ALI v2 uses an 8x16 interleaved Q8 stream. */
    cmp qword ptr [rip+g_model_version], 2
    je .qwd_v2
    cmp dword ptr [rip+g_use_avx512], 0
    jne q8_worker_do_avx512
    jmp q8_worker_do_avx2
.qwd_v2:
    cmp dword ptr [rip+g_use_avx512], 0
    jne q8_worker_do_v2_avx512
    jmp q8_worker_do_v2_avx2

q8_worker_do_v2_avx512:
    /* ALI v2: one contiguous block contains 8 rows. 64 input features are
       unrolled per iteration; every row keeps the same FP32 FMA order as v1. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov r10, [rbx+JOB_OUT]
    shr r10, 3                     /* total 8-row groups */
    mov rax, r10
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax                    /* start group */
    lea rax, [r15+1]
    mul r10
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax                    /* end group */
    cmp r12, r13
    jae .qv2_done
    mov r11, [rbx+JOB_IN]
    mov r14, r11
    shl r14, 3
    add r14, 64                    /* packed bytes per 8-row group */
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax                    /* packed group ptr */
    mov rax, r12
    shl rax, 5
    add rax, [rbx+JOB_Y]
    mov r15, rax                    /* y ptr */
    mov rdx, [rbx+JOB_X]
.qv2_group:
    vbroadcastss zmm16, dword ptr [rbp+0]
    vbroadcastss zmm17, dword ptr [rbp+4]
    vbroadcastss zmm18, dword ptr [rbp+8]
    vbroadcastss zmm19, dword ptr [rbp+12]
    vbroadcastss zmm20, dword ptr [rbp+16]
    vbroadcastss zmm21, dword ptr [rbp+20]
    vbroadcastss zmm22, dword ptr [rbp+24]
    vbroadcastss zmm23, dword ptr [rbp+28]
    vxorps zmm24, zmm24, zmm24
    vxorps zmm25, zmm25, zmm25
    vxorps zmm26, zmm26, zmm26
    vxorps zmm27, zmm27, zmm27
    vxorps zmm28, zmm28, zmm28
    vxorps zmm29, zmm29, zmm29
    vxorps zmm30, zmm30, zmm30
    vxorps zmm31, zmm31, zmm31
    xor ecx, ecx
.qv2_k64:
    lea rax, [rcx+64]
    cmp rax, r11
    ja .qv2_k16
    vmovups zmm1, zmmword ptr [rdx+rcx*4+0]
    vmovups zmm2, zmmword ptr [rdx+rcx*4+64]
    vmovups zmm3, zmmword ptr [rdx+rcx*4+128]
    vmovups zmm4, zmmword ptr [rdx+rcx*4+192]
    lea r8, [rbp+rcx*8+64]
    vpmovsxbd zmm0, xmmword ptr [r8+0]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+16]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+32]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+48]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+64]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+80]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+96]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+112]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+128]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+144]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+160]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+176]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+192]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+208]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+224]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+240]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+256]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+272]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+288]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+304]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+320]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+336]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+352]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+368]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+384]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+400]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+416]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+432]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+448]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+464]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+480]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+496]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm4
    add rcx, 64
    jmp .qv2_k64
.qv2_k16:
    lea rax, [rcx+16]
    cmp rax, r11
    ja .qv2_reduce
    vmovups zmm1, zmmword ptr [rdx+rcx*4]
    lea r8, [rbp+rcx*8+64]
    vpmovsxbd zmm0, xmmword ptr [r8+0]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+16]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+32]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+48]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+64]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+80]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+96]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+112]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm1
    add rcx, 16
    jmp .qv2_k16
.qv2_reduce:
    vextractf32x8 ymm0, zmm24, 1
    vaddps ymm0, ymm0, ymm24
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm24, xmm0
    vextractf32x8 ymm0, zmm25, 1
    vaddps ymm0, ymm0, ymm25
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm25, xmm0
    vextractf32x8 ymm0, zmm26, 1
    vaddps ymm0, ymm0, ymm26
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm26, xmm0
    vextractf32x8 ymm0, zmm27, 1
    vaddps ymm0, ymm0, ymm27
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm27, xmm0
    vextractf32x8 ymm0, zmm28, 1
    vaddps ymm0, ymm0, ymm28
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm28, xmm0
    vextractf32x8 ymm0, zmm29, 1
    vaddps ymm0, ymm0, ymm29
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm29, xmm0
    vextractf32x8 ymm0, zmm30, 1
    vaddps ymm0, ymm0, ymm30
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm30, xmm0
    vextractf32x8 ymm0, zmm31, 1
    vaddps ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm31, xmm0
.qv2_store:
    vmulss xmm24, xmm24, xmm16
    vmovss dword ptr [r15+0], xmm24
    vmulss xmm25, xmm25, xmm17
    vmovss dword ptr [r15+4], xmm25
    vmulss xmm26, xmm26, xmm18
    vmovss dword ptr [r15+8], xmm26
    vmulss xmm27, xmm27, xmm19
    vmovss dword ptr [r15+12], xmm27
    vmulss xmm28, xmm28, xmm20
    vmovss dword ptr [r15+16], xmm28
    vmulss xmm29, xmm29, xmm21
    vmovss dword ptr [r15+20], xmm29
    vmulss xmm30, xmm30, xmm22
    vmovss dword ptr [r15+24], xmm30
    vmulss xmm31, xmm31, xmm23
    vmovss dword ptr [r15+28], xmm31
    add rbp, r14
    add r15, 32
    inc r12
    cmp r12, r13
    jb .qv2_group
.qv2_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_worker_do_v2_avx2:
    /* Portable v2 fallback: eight independent AVX2 accumulators. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov r10, [rbx+JOB_OUT]
    shr r10, 3
    mov rax, r10
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax
    lea rax, [r15+1]
    mul r10
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax
    cmp r12, r13
    jae .qv2a_done
    mov r11, [rbx+JOB_IN]
    mov r14, r11
    shl r14, 3
    add r14, 64
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax
    mov rax, r12
    shl rax, 5
    add rax, [rbx+JOB_Y]
    mov r15, rax
    mov rdx, [rbx+JOB_X]
.qv2a_group:
    vxorps ymm8, ymm8, ymm8
    vxorps ymm9, ymm9, ymm9
    vxorps ymm10, ymm10, ymm10
    vxorps ymm11, ymm11, ymm11
    vxorps ymm12, ymm12, ymm12
    vxorps ymm13, ymm13, ymm13
    vxorps ymm14, ymm14, ymm14
    vxorps ymm15, ymm15, ymm15
    xor ecx, ecx
.qv2a_k16:
    cmp rcx, r11
    jae .qv2a_reduce
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vmovups ymm2, ymmword ptr [rdx+rcx*4+32]
    lea r8, [rbp+rcx*8+64]
    vbroadcastss ymm3, dword ptr [rbp+0]
    vpmovsxbd ymm0, qword ptr [r8+0]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm8, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+8]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm8, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+4]
    vpmovsxbd ymm0, qword ptr [r8+16]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm9, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+24]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm9, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+8]
    vpmovsxbd ymm0, qword ptr [r8+32]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm10, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+40]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm10, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+12]
    vpmovsxbd ymm0, qword ptr [r8+48]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm11, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+56]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm11, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+16]
    vpmovsxbd ymm0, qword ptr [r8+64]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm12, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+72]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm12, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+20]
    vpmovsxbd ymm0, qword ptr [r8+80]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm13, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+88]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm13, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+24]
    vpmovsxbd ymm0, qword ptr [r8+96]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm14, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+104]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm14, ymm0, ymm2
    vbroadcastss ymm3, dword ptr [rbp+28]
    vpmovsxbd ymm0, qword ptr [r8+112]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm15, ymm0, ymm1
    vpmovsxbd ymm0, qword ptr [r8+120]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vfmadd231ps ymm15, ymm0, ymm2
    add rcx, 16
    jmp .qv2a_k16
.qv2a_reduce:
    vextractf128 xmm0, ymm8, 1
    vaddps xmm0, xmm0, xmm8
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+0], xmm0
    vextractf128 xmm0, ymm9, 1
    vaddps xmm0, xmm0, xmm9
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+4], xmm0
    vextractf128 xmm0, ymm10, 1
    vaddps xmm0, xmm0, xmm10
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+8], xmm0
    vextractf128 xmm0, ymm11, 1
    vaddps xmm0, xmm0, xmm11
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+12], xmm0
    vextractf128 xmm0, ymm12, 1
    vaddps xmm0, xmm0, xmm12
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+16], xmm0
    vextractf128 xmm0, ymm13, 1
    vaddps xmm0, xmm0, xmm13
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+20], xmm0
    vextractf128 xmm0, ymm14, 1
    vaddps xmm0, xmm0, xmm14
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+24], xmm0
    vextractf128 xmm0, ymm15, 1
    vaddps xmm0, xmm0, xmm15
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss dword ptr [r15+28], xmm0
    add rbp, r14
    add r15, 32
    inc r12
    cmp r12, r13
    jb .qv2a_group
.qv2a_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
q8_worker_do_avx512:
    cmp qword ptr [rip+g_workers], 1
    je q8_worker_do_avx512_8row_compact
    cmp qword ptr [rip+g_workers], 6
    jbe q8_worker_do_avx512_8row_unroll4
    cmp qword ptr [rip+g_workers], 7
    je q8_worker_do_avx512_6row
    jmp q8_worker_do_avx512_4row

q8_worker_do_avx512_8row_compact:
    /* Eight output rows in parallel. On the target AVX-512 CPU this halves
       activation-load pressure versus the four-row kernel while preserving
       each row's FP32 accumulation order exactly. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov rax, [rbx+JOB_OUT]
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax                    /* start row */
    lea rax, [r15+1]
    mul qword ptr [rbx+JOB_OUT]
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax                    /* end row */
    cmp r12, r13
    jae .q8c_done

    mov r14, [rbx+JOB_STRIDE]
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax                    /* row0 ptr */
    mov rax, r12
    shl rax, 2
    add rax, [rbx+JOB_Y]
    mov r15, rax                    /* y ptr */
    mov rax, [rbx+JOB_Y]
    lea r12, [rax+r13*4]            /* end y ptr */
    mov rdx, [rbx+JOB_X]
    mov r11, [rbx+JOB_IN]

.q8c_rows8:
    lea rax, [r15+32]
    cmp rax, r12
    ja .q8c_tail_rows

    mov r8, rbp
    add r8, r14                     /* row1 */
    mov r9, r8
    add r9, r14                     /* row2 */
    mov r10, r9
    add r10, r14                    /* row3 */
    mov rsi, r10
    add rsi, r14                    /* row4 */
    mov rdi, rsi
    add rdi, r14                    /* row5 */
    mov rbx, rdi
    add rbx, r14                    /* row6 */
    mov r13, rbx
    add r13, r14                    /* row7 */

    vbroadcastss zmm16, dword ptr [rbp]
    vbroadcastss zmm17, dword ptr [r8]
    vbroadcastss zmm18, dword ptr [r9]
    vbroadcastss zmm19, dword ptr [r10]
    vbroadcastss zmm20, dword ptr [rsi]
    vbroadcastss zmm21, dword ptr [rdi]
    vbroadcastss zmm22, dword ptr [rbx]
    vbroadcastss zmm23, dword ptr [r13]
    vxorps zmm24, zmm24, zmm24
    vxorps zmm25, zmm25, zmm25
    vxorps zmm26, zmm26, zmm26
    vxorps zmm27, zmm27, zmm27
    vxorps zmm28, zmm28, zmm28
    vxorps zmm29, zmm29, zmm29
    vxorps zmm30, zmm30, zmm30
    vxorps zmm31, zmm31, zmm31
    xor ecx, ecx

.q8c_k16:
    lea rax, [rcx+16]
    cmp rax, r11
    ja .q8c_reduce8
    vmovups zmm1, zmmword ptr [rdx+rcx*4]

    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r8+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r9+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r10+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r13+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm1

    add rcx, 16
    jmp .q8c_k16

.q8c_reduce8:
    vextractf32x8 ymm0, zmm24, 1
    vaddps ymm0, ymm0, ymm24
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm24, xmm0

    vextractf32x8 ymm0, zmm25, 1
    vaddps ymm0, ymm0, ymm25
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm25, xmm0

    vextractf32x8 ymm0, zmm26, 1
    vaddps ymm0, ymm0, ymm26
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm26, xmm0

    vextractf32x8 ymm0, zmm27, 1
    vaddps ymm0, ymm0, ymm27
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm27, xmm0

    vextractf32x8 ymm0, zmm28, 1
    vaddps ymm0, ymm0, ymm28
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm28, xmm0

    vextractf32x8 ymm0, zmm29, 1
    vaddps ymm0, ymm0, ymm29
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm29, xmm0

    vextractf32x8 ymm0, zmm30, 1
    vaddps ymm0, ymm0, ymm30
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm30, xmm0

    vextractf32x8 ymm0, zmm31, 1
    vaddps ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm31, xmm0

.q8c_scalar8:
    cmp rcx, r11
    jae .q8c_store8
    vmovss xmm1, dword ptr [rdx+rcx*4]

    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm24, xmm0, xmm1
    movsx eax, byte ptr [r8+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm25, xmm0, xmm1
    movsx eax, byte ptr [r9+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm26, xmm0, xmm1
    movsx eax, byte ptr [r10+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm27, xmm0, xmm1
    movsx eax, byte ptr [rsi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm28, xmm0, xmm1
    movsx eax, byte ptr [rdi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm29, xmm0, xmm1
    movsx eax, byte ptr [rbx+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm30, xmm0, xmm1
    movsx eax, byte ptr [r13+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm31, xmm0, xmm1
    inc rcx
    jmp .q8c_scalar8

.q8c_store8:
    vmulss xmm24, xmm24, xmm16
    vmulss xmm25, xmm25, xmm17
    vmulss xmm26, xmm26, xmm18
    vmulss xmm27, xmm27, xmm19
    vmulss xmm28, xmm28, xmm20
    vmulss xmm29, xmm29, xmm21
    vmulss xmm30, xmm30, xmm22
    vmulss xmm31, xmm31, xmm23
    vmovss dword ptr [r15], xmm24
    vmovss dword ptr [r15+4], xmm25
    vmovss dword ptr [r15+8], xmm26
    vmovss dword ptr [r15+12], xmm27
    vmovss dword ptr [r15+16], xmm28
    vmovss dword ptr [r15+20], xmm29
    vmovss dword ptr [r15+24], xmm30
    vmovss dword ptr [r15+28], xmm31
    lea rbp, [r13+r14]
    add r15, 32
    cmp r15, r12
    jb .q8c_rows8
    jmp .q8c_done

.q8c_tail_rows:
    /* Generic remainder path, at most seven rows. */
.q8c_tail_row:
    cmp r15, r12
    jae .q8c_done
    vxorps ymm2, ymm2, ymm2
    vbroadcastss ymm3, dword ptr [rbp]
    xor ecx, ecx
.q8c_tk8:
    lea rax, [rcx+8]
    cmp rax, r11
    ja .q8c_treduce
    vpmovsxbd ymm0, qword ptr [rbp+rcx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm2, ymm0, ymm1
    add rcx, 8
    jmp .q8c_tk8
.q8c_treduce:
    vextractf128 xmm0, ymm2, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm4, xmm0
.q8c_tscalar:
    cmp rcx, r11
    jae .q8c_tstore
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm3
    vmulss xmm0, xmm0, dword ptr [rdx+rcx*4]
    vaddss xmm4, xmm4, xmm0
    inc rcx
    jmp .q8c_tscalar
.q8c_tstore:
    vmovss dword ptr [r15], xmm4
    add rbp, r14
    add r15, 4
    jmp .q8c_tail_row

.q8c_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_worker_do_avx512_8row_unroll4:
    /* Eight output rows in parallel. On the target AVX-512 CPU this halves
       activation-load pressure versus the four-row kernel while preserving
       each row's FP32 accumulation order exactly. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov rax, [rbx+JOB_OUT]
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax                    /* start row */
    lea rax, [r15+1]
    mul qword ptr [rbx+JOB_OUT]
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax                    /* end row */
    cmp r12, r13
    jae .q8_done

    mov r14, [rbx+JOB_STRIDE]
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax                    /* row0 ptr */
    mov rax, r12
    shl rax, 2
    add rax, [rbx+JOB_Y]
    mov r15, rax                    /* y ptr */
    mov rax, [rbx+JOB_Y]
    lea r12, [rax+r13*4]            /* end y ptr */
    mov rdx, [rbx+JOB_X]
    mov r11, [rbx+JOB_IN]

.q8_rows8:
    lea rax, [r15+32]
    cmp rax, r12
    ja .q8_tail_rows

    mov r8, rbp
    add r8, r14                     /* row1 */
    mov r9, r8
    add r9, r14                     /* row2 */
    mov r10, r9
    add r10, r14                    /* row3 */
    mov rsi, r10
    add rsi, r14                    /* row4 */
    mov rdi, rsi
    add rdi, r14                    /* row5 */
    mov rbx, rdi
    add rbx, r14                    /* row6 */
    mov r13, rbx
    add r13, r14                    /* row7 */

    vbroadcastss zmm16, dword ptr [rbp]
    vbroadcastss zmm17, dword ptr [r8]
    vbroadcastss zmm18, dword ptr [r9]
    vbroadcastss zmm19, dword ptr [r10]
    vbroadcastss zmm20, dword ptr [rsi]
    vbroadcastss zmm21, dword ptr [rdi]
    vbroadcastss zmm22, dword ptr [rbx]
    vbroadcastss zmm23, dword ptr [r13]
    vxorps zmm24, zmm24, zmm24
    vxorps zmm25, zmm25, zmm25
    vxorps zmm26, zmm26, zmm26
    vxorps zmm27, zmm27, zmm27
    vxorps zmm28, zmm28, zmm28
    vxorps zmm29, zmm29, zmm29
    vxorps zmm30, zmm30, zmm30
    vxorps zmm31, zmm31, zmm31
    xor ecx, ecx

.q8_k64:
    lea rax, [rcx+64]
    cmp rax, r11
    ja .q8_k16
    vmovups zmm1, zmmword ptr [rdx+rcx*4+0]
    vmovups zmm2, zmmword ptr [rdx+rcx*4+64]
    vmovups zmm3, zmmword ptr [rdx+rcx*4+128]
    vmovups zmm4, zmmword ptr [rdx+rcx*4+192]
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r13+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [r13+rcx+20]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm2
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [r13+rcx+36]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm3
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm4
    vpmovsxbd zmm0, xmmword ptr [r13+rcx+52]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm4
    add rcx, 64
    jmp .q8_k64
.q8_k16:
    lea rax, [rcx+16]
    cmp rax, r11
    ja .q8_reduce8
    vmovups zmm1, zmmword ptr [rdx+rcx*4]
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm24, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm25, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm26, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm27, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm28, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm29, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rbx+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm30, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r13+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vfmadd231ps zmm31, zmm0, zmm1
    add rcx, 16
    jmp .q8_k16
.q8_reduce8:
    vextractf32x8 ymm0, zmm24, 1
    vaddps ymm0, ymm0, ymm24
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm24, xmm0

    vextractf32x8 ymm0, zmm25, 1
    vaddps ymm0, ymm0, ymm25
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm25, xmm0

    vextractf32x8 ymm0, zmm26, 1
    vaddps ymm0, ymm0, ymm26
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm26, xmm0

    vextractf32x8 ymm0, zmm27, 1
    vaddps ymm0, ymm0, ymm27
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm27, xmm0

    vextractf32x8 ymm0, zmm28, 1
    vaddps ymm0, ymm0, ymm28
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm28, xmm0

    vextractf32x8 ymm0, zmm29, 1
    vaddps ymm0, ymm0, ymm29
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm29, xmm0

    vextractf32x8 ymm0, zmm30, 1
    vaddps ymm0, ymm0, ymm30
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm30, xmm0

    vextractf32x8 ymm0, zmm31, 1
    vaddps ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm31, xmm0

.q8_scalar8:
    cmp rcx, r11
    jae .q8_store8
    vmovss xmm1, dword ptr [rdx+rcx*4]

    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm24, xmm0, xmm1
    movsx eax, byte ptr [r8+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm25, xmm0, xmm1
    movsx eax, byte ptr [r9+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm26, xmm0, xmm1
    movsx eax, byte ptr [r10+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm27, xmm0, xmm1
    movsx eax, byte ptr [rsi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm28, xmm0, xmm1
    movsx eax, byte ptr [rdi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm29, xmm0, xmm1
    movsx eax, byte ptr [rbx+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm30, xmm0, xmm1
    movsx eax, byte ptr [r13+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vfmadd231ss xmm31, xmm0, xmm1
    inc rcx
    jmp .q8_scalar8

.q8_store8:
    vmulss xmm24, xmm24, xmm16
    vmulss xmm25, xmm25, xmm17
    vmulss xmm26, xmm26, xmm18
    vmulss xmm27, xmm27, xmm19
    vmulss xmm28, xmm28, xmm20
    vmulss xmm29, xmm29, xmm21
    vmulss xmm30, xmm30, xmm22
    vmulss xmm31, xmm31, xmm23
    vmovss dword ptr [r15], xmm24
    vmovss dword ptr [r15+4], xmm25
    vmovss dword ptr [r15+8], xmm26
    vmovss dword ptr [r15+12], xmm27
    vmovss dword ptr [r15+16], xmm28
    vmovss dword ptr [r15+20], xmm29
    vmovss dword ptr [r15+24], xmm30
    vmovss dword ptr [r15+28], xmm31
    lea rbp, [r13+r14]
    add r15, 32
    cmp r15, r12
    jb .q8_rows8
    jmp .q8_done

.q8_tail_rows:
    /* Generic remainder path, at most seven rows. */
.q8_tail_row:
    cmp r15, r12
    jae .q8_done
    vxorps ymm2, ymm2, ymm2
    vbroadcastss ymm3, dword ptr [rbp]
    xor ecx, ecx
.q8_tk8:
    lea rax, [rcx+8]
    cmp rax, r11
    ja .q8_treduce
    vpmovsxbd ymm0, qword ptr [rbp+rcx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm2, ymm0, ymm1
    add rcx, 8
    jmp .q8_tk8
.q8_treduce:
    vextractf128 xmm0, ymm2, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm4, xmm0
.q8_tscalar:
    cmp rcx, r11
    jae .q8_tstore
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm3
    vmulss xmm0, xmm0, dword ptr [rdx+rcx*4]
    vaddss xmm4, xmm4, xmm0
    inc rcx
    jmp .q8_tscalar
.q8_tstore:
    vmovss dword ptr [r15], xmm4
    add rbp, r14
    add r15, 4
    jmp .q8_tail_row

.q8_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_worker_do_avx512_6row:
    /* Six-row middle-width kernel for high-core-count decode. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov rax, [rbx+JOB_OUT]
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax
    lea rax, [r15+1]
    mul qword ptr [rbx+JOB_OUT]
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax
    cmp r12, r13
    jae .q6_done
    mov r14, [rbx+JOB_STRIDE]
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax
    mov rax, r12
    shl rax, 2
    add rax, [rbx+JOB_Y]
    mov r15, rax
    mov rdx, [rbx+JOB_X]
    mov r11, [rbx+JOB_IN]
.q6_rows:
    mov rax, r13
    sub rax, r12
    cmp rax, 6
    jb .q6_tail_rows
    mov r8, rbp
    add r8, r14
    mov r9, r8
    add r9, r14
    mov r10, r9
    add r10, r14
    mov rsi, r10
    add rsi, r14
    mov rdi, rsi
    add rdi, r14
    vbroadcastss zmm18, dword ptr [rbp]
    vbroadcastss zmm19, dword ptr [r8]
    vbroadcastss zmm20, dword ptr [r9]
    vbroadcastss zmm21, dword ptr [r10]
    vbroadcastss zmm22, dword ptr [rsi]
    vbroadcastss zmm23, dword ptr [rdi]
    vxorps zmm26, zmm26, zmm26
    vxorps zmm27, zmm27, zmm27
    vxorps zmm28, zmm28, zmm28
    vxorps zmm29, zmm29, zmm29
    vxorps zmm30, zmm30, zmm30
    vxorps zmm31, zmm31, zmm31
    xor ecx, ecx
.q6_k16:
    lea rax, [rcx+16]
    cmp rax, r11
    ja .q6_reduce
    vmovups zmm1, zmmword ptr [rdx+rcx*4]
    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm18
    vfmadd231ps zmm26, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r8+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm19
    vfmadd231ps zmm27, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r9+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm20
    vfmadd231ps zmm28, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [r10+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm21
    vfmadd231ps zmm29, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rsi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm22
    vfmadd231ps zmm30, zmm0, zmm1
    vpmovsxbd zmm0, xmmword ptr [rdi+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm23
    vfmadd231ps zmm31, zmm0, zmm1
    add rcx, 16
    jmp .q6_k16
.q6_reduce:
    vextractf32x8 ymm0, zmm26, 1
    vaddps ymm0, ymm0, ymm26
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm26, xmm0
    vextractf32x8 ymm0, zmm27, 1
    vaddps ymm0, ymm0, ymm27
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm27, xmm0
    vextractf32x8 ymm0, zmm28, 1
    vaddps ymm0, ymm0, ymm28
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm28, xmm0
    vextractf32x8 ymm0, zmm29, 1
    vaddps ymm0, ymm0, ymm29
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm29, xmm0
    vextractf32x8 ymm0, zmm30, 1
    vaddps ymm0, ymm0, ymm30
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm30, xmm0
    vextractf32x8 ymm0, zmm31, 1
    vaddps ymm0, ymm0, ymm31
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm31, xmm0
.q6_scalar:
    cmp rcx, r11
    jae .q6_store
    vmovss xmm1, dword ptr [rdx+rcx*4]
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm18
    vfmadd231ss xmm26, xmm0, xmm1
    movsx eax, byte ptr [r8+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm19
    vfmadd231ss xmm27, xmm0, xmm1
    movsx eax, byte ptr [r9+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm20
    vfmadd231ss xmm28, xmm0, xmm1
    movsx eax, byte ptr [r10+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm21
    vfmadd231ss xmm29, xmm0, xmm1
    movsx eax, byte ptr [rsi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm22
    vfmadd231ss xmm30, xmm0, xmm1
    movsx eax, byte ptr [rdi+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm23
    vfmadd231ss xmm31, xmm0, xmm1
    inc rcx
    jmp .q6_scalar
.q6_store:
    vmovss dword ptr [r15], xmm26
    vmovss dword ptr [r15+4], xmm27
    vmovss dword ptr [r15+8], xmm28
    vmovss dword ptr [r15+12], xmm29
    vmovss dword ptr [r15+16], xmm30
    vmovss dword ptr [r15+20], xmm31
    lea rbp, [rdi+r14]
    add r15, 24
    add r12, 6
    cmp r12, r13
    jb .q6_rows
    jmp .q6_done
.q6_tail_rows:
.q6_tail_row:
    cmp r12, r13
    jae .q6_done
    vxorps ymm2, ymm2, ymm2
    vbroadcastss ymm3, dword ptr [rbp]
    xor ecx, ecx
.q6_tk8:
    lea rax, [rcx+8]
    cmp rax, r11
    ja .q6_treduce
    vpmovsxbd ymm0, qword ptr [rbp+rcx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm2, ymm0, ymm1
    add rcx, 8
    jmp .q6_tk8
.q6_treduce:
    vextractf128 xmm0, ymm2, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm4, xmm0
.q6_tscalar:
    cmp rcx, r11
    jae .q6_tstore
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm3
    vmulss xmm0, xmm0, dword ptr [rdx+rcx*4]
    vaddss xmm4, xmm4, xmm0
    inc rcx
    jmp .q6_tscalar
.q6_tstore:
    vmovss dword ptr [r15], xmm4
    add rbp, r14
    add r15, 4
    inc r12
    jmp .q6_tail_row
.q6_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_worker_do_avx512_4row:
    /* Four output rows in parallel. Reuses each 16-float activation load across
       four rows and keeps four independent accumulation chains in flight. */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov rax, [rbx+JOB_OUT]
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax                    /* start row */
    lea rax, [r15+1]
    mul qword ptr [rbx+JOB_OUT]
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax                    /* end row */
    cmp r12, r13
    jae .q5_done

    mov r14, [rbx+JOB_STRIDE]
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax                    /* row0 ptr */
    mov rax, r12
    shl rax, 2
    add rax, [rbx+JOB_Y]
    mov r15, rax                    /* y ptr */
    mov rdx, [rbx+JOB_X]
    mov r11, [rbx+JOB_IN]

.q5_rows4:
    mov rax, r13
    sub rax, r12
    cmp rax, 4
    jb .q5_tail_rows

    mov r8, rbp
    add r8, r14                     /* row1 */
    mov r9, r8
    add r9, r14                     /* row2 */
    mov r10, r9
    add r10, r14                    /* row3 */

    vbroadcastss zmm20, dword ptr [rbp]
    vbroadcastss zmm21, dword ptr [r8]
    vbroadcastss zmm22, dword ptr [r9]
    vbroadcastss zmm23, dword ptr [r10]
    vxorps zmm24, zmm24, zmm24
    vxorps zmm25, zmm25, zmm25
    vxorps zmm26, zmm26, zmm26
    vxorps zmm27, zmm27, zmm27
    xor ecx, ecx

.q5_k16:
    lea rax, [rcx+16]
    cmp rax, r11
    ja .q5_reduce4
    vmovups zmm1, zmmword ptr [rdx+rcx*4]

    vpmovsxbd zmm0, xmmword ptr [rbp+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm20
    vfmadd231ps zmm24, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r8+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm21
    vfmadd231ps zmm25, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r9+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm22
    vfmadd231ps zmm26, zmm0, zmm1

    vpmovsxbd zmm0, xmmword ptr [r10+rcx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm23
    vfmadd231ps zmm27, zmm0, zmm1

    add rcx, 16
    jmp .q5_k16

.q5_reduce4:
    /* Deterministic horizontal reductions, one scalar result per row. */
    vextractf32x8 ymm0, zmm24, 1
    vaddps ymm0, ymm0, ymm24
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm24, xmm0

    vextractf32x8 ymm0, zmm25, 1
    vaddps ymm0, ymm0, ymm25
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm25, xmm0

    vextractf32x8 ymm0, zmm26, 1
    vaddps ymm0, ymm0, ymm26
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm26, xmm0

    vextractf32x8 ymm0, zmm27, 1
    vaddps ymm0, ymm0, ymm27
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm27, xmm0

.q5_tail4:
    cmp rcx, r11
    jae .q5_store4
    vmovss xmm1, dword ptr [rdx+rcx*4]

    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm20
    vfmadd231ss xmm24, xmm0, xmm1

    movsx eax, byte ptr [r8+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm21
    vfmadd231ss xmm25, xmm0, xmm1

    movsx eax, byte ptr [r9+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm22
    vfmadd231ss xmm26, xmm0, xmm1

    movsx eax, byte ptr [r10+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm23
    vfmadd231ss xmm27, xmm0, xmm1

    inc rcx
    jmp .q5_tail4

.q5_store4:
    vmovss dword ptr [r15], xmm24
    vmovss dword ptr [r15+4], xmm25
    vmovss dword ptr [r15+8], xmm26
    vmovss dword ptr [r15+12], xmm27
    lea rbp, [r10+r14]
    add r15, 16
    add r12, 4
    cmp r12, r13
    jb .q5_rows4
    jmp .q5_done

.q5_tail_rows:
    /* Rare 1..3-row worker remainder; keep AVX2 summation path for simplicity. */
.q5_tail_row:
    cmp r12, r13
    jae .q5_done
    vxorps ymm2, ymm2, ymm2
    vbroadcastss ymm3, dword ptr [rbp]
    xor ecx, ecx
.q5_tk8:
    lea rax, [rcx+8]
    cmp rax, r11
    ja .q5_treduce
    vpmovsxbd ymm0, qword ptr [rbp+rcx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm2, ymm0, ymm1
    add rcx, 8
    jmp .q5_tk8
.q5_treduce:
    vextractf128 xmm0, ymm2, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm4, xmm0
.q5_tscalar:
    cmp rcx, r11
    jae .q5_tstore
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm3
    vmulss xmm0, xmm0, dword ptr [rdx+rcx*4]
    vaddss xmm4, xmm4, xmm0
    inc rcx
    jmp .q5_tscalar
.q5_tstore:
    vmovss dword ptr [r15], xmm4
    add rbp, r14
    add r15, 4
    inc r12
    jmp .q5_tail_row

.q5_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_worker_do_avx2:
    /* rdi=worker ID; reads shared job */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r15, rdi
    mov rbx, [rip+g_job]
    mov rax, [rbx+JOB_OUT]
    mul r15
    div qword ptr [rbx+JOB_NWORK]
    mov r12, rax                    /* start row */
    lea rax, [r15+1]
    mul qword ptr [rbx+JOB_OUT]
    div qword ptr [rbx+JOB_NWORK]
    mov r13, rax                    /* end row */
    cmp r12, r13
    jae .qwd_done

    mov r14, [rbx+JOB_STRIDE]
    mov rax, r12
    imul rax, r14
    add rax, [rbx+JOB_W]
    mov rbp, rax                    /* row ptr */
    mov rax, r12
    shl rax,2
    add rax, [rbx+JOB_Y]
    mov r15, rax                    /* y ptr */
.qwd_row:
    vxorps ymm2, ymm2, ymm2
    vbroadcastss ymm3, dword ptr [rbp]
    xor ecx, ecx
    mov r11, [rbx+JOB_IN]
.qwd_k8:
    lea rax, [rcx+8]
    cmp rax, r11
    ja .qwd_tail
    vpmovsxbd ymm0, qword ptr [rbp+rcx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm3
    mov rdx, [rbx+JOB_X]
    vmovups ymm1, ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm2, ymm0, ymm1
    add rcx,8
    jmp .qwd_k8
.qwd_tail:
    vextractf128 xmm0, ymm2, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovaps xmm4, xmm0
.qwd_tail_loop:
    cmp rcx, r11
    jae .qwd_store
    movsx eax, byte ptr [rbp+rcx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm3
    mov rdx, [rbx+JOB_X]
    vmulss xmm0, xmm0, dword ptr [rdx+rcx*4]
    vaddss xmm4, xmm4, xmm0
    inc rcx
    jmp .qwd_tail_loop
.qwd_store:
    vmovss dword ptr [r15], xmm4
    add rbp, r14
    add r15,4
    inc r12
    cmp r12, r13
    jb .qwd_row
.qwd_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

q8_dequant_row:
    cmp dword ptr [rip+g_use_avx512], 0
    je q8_dequant_row_avx2
    push rbx
    mov rax, rsi
    imul rax, r8
    add rdi, rax
    vbroadcastss zmm2, dword ptr [rdi]
    xor ebx, ebx
.qdr5_loop:
    lea rax, [rbx+16]
    cmp rax, rcx
    ja .qdr5_tail
    vpmovsxbd zmm0, xmmword ptr [rdi+rbx+4]
    vcvtdq2ps zmm0, zmm0
    vmulps zmm0, zmm0, zmm2
    vmovups zmmword ptr [rdx+rbx*4], zmm0
    add rbx, 16
    jmp .qdr5_loop
.qdr5_tail:
    cmp rbx, rcx
    jae .qdr5_done
.qdr5_tloop:
    movsx eax, byte ptr [rdi+rbx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0, xmm0, xmm2
    vmovss dword ptr [rdx+rbx*4], xmm0
    inc rbx
    cmp rbx, rcx
    jb .qdr5_tloop
.qdr5_done:
    vzeroupper
    pop rbx
    ret

q8_dequant_row_avx2:
    /* rdi=matrix, rsi=row, rdx=out, rcx=dim, r8=stride */
    push rbx
    mov rax, rsi
    imul rax, r8
    add rdi, rax
    vbroadcastss ymm2, dword ptr [rdi]
    xor ebx, ebx
.qdr_loop:
    lea rax, [rbx+8]
    cmp rax, rcx
    ja .qdr_tail
    vpmovsxbd ymm0, qword ptr [rdi+rbx+4]
    vcvtdq2ps ymm0, ymm0
    vmulps ymm0, ymm0, ymm2
    vmovups ymmword ptr [rdx+rbx*4], ymm0
    add rbx,8
    jmp .qdr_loop
.qdr_tail:
    cmp rbx, rcx
    jae .qdr_done
.qdr_tloop:
    movsx eax, byte ptr [rdi+rbx+4]
    vcvtsi2ss xmm0, xmm0, eax
    vmulss xmm0,xmm0,xmm2
    vmovss dword ptr [rdx+rbx*4],xmm0
    inc rbx
    cmp rbx,rcx
    jb .qdr_tloop
.qdr_done:
    vzeroupper
    pop rbx
    ret

/* -------------------------------------------------------------------------- */
/* Math kernels                                                               */
/* -------------------------------------------------------------------------- */

rmsnorm:
    /* rdi=out rsi=x rdx=weight rcx=n */
    push rbx
    vxorps ymm0,ymm0,ymm0
    xor ebx,ebx
.rn_sum:
    lea rax,[rbx+8]
    cmp rax,rcx
    ja .rn_sum_tail
    vmovups ymm1,ymmword ptr [rsi+rbx*4]
    vfmadd231ps ymm0,ymm1,ymm1
    add rbx,8
    jmp .rn_sum
.rn_sum_tail:
    vextractf128 xmm1,ymm0,1
    vaddps xmm0,xmm0,xmm1
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovaps xmm4,xmm0
.rn_st_loop:
    cmp rbx,rcx
    jae .rn_scale
    vmovss xmm1,dword ptr [rsi+rbx*4]
    vfmadd231ss xmm4,xmm1,xmm1
    inc rbx
    jmp .rn_st_loop
.rn_scale:
    vcvtsi2ss xmm1,xmm1,rcx
    vdivss xmm4,xmm4,xmm1
    vaddss xmm4,xmm4,dword ptr [rip+g_rms_norm_eps]
    vsqrtss xmm4,xmm4,xmm4
    vmovss xmm5,dword ptr [rip+f_one]
    vdivss xmm5,xmm5,xmm4
    cmp dword ptr [rip+g_use_avx512], 0
    je .rn_apply_avx2
    vbroadcastss zmm5,xmm5
    xor ebx,ebx
.rn_apply_avx512:
    lea rax,[rbx+16]
    cmp rax,rcx
    ja .rn_apply_tail
    vmovups zmm0,zmmword ptr [rsi+rbx*4]
    vmulps zmm0,zmm0,zmm5
    vmulps zmm0,zmm0,zmmword ptr [rdx+rbx*4]
    vmovups zmmword ptr [rdi+rbx*4],zmm0
    add rbx,16
    jmp .rn_apply_avx512
.rn_apply_avx2:
    vbroadcastss ymm5,xmm5
    xor ebx,ebx
.rn_apply:
    lea rax,[rbx+8]
    cmp rax,rcx
    ja .rn_apply_tail
    vmovups ymm0,ymmword ptr [rsi+rbx*4]
    vmulps ymm0,ymm0,ymm5
    vmulps ymm0,ymm0,ymmword ptr [rdx+rbx*4]
    vmovups ymmword ptr [rdi+rbx*4],ymm0
    add rbx,8
    jmp .rn_apply
.rn_apply_tail:
    cmp rbx,rcx
    jae .rn_done
.rn_at_loop:
    vmovss xmm0,dword ptr [rsi+rbx*4]
    vmulss xmm0,xmm0,xmm5
    vmulss xmm0,xmm0,dword ptr [rdx+rbx*4]
    vmovss dword ptr [rdi+rbx*4],xmm0
    inc rbx
    cmp rbx,rcx
    jb .rn_at_loop
.rn_done:
    vzeroupper
    pop rbx
    ret

vec_add_inplace:
    cmp dword ptr [rip+g_use_avx512], 0
    je vec_add_inplace_avx2
    xor ecx, ecx
.vai5_loop:
    lea rax, [rcx+16]
    cmp rax, rdx
    ja .vai5_tail
    vmovups zmm0, zmmword ptr [rdi+rcx*4]
    vaddps zmm0, zmm0, zmmword ptr [rsi+rcx*4]
    vmovups zmmword ptr [rdi+rcx*4], zmm0
    add rcx, 16
    jmp .vai5_loop
.vai5_tail:
    cmp rcx, rdx
    jae .vai5_done
    vmovss xmm0, dword ptr [rdi+rcx*4]
    vaddss xmm0, xmm0, dword ptr [rsi+rcx*4]
    vmovss dword ptr [rdi+rcx*4], xmm0
    inc rcx
    jmp .vai5_tail
.vai5_done:
    vzeroupper
    ret

vec_add_inplace_avx2:
    /* rdi dst, rsi src, rdx n */
    xor ecx,ecx
.vai_loop:
    lea rax,[rcx+8]
    cmp rax,rdx
    ja .vai_tail
    vmovups ymm0,ymmword ptr [rdi+rcx*4]
    vaddps ymm0,ymm0,ymmword ptr [rsi+rcx*4]
    vmovups ymmword ptr [rdi+rcx*4],ymm0
    add rcx,8
    jmp .vai_loop
.vai_tail:
    cmp rcx,rdx
    jae .vai_done
    vmovss xmm0,dword ptr [rdi+rcx*4]
    vaddss xmm0,xmm0,dword ptr [rsi+rcx*4]
    vmovss dword ptr [rdi+rcx*4],xmm0
    inc rcx
    jmp .vai_tail
.vai_done:
    vzeroupper
    ret

vec_dot:
    /* rdi a, rsi b, rdx n -> xmm0 */
    vxorps ymm0,ymm0,ymm0
    xor ecx,ecx
.vd_loop:
    lea rax,[rcx+8]
    cmp rax,rdx
    ja .vd_tail
    vmovups ymm1,ymmword ptr [rdi+rcx*4]
    vfmadd231ps ymm0,ymm1,ymmword ptr [rsi+rcx*4]
    add rcx,8
    jmp .vd_loop
.vd_tail:
    vextractf128 xmm1,ymm0,1
    vaddps xmm0,xmm0,xmm1
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
.vd_tloop:
    cmp rcx,rdx
    jae .vd_done
    vmovss xmm1,dword ptr [rdi+rcx*4]
    vmulss xmm1,xmm1,dword ptr [rsi+rcx*4]
    vaddss xmm0,xmm0,xmm1
    inc rcx
    jmp .vd_tloop
.vd_done:
    vzeroupper
    ret

exp_approx:
    /* xmm0=x; returns approximate exp(x) in xmm0. destroys xmm1..xmm7, eax */
    vmaxss xmm0,xmm0,dword ptr [rip+f_minexp]
    vminss xmm0,xmm0,dword ptr [rip+f_maxexp]
    vmulss xmm1,xmm0,dword ptr [rip+f_log2e]  /* y */
    vroundss xmm2,xmm1,xmm1,1                /* floor(y) */
    vsubss xmm3,xmm1,xmm2                    /* f */
    /* polynomial 2^f */
    vmovss xmm4,dword ptr [rip+f_ln2_5]
    vmulss xmm4,xmm4,xmm3
    vaddss xmm4,xmm4,dword ptr [rip+f_ln2_4]
    vmulss xmm4,xmm4,xmm3
    vaddss xmm4,xmm4,dword ptr [rip+f_ln2_3]
    vmulss xmm4,xmm4,xmm3
    vaddss xmm4,xmm4,dword ptr [rip+f_ln2_2]
    vmulss xmm4,xmm4,xmm3
    vaddss xmm4,xmm4,dword ptr [rip+f_ln2]
    vmulss xmm4,xmm4,xmm3
    vaddss xmm4,xmm4,dword ptr [rip+f_one]
    vcvtss2si eax,xmm2
    add eax,127
    cmp eax,1
    jge .ea_lowok
    mov eax,1
.ea_lowok:
    cmp eax,254
    jle .ea_highok
    mov eax,254
.ea_highok:
    shl eax,23
    vmovd xmm5,eax
    vmulss xmm0,xmm4,xmm5
    ret

silu_mul:
    cmp dword ptr [rip+g_use_avx512], 0
    jne silu_mul_avx512
    jmp silu_mul_scalar

silu_mul_avx512:
    /* rdi=a(gate, overwritten), rsi=b, rdx=n. Vector form is lane-wise
       identical to exp_approx + scalar SiLU, just 16 elements at once. */
    push rbx
    vbroadcastss zmm8,  dword ptr [rip+f_minexp]
    vbroadcastss zmm9,  dword ptr [rip+f_maxexp]
    vbroadcastss zmm10, dword ptr [rip+f_log2e]
    vbroadcastss zmm11, dword ptr [rip+f_ln2_5]
    vbroadcastss zmm12, dword ptr [rip+f_ln2_4]
    vbroadcastss zmm13, dword ptr [rip+f_ln2_3]
    vbroadcastss zmm14, dword ptr [rip+f_ln2_2]
    vbroadcastss zmm15, dword ptr [rip+f_ln2]
    vbroadcastss zmm16, dword ptr [rip+f_one]
    mov eax, 127
    vpbroadcastd zmm17, eax
    mov eax, 1
    vpbroadcastd zmm18, eax
    mov eax, 254
    vpbroadcastd zmm19, eax
    xor ebx, ebx
.smv_loop:
    lea rax, [rbx+16]
    cmp rax, rdx
    ja .smv_tail
    vmovups zmm6, zmmword ptr [rdi+rbx*4]   /* x */
    vxorps zmm0, zmm0, zmm0
    vsubps zmm0, zmm0, zmm6                 /* -x */
    vmaxps zmm0, zmm0, zmm8
    vminps zmm0, zmm0, zmm9
    vmulps zmm1, zmm0, zmm10                /* y */
    vrndscaleps zmm2, zmm1, 1               /* floor(y) */
    vsubps zmm3, zmm1, zmm2                 /* f */
    vmulps zmm4, zmm11, zmm3
    vaddps zmm4, zmm4, zmm12
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm13
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm14
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm15
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm16
    vcvtps2dq zmm5, zmm2
    vpaddd zmm5, zmm5, zmm17
    vpmaxsd zmm5, zmm5, zmm18
    vpminsd zmm5, zmm5, zmm19
    vpslld zmm5, zmm5, 23
    vmulps zmm0, zmm4, zmm5
    vaddps zmm0, zmm0, zmm16
    vdivps zmm6, zmm6, zmm0
    vmulps zmm6, zmm6, zmmword ptr [rsi+rbx*4]
    vmovups zmmword ptr [rdi+rbx*4], zmm6
    add rbx, 16
    jmp .smv_loop
.smv_tail:
    cmp rbx, rdx
    jae .smv_done
.smv_tail_loop:
    vmovss xmm6, dword ptr [rdi+rbx*4]
    vxorps xmm0, xmm0, xmm0
    vsubss xmm0, xmm0, xmm6
    call exp_approx
    vaddss xmm0, xmm0, dword ptr [rip+f_one]
    vdivss xmm6, xmm6, xmm0
    vmulss xmm6, xmm6, dword ptr [rsi+rbx*4]
    vmovss dword ptr [rdi+rbx*4], xmm6
    inc rbx
    cmp rbx, rdx
    jb .smv_tail_loop
.smv_done:
    vzeroupper
    pop rbx
    ret

silu_mul_scalar:
    /* rdi=a(gate, overwritten), rsi=b, rdx=n */
    push rbx
    xor ebx,ebx
.sm_loop:
    cmp rbx,rdx
    jae .sm_done
    vmovss xmm6,dword ptr [rdi+rbx*4]   /* x */
    vxorps xmm0,xmm0,xmm0
    vsubss xmm0,xmm0,xmm6
    call exp_approx
    vaddss xmm0,xmm0,dword ptr [rip+f_one]
    vdivss xmm6,xmm6,xmm0
    vmulss xmm6,xmm6,dword ptr [rsi+rbx*4]
    vmovss dword ptr [rdi+rbx*4],xmm6
    inc rbx
    jmp .sm_loop
.sm_done:
    pop rbx
    ret

softmax:
    cmp dword ptr [rip+g_use_avx512], 0
    jne softmax_avx512
    jmp softmax_scalar

softmax_avx512:
    /* Preserve scalar max and scalar summation order; vectorize only the
       expensive lane-independent exp polynomial and normalization. */
    push rbx
    push r12
    push r13
    test rsi, rsi
    jz .sf5_done
    mov r13, rsi
    vmovss xmm7, dword ptr [rdi]
    mov rbx, 1
.sf5_max:
    cmp rbx, r13
    jae .sf5_exp_setup
    vmaxss xmm7, xmm7, dword ptr [rdi+rbx*4]
    inc rbx
    jmp .sf5_max
.sf5_exp_setup:
    vbroadcastss zmm20, xmm7
    vbroadcastss zmm8,  dword ptr [rip+f_minexp]
    vbroadcastss zmm9,  dword ptr [rip+f_maxexp]
    vbroadcastss zmm10, dword ptr [rip+f_log2e]
    vbroadcastss zmm11, dword ptr [rip+f_ln2_5]
    vbroadcastss zmm12, dword ptr [rip+f_ln2_4]
    vbroadcastss zmm13, dword ptr [rip+f_ln2_3]
    vbroadcastss zmm14, dword ptr [rip+f_ln2_2]
    vbroadcastss zmm15, dword ptr [rip+f_ln2]
    vbroadcastss zmm16, dword ptr [rip+f_one]
    mov eax, 127
    vpbroadcastd zmm17, eax
    mov eax, 1
    vpbroadcastd zmm18, eax
    mov eax, 254
    vpbroadcastd zmm19, eax
    xor r12d, r12d
.sf5_exp_vec:
    lea rax, [r12+16]
    cmp rax, r13
    ja .sf5_exp_tail
    vmovups zmm0, zmmword ptr [rdi+r12*4]
    vsubps zmm0, zmm0, zmm20
    vmaxps zmm0, zmm0, zmm8
    vminps zmm0, zmm0, zmm9
    vmulps zmm1, zmm0, zmm10
    vrndscaleps zmm2, zmm1, 1
    vsubps zmm3, zmm1, zmm2
    vmulps zmm4, zmm11, zmm3
    vaddps zmm4, zmm4, zmm12
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm13
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm14
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm15
    vmulps zmm4, zmm4, zmm3
    vaddps zmm4, zmm4, zmm16
    vcvtps2dq zmm5, zmm2
    vpaddd zmm5, zmm5, zmm17
    vpmaxsd zmm5, zmm5, zmm18
    vpminsd zmm5, zmm5, zmm19
    vpslld zmm5, zmm5, 23
    vmulps zmm0, zmm4, zmm5
    vmovups zmmword ptr [rdi+r12*4], zmm0
    add r12, 16
    jmp .sf5_exp_vec
.sf5_exp_tail:
    cmp r12, r13
    jae .sf5_sum
.sf5_exp_tail_loop:
    vmovss xmm0, dword ptr [rdi+r12*4]
    vsubss xmm0, xmm0, xmm7
    call exp_approx
    vmovss dword ptr [rdi+r12*4], xmm0
    inc r12
    cmp r12, r13
    jb .sf5_exp_tail_loop
.sf5_sum:
    /* Same element order as the original scalar loop => same FP32 sum. */
    xor r12d, r12d
    vxorps xmm6, xmm6, xmm6
.sf5_sum_loop:
    cmp r12, r13
    jae .sf5_norm_setup
    vaddss xmm6, xmm6, dword ptr [rdi+r12*4]
    inc r12
    jmp .sf5_sum_loop
.sf5_norm_setup:
    vmovss xmm5, dword ptr [rip+f_one]
    vdivss xmm5, xmm5, xmm6
    vbroadcastss zmm5, xmm5
    xor ebx, ebx
.sf5_norm_vec:
    lea rax, [rbx+16]
    cmp rax, r13
    ja .sf5_norm_tail
    vmovups zmm0, zmmword ptr [rdi+rbx*4]
    vmulps zmm0, zmm0, zmm5
    vmovups zmmword ptr [rdi+rbx*4], zmm0
    add rbx, 16
    jmp .sf5_norm_vec
.sf5_norm_tail:
    cmp rbx, r13
    jae .sf5_done
.sf5_norm_tail_loop:
    vmovss xmm0, dword ptr [rdi+rbx*4]
    vmulss xmm0, xmm0, xmm5
    vmovss dword ptr [rdi+rbx*4], xmm0
    inc rbx
    cmp rbx, r13
    jb .sf5_norm_tail_loop
.sf5_done:
    vzeroupper
    pop r13
    pop r12
    pop rbx
    ret

softmax_scalar:
    /* rdi=array, rsi=n */
    push rbx
    push r12
    test rsi,rsi
    jz .sf_done
    movss xmm7,dword ptr [rdi]
    mov rbx,1
.sf_max:
    cmp rbx,rsi
    jae .sf_exp
    maxss xmm7,dword ptr [rdi+rbx*4]
    inc rbx
    jmp .sf_max
.sf_exp:
    xor r12d,r12d
    vxorps xmm6,xmm6,xmm6
.sf_el:
    cmp r12,rsi
    jae .sf_norm
    movss xmm0,dword ptr [rdi+r12*4]
    subss xmm0,xmm7
    call exp_approx
    movss dword ptr [rdi+r12*4],xmm0
    addss xmm6,xmm0
    inc r12
    jmp .sf_el
.sf_norm:
    movss xmm5,dword ptr [rip+f_one]
    divss xmm5,xmm6
    xor ebx,ebx
.sf_nl:
    cmp rbx,rsi
    jae .sf_done
    movss xmm0,dword ptr [rdi+rbx*4]
    mulss xmm0,xmm5
    movss dword ptr [rdi+rbx*4],xmm0
    inc rbx
    jmp .sf_nl
.sf_done:
    pop r12
    pop rbx
    ret

apply_rope:
    /* rdi=q, rsi=k, rdx=pos */
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov r12,rdx
    mov r13,[rip+g_head_dim]
    mov r14,r13
    shr r14,1
    /* cos/sin row pointer */
    mov rax,r12
    imul rax,r14
    shl rax,2
    mov rbx,[rip+g_rope_cos]
    add rbx,rax
    mov r15,[rip+g_rope_sin]
    add r15,rax

    /* q heads */
    xor r8d,r8d
.ar_qh:
    cmp r8,[rip+g_heads]
    jae .ar_kstart
    mov rax,r8
    imul rax,r13
    lea r9,[rdi+rax*4]
    xor ecx,ecx
.ar_qi:
    cmp rcx,r14
    jae .ar_qnext
    movss xmm0,dword ptr [r9+rcx*4]
    mov rax,rcx
    add rax,r14
    movss xmm1,dword ptr [r9+rax*4]
    movss xmm2,dword ptr [rbx+rcx*4]
    movss xmm3,dword ptr [r15+rcx*4]
    movaps xmm4,xmm0
    mulss xmm4,xmm2
    movaps xmm5,xmm1
    mulss xmm5,xmm3
    subss xmm4,xmm5
    mulss xmm0,xmm3
    mulss xmm1,xmm2
    addss xmm0,xmm1
    movss dword ptr [r9+rcx*4],xmm4
    movss dword ptr [r9+rax*4],xmm0
    inc rcx
    jmp .ar_qi
.ar_qnext:
    inc r8
    jmp .ar_qh

.ar_kstart:
    xor r8d,r8d
.ar_kh:
    cmp r8,[rip+g_kv_heads]
    jae .ar_done
    mov rax,r8
    imul rax,r13
    lea r9,[rsi+rax*4]
    xor ecx,ecx
.ar_ki:
    cmp rcx,r14
    jae .ar_knext
    movss xmm0,dword ptr [r9+rcx*4]
    mov rax,rcx
    add rax,r14
    movss xmm1,dword ptr [r9+rax*4]
    movss xmm2,dword ptr [rbx+rcx*4]
    movss xmm3,dword ptr [r15+rcx*4]
    movaps xmm4,xmm0
    mulss xmm4,xmm2
    movaps xmm5,xmm1
    mulss xmm5,xmm3
    subss xmm4,xmm5
    mulss xmm0,xmm3
    mulss xmm1,xmm2
    addss xmm0,xmm1
    movss dword ptr [r9+rcx*4],xmm4
    movss dword ptr [r9+rax*4],xmm0
    inc rcx
    jmp .ar_ki
.ar_knext:
    inc r8
    jmp .ar_kh
.ar_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

/* -------------------------------------------------------------------------- */
/* Transformer inference                                                      */
/* -------------------------------------------------------------------------- */

attention_layer:
    /* rdi=layer index, rsi=position. Uses g_q/k/v -> g_att */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rdi
    mov r13,rsi
    mov r14,[rip+g_head_dim]
    mov r15,[rip+g_kv_dim]

    /* cache current k/v */
    mov rax,r12
    imul rax,[rip+g_max_seq]
    add rax,r13
    imul rax,r15
    shl rax,2
    mov rbp,[rip+g_kcache]
    add rbp,rax
    mov rdi,rbp
    mov rsi,[rip+g_k]
    mov rdx,r15
    shl rdx,2
    call memcpy_bytes
    mov rbp,[rip+g_vcache]
    add rbp,rax
    mov rdi,rbp
    mov rsi,[rip+g_v]
    mov rdx,r15
    shl rdx,2
    call memcpy_bytes

    /* zero output attention vector */
    mov rdi,[rip+g_att]
    mov rdx,[rip+g_q_dim]
    shl rdx,2
    call memzero_bytes

    xor ebx,ebx                         /* head */
.al_head:
    cmp rbx,[rip+g_heads]
    jae .al_done
    /* q pointer for head */
    mov rax,rbx
    imul rax,r14
    mov r10,[rip+g_q]
    lea r10,[r10+rax*4]
    /* kv head = head / (heads/kv_heads) */
    mov rax,[rip+g_heads]
    xor edx,edx
    div qword ptr [rip+g_kv_heads]
    mov rcx,rax                          /* group size */
    mov rax,rbx
    xor edx,edx
    div rcx
    mov r11,rax                          /* kv head */

    xor ebp,ebp                          /* t */
.al_score_loop:
    cmp rbp,r13
    ja .al_scores_done
    /* Four independent time positions hide FMA latency and reuse each Q load.
       Each score retains the exact old per-chunk accumulation order. */
    test r14, 7
    jnz .al_score_single
    lea rax, [rbp+3]
    cmp rax, r13
    ja .al_score_single

    /* K pointer at time t for this layer/KV head. */
    mov rax,r12
    imul rax,[rip+g_max_seq]
    add rax,rbp
    imul rax,r15
    mov rcx,r11
    imul rcx,r14
    add rax,rcx
    shl rax,2
    mov rsi,[rip+g_kcache]
    add rsi,rax                       /* k0 */
    mov r8,r15
    shl r8,2                         /* bytes per cached time row */
    lea rdi,[rsi+r8]                 /* k1 */
    lea rdx,[rdi+r8]                 /* k2 */
    lea r9,[rdx+r8]                  /* k3 */

    vxorps ymm2,ymm2,ymm2
    vxorps ymm3,ymm3,ymm3
    vxorps ymm4,ymm4,ymm4
    vxorps ymm5,ymm5,ymm5
    xor ecx,ecx
.al_score4_dim:
    cmp rcx,r14
    jae .al_score4_reduce
    vmovups ymm0,ymmword ptr [r10+rcx*4]
    vfmadd231ps ymm2,ymm0,ymmword ptr [rsi+rcx*4]
    vfmadd231ps ymm3,ymm0,ymmword ptr [rdi+rcx*4]
    vfmadd231ps ymm4,ymm0,ymmword ptr [rdx+rcx*4]
    vfmadd231ps ymm5,ymm0,ymmword ptr [r9+rcx*4]
    add rcx,8
    jmp .al_score4_dim
.al_score4_reduce:
    vextractf128 xmm0,ymm2,1
    vaddps xmm0,xmm0,xmm2
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovaps xmm2,xmm0
    vextractf128 xmm0,ymm3,1
    vaddps xmm0,xmm0,xmm3
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovaps xmm3,xmm0
    vextractf128 xmm0,ymm4,1
    vaddps xmm0,xmm0,xmm4
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovaps xmm4,xmm0
    vextractf128 xmm0,ymm5,1
    vaddps xmm0,xmm0,xmm5
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
    vmovaps xmm5,xmm0
    vmulss xmm2,xmm2,dword ptr [rip+g_att_scale]
    vmulss xmm3,xmm3,dword ptr [rip+g_att_scale]
    vmulss xmm4,xmm4,dword ptr [rip+g_att_scale]
    vmulss xmm5,xmm5,dword ptr [rip+g_att_scale]
    mov rax,[rip+g_scores]
    vmovss dword ptr [rax+rbp*4],xmm2
    vmovss dword ptr [rax+rbp*4+4],xmm3
    vmovss dword ptr [rax+rbp*4+8],xmm4
    vmovss dword ptr [rax+rbp*4+12],xmm5
    add rbp,4
    jmp .al_score_loop

.al_score_single:
    mov rax,r12
    imul rax,[rip+g_max_seq]
    add rax,rbp
    imul rax,r15
    mov rcx,r11
    imul rcx,r14
    add rax,rcx
    shl rax,2
    mov rsi,[rip+g_kcache]
    add rsi,rax
    mov rdi,r10
    mov rdx,r14
    call vec_dot
    vmulss xmm0,xmm0,dword ptr [rip+g_att_scale]
    mov rax,[rip+g_scores]
    vmovss dword ptr [rax+rbp*4],xmm0
    inc rbp
    jmp .al_score_loop
.al_scores_done:
    mov rdi,[rip+g_scores]
    lea rsi,[r13+1]
    call softmax

    /* weighted sum into head output */
    mov rax,rbx
    imul rax,r14
    mov r9,[rip+g_att]
    lea r9,[r9+rax*4]
    mov rdi,r9
    mov rdx,r14
    shl rdx,2
    call memzero_bytes
    xor ebp,ebp
.al_v_time:
    cmp rbp,r13
    ja .al_head_done
    mov rax,r12
    imul rax,[rip+g_max_seq]
    add rax,rbp
    imul rax,r15
    mov rcx,r11
    imul rcx,r14
    add rax,rcx
    shl rax,2
    mov rsi,[rip+g_vcache]
    add rsi,rax
    mov rdx,[rip+g_scores]
    cmp dword ptr [rip+g_use_avx512], 0
    je .al_v_score_avx2
    vbroadcastss zmm7,dword ptr [rdx+rbp*4]
    xor ecx,ecx
.al_v_dim512:
    lea rax,[rcx+16]
    cmp rax,r14
    ja .al_v_tail
    vmovups zmm0,zmmword ptr [r9+rcx*4]
    vmovups zmm1,zmmword ptr [rsi+rcx*4]
    vfmadd231ps zmm0,zmm1,zmm7
    vmovups zmmword ptr [r9+rcx*4],zmm0
    add rcx,16
    jmp .al_v_dim512
.al_v_score_avx2:
    vbroadcastss ymm7,dword ptr [rdx+rbp*4]
    xor ecx,ecx
.al_v_dim:
    lea rax,[rcx+8]
    cmp rax,r14
    ja .al_v_tail
    vmovups ymm0,ymmword ptr [r9+rcx*4]
    vmovups ymm1,ymmword ptr [rsi+rcx*4]
    vfmadd231ps ymm0,ymm1,ymm7
    vmovups ymmword ptr [r9+rcx*4],ymm0
    add rcx,8
    jmp .al_v_dim
.al_v_tail:
    cmp rcx,r14
    jae .al_vdone
    movss xmm0,dword ptr [rsi+rcx*4]
    mulss xmm0,xmm7
    addss xmm0,dword ptr [r9+rcx*4]
    movss dword ptr [r9+rcx*4],xmm0
    inc rcx
    jmp .al_v_tail
.al_vdone:
    inc rbp
    jmp .al_v_time
.al_head_done:
    inc rbx
    jmp .al_head
.al_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

forward_token:
    /* rdi=token id, rsi=position; returns rax=argmax next token */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rdi
    mov r13,rsi

    /* embedding row -> x */
    mov rdi,[rip+g_embeddings]
    mov rsi,r12
    mov rdx,[rip+g_x]
    mov rcx,[rip+g_dim]
    mov r8,[rip+g_stride_dim]
    call q8_dequant_row

    xor r14d,r14d
    lea r15,[rip+layer_descs]
.ft_layer:
    cmp r14,[rip+g_layers_n]
    jae .ft_final

    /* attention rmsnorm */
    mov rdi,[rip+g_xb]
    mov rsi,[rip+g_x]
    mov rdx,[r15+LD_ATTN_NORM]
    mov rcx,[rip+g_dim]
    call rmsnorm

    /* q = wq*xb */
    mov rdi,[r15+LD_WQ]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_q]
    mov rcx,[rip+g_q_dim]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec
    /* k */
    mov rdi,[r15+LD_WK]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_k]
    mov rcx,[rip+g_kv_dim]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec
    /* v */
    mov rdi,[r15+LD_WV]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_v]
    mov rcx,[rip+g_kv_dim]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec

    mov rdi,[rip+g_q]
    mov rsi,[rip+g_k]
    mov rdx,r13
    call apply_rope

    mov rdi,r14
    mov rsi,r13
    call attention_layer

    /* wo(att) -> xb2, residual */
    mov rdi,[r15+LD_WO]
    mov rsi,[rip+g_att]
    mov rdx,[rip+g_xb2]
    mov rcx,[rip+g_dim]
    mov r8,[rip+g_q_dim]
    mov r9,[rip+g_stride_q]
    call q8_matvec
    mov rdi,[rip+g_x]
    mov rsi,[rip+g_xb2]
    mov rdx,[rip+g_dim]
    call vec_add_inplace

    /* FFN norm */
    mov rdi,[rip+g_xb]
    mov rsi,[rip+g_x]
    mov rdx,[r15+LD_FFN_NORM]
    mov rcx,[rip+g_dim]
    call rmsnorm
    /* gate */
    mov rdi,[r15+LD_W1]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_h1]
    mov rcx,[rip+g_hidden]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec
    /* up */
    mov rdi,[r15+LD_W3]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_h2]
    mov rcx,[rip+g_hidden]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec
    mov rdi,[rip+g_h1]
    mov rsi,[rip+g_h2]
    mov rdx,[rip+g_hidden]
    call silu_mul
    /* down */
    mov rdi,[r15+LD_W2]
    mov rsi,[rip+g_h1]
    mov rdx,[rip+g_xb2]
    mov rcx,[rip+g_dim]
    mov r8,[rip+g_hidden]
    mov r9,[rip+g_stride_hidden]
    call q8_matvec
    mov rdi,[rip+g_x]
    mov rsi,[rip+g_xb2]
    mov rdx,[rip+g_dim]
    call vec_add_inplace

    inc r14
    add r15,LAYER_DESC_SIZE
    jmp .ft_layer

.ft_final:
    mov rdi,[rip+g_xb]
    mov rsi,[rip+g_x]
    mov rdx,[rip+g_final_norm]
    mov rcx,[rip+g_dim]
    call rmsnorm
    mov rdi,[rip+g_output]
    mov rsi,[rip+g_xb]
    mov rdx,[rip+g_logits]
    mov rcx,[rip+g_vocab]
    mov r8,[rip+g_dim]
    mov r9,[rip+g_stride_dim]
    call q8_matvec

    /* greedy argmax */
    mov rdi,[rip+g_logits]
    mov rcx,[rip+g_vocab]
    xor eax,eax
    movss xmm0,dword ptr [rdi]
    mov rdx,1
.ft_argmax:
    cmp rdx,rcx
    jae .ft_ret
    movss xmm1,dword ptr [rdi+rdx*4]
    comiss xmm1,xmm0
    jbe .ft_am_next
    movaps xmm0,xmm1
    mov rax,rdx
.ft_am_next:
    inc rdx
    jmp .ft_argmax
.ft_ret:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

generate_tokens:
    /* rdi=prompt ptr u64 IDs, rsi=prompt_n, rdx=max_new; returns rax=count */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rdi
    mov r13,rsi
    mov r14,rdx
    xor r15d,r15d
    test r13,r13
    jz .gt_none
    cmp r13,[rip+g_max_seq]
    jae .gt_none
    /* prefill all prompt tokens; next candidate from last logits */
    xor ebx,ebx
.gt_prefill:
    mov rdi,[r12+rbx*8]
    mov rsi,rbx
    call forward_token
    inc rbx
    cmp rbx,r13
    jb .gt_prefill
    /* rax is token sampled from final prompt token */
    mov rbp,rax
.gt_gen:
    cmp r15,r14
    jae .gt_done
    mov rax,r13
    add rax,r15
    cmp rax,[rip+g_max_seq]
    jae .gt_done
    mov rcx,[rip+g_output_tokens]
    mov [rcx+r15*8],rbp
    inc r15
    cmp qword ptr [rip+g_eos_count],0
    je .gt_continue
    cmp rbp,[rip+g_eos1]
    je .gt_done
    cmp qword ptr [rip+g_eos_count],1
    je .gt_continue
    cmp rbp,[rip+g_eos2]
    je .gt_done
.gt_continue:
    cmp r15,r14
    jae .gt_done
    mov rdi,rbp
    mov rsi,r13
    add rsi,r15
    dec rsi
    call forward_token
    mov rbp,rax
    jmp .gt_gen
.gt_done:
    mov rax,r15
    jmp .gt_ret
.gt_none:
    xor eax,eax
.gt_ret:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

/* -------------------------------------------------------------------------- */
/* Minimal HTTP serving                                                       */
/* -------------------------------------------------------------------------- */

setup_server:
    mov eax,SYS_socket
    mov edi,AF_INET
    mov esi,SOCK_STREAM
    xor edx,edx
    syscall
    test rax,rax
    js .ss_fail
    mov [rip+g_listen_fd],rax
    mov rbx,rax
    sub rsp,32
    mov dword ptr [rsp],1
    mov eax,SYS_setsockopt
    mov rdi,rbx
    mov esi,SOL_SOCKET
    mov edx,SO_REUSEADDR
    lea r10,[rsp]
    mov r8d,4
    syscall

    /* sockaddr_in */
    mov qword ptr [rsp+8],0
    mov qword ptr [rsp+16],0
    mov word ptr [rsp+8],AF_INET
    mov rax,[rip+g_port]
    /* htons 16-bit */
    rol ax,8
    mov word ptr [rsp+10],ax
    mov eax,SYS_bind
    mov rdi,rbx
    lea rsi,[rsp+8]
    mov edx,16
    syscall
    test rax,rax
    js .ss_fail_stack
    mov eax,SYS_listen
    mov rdi,rbx
    mov esi,256
    syscall
    test rax,rax
    js .ss_fail_stack
    add rsp,32
    xor eax,eax
    ret
.ss_fail_stack:
    add rsp,32
.ss_fail:
    mov eax,1
    ret

parse_request:
    /* rdi=buffer rsi=len => rax=prompt_n, rdx=max_new; 0 invalid */
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12,rdi
    mov r13,rsi

    mov rdi,r12
    mov rsi,r13
    lea rdx,[rip+key_tokens]
    mov ecx,key_tokens_end-key_tokens
    call find_bytes
    test rax,rax
    jz .pr_bad
    mov rbx,rax
    /* find '[' after key */
    mov rbp,r12
    add rbp,r13
.pr_find_lb:
    cmp rbx,rbp
    jae .pr_bad
    cmp byte ptr [rbx],'['
    je .pr_array
    inc rbx
    jmp .pr_find_lb
.pr_array:
    inc rbx
    xor r14d,r14d
    mov r15,[rip+g_prompt_buf]
.pr_next:
    cmp rbx,rbp
    jae .pr_bad
    mov al,byte ptr [rbx]
    cmp al,']'
    je .pr_arr_done
    cmp al,'0'
    jb .pr_skip
    cmp al,'9'
    ja .pr_skip
    xor rax,rax
.pr_num:
    cmp rbx,rbp
    jae .pr_bad
    movzx ecx,byte ptr [rbx]
    cmp cl,'0'
    jb .pr_num_done
    cmp cl,'9'
    ja .pr_num_done
    imul rax,rax,10
    sub ecx,'0'
    add rax,rcx
    inc rbx
    jmp .pr_num
.pr_num_done:
    cmp rax,[rip+g_vocab]
    jae .pr_bad
    cmp r14,MAX_PROMPT_TOKENS
    jae .pr_bad
    mov [r15+r14*8],rax
    inc r14
    jmp .pr_next
.pr_skip:
    inc rbx
    jmp .pr_next
.pr_arr_done:
    test r14,r14
    jz .pr_bad

    /* optional max_tokens */
    mov rdi,r12
    mov rsi,r13
    lea rdx,[rip+key_max]
    mov ecx,key_max_end-key_max
    call find_bytes
    mov rdx,8
    test rax,rax
    jz .pr_ok
    mov rbx,rax
.pr_find_colon:
    cmp rbx,rbp
    jae .pr_ok
    cmp byte ptr [rbx],':'
    je .pr_maxnum_seek
    inc rbx
    jmp .pr_find_colon
.pr_maxnum_seek:
    inc rbx
.pr_ms:
    cmp rbx,rbp
    jae .pr_ok
    mov al,byte ptr [rbx]
    cmp al,'0'
    jb .pr_mskip
    cmp al,'9'
    jbe .pr_mparse
.pr_mskip:
    inc rbx
    jmp .pr_ms
.pr_mparse:
    xor rdx,rdx
.pr_ml:
    cmp rbx,rbp
    jae .pr_mdone
    movzx ecx,byte ptr [rbx]
    cmp cl,'0'
    jb .pr_mdone
    cmp cl,'9'
    ja .pr_mdone
    imul rdx,rdx,10
    sub ecx,'0'
    add rdx,rcx
    inc rbx
    jmp .pr_ml
.pr_mdone:
    test rdx,rdx
    jnz .pr_cap
    mov rdx,1
.pr_cap:
    cmp rdx,MAX_GEN_TOKENS
    jbe .pr_ok
    mov rdx,MAX_GEN_TOKENS
.pr_ok:
    mov rax,r14
    jmp .pr_ret
.pr_bad:
    xor eax,eax
    xor edx,edx
.pr_ret:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

handle_client:
    /* rdi=client fd */
    push rbx
    push r12
    push r13
    push r14
    mov rbx,rdi
    mov eax,SYS_read
    mov rdi,rbx
    mov rsi,[rip+g_http_buf]
    mov edx,HTTP_BUF_SIZE
    syscall
    test rax,rax
    jle .hc_close
    mov rdi,[rip+g_http_buf]
    mov rsi,rax
    call parse_request
    test rax,rax
    jz .hc_bad
    mov r12,rax
    mov r14,rdx
    mov rdi,[rip+g_prompt_buf]
    mov rsi,r12
    mov rdx,r14
    call generate_tokens
    mov r13,rax

    mov rdi,rbx
    lea rsi,[rip+http_ok]
    mov edx,http_ok_end-http_ok
    call write_all
    xor r12d,r12d
.hc_tokloop:
    cmp r12,r13
    jae .hc_tail
    test r12,r12
    jz .hc_nocomma
    mov rdi,rbx
    lea rsi,[rip+comma]
    mov edx,1
    call write_all
.hc_nocomma:
    sub rsp,32
    mov rax,[rip+g_output_tokens]
    mov rdi,[rax+r12*8]
    lea rsi,[rsp]
    call u64_to_ascii
    mov rdx,rax
    mov rdi,rbx
    lea rsi,[rsp]
    call write_all
    add rsp,32
    inc r12
    jmp .hc_tokloop
.hc_tail:
    mov rdi,rbx
    lea rsi,[rip+http_tail]
    mov edx,http_tail_end-http_tail
    call write_all
    jmp .hc_close
.hc_bad:
    mov rdi,rbx
    lea rsi,[rip+http_bad]
    mov edx,http_bad_end-http_bad
    call write_all
.hc_close:
    mov eax,SYS_close
    mov rdi,rbx
    syscall
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

server_loop:
.sl_accept:
    mov eax,SYS_accept
    mov rdi,[rip+g_listen_fd]
    xor esi,esi
    xor edx,edx
    syscall
    test rax,rax
    js .sl_accept
    mov rdi,rax
    call handle_client
    jmp .sl_accept

/* -------------------------------------------------------------------------- */
/* Entry                                                                       */
/* -------------------------------------------------------------------------- */

_start:
    mov r15,rsp
    and rsp,-16
    lea rsi,[rip+banner]
    mov edx,banner_end-banner
    mov edi,2
    call write_all

    mov r14,[r15]                  /* argc */
    lea r13,[r15+8]                /* argv */
    cmp r14,2
    jae .main_have_model
    lea rdi,[rip+default_model]
    jmp .main_map_model
.main_have_model:
    mov rdi,[r13+8]
.main_map_model:
    call map_model
    test rax,rax
    jz .main_args
    cmp rax,1
    jne .main_not_open
    lea rsi,[rip+err_open]
    mov edx,err_open_end-err_open
    call print_stderr
    mov edi,1
    jmp exit_now
.main_not_open:
    cmp rax,2
    jne .main_bad_model
    lea rsi,[rip+err_mmap]
    mov edx,err_mmap_end-err_mmap
    call print_stderr
    mov edi,1
    jmp exit_now
.main_bad_model:
    lea rsi,[rip+err_magic]
    mov edx,err_magic_end-err_magic
    call print_stderr
    mov edi,1
    jmp exit_now

.main_args:
    call detect_avx512
    mov dword ptr [rip+g_use_avx512], eax
    call detect_workers
    call tune_auto_workers
    cmp rax,128
    jbe .main_auto_workers_ok
    mov rax,128
.main_auto_workers_ok:
    mov [rip+g_workers],rax
    cmp r14,3
    jb .main_port
    mov rdi,[r13+16]
    call parse_u64
    test rax,rax
    jz .main_port
    cmp rax,128
    jbe .main_workers_ok
    mov rax,128
.main_workers_ok:
    mov rcx, [rip+g_allowed_workers]
    test rcx, rcx
    jz .main_workers_store
    cmp rax, rcx
    jbe .main_workers_store
    mov rax, rcx
.main_workers_store:
    mov [rip+g_workers],rax
.main_port:
    mov qword ptr [rip+g_port],8080
    cmp r14,4
    jb .main_alloc
    mov rdi,[r13+24]
    call parse_u64
    test rax,rax
    jz .main_alloc
    cmp rax,65535
    ja .main_alloc
    mov [rip+g_port],rax

.main_alloc:
    call alloc_arena
    test rax,rax
    jz .main_spawn
    lea rsi,[rip+err_mmap]
    mov edx,err_mmap_end-err_mmap
    call print_stderr
    mov edi,1
    jmp exit_now
.main_spawn:
    call spawn_workers
    call setup_server
    test rax,rax
    jz .main_ready
    lea rsi,[rip+err_socket]
    mov edx,err_socket_end-err_socket
    call print_stderr
    mov edi,1
    jmp exit_now

.main_ready:
    mov edi,2
    lea rsi,[rip+ready1]
    mov edx,ready1_end-ready1
    call write_all
    sub rsp,32
    mov rdi,[rip+g_port]
    lea rsi,[rsp]
    call u64_to_ascii
    mov rdx,rax
    mov edi,2
    lea rsi,[rsp]
    call write_all
    mov edi,2
    lea rsi,[rip+ready2]
    mov edx,ready2_end-ready2
    call write_all
    mov rdi,[rip+g_workers]
    lea rsi,[rsp]
    call u64_to_ascii
    mov rdx,rax
    mov edi,2
    lea rsi,[rsp]
    call write_all
    mov edi,2
    lea rsi,[rip+nl]
    mov edx,1
    call write_all
    add rsp,32
    call server_loop
    xor edi,edi
    jmp exit_now
