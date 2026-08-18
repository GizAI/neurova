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
g_port:             .quad 0

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
    cmp dword ptr [rdi+8], 1
    jne .mm_magic_fail

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
    mov rax, [rip+g_q_dim]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    /* wk */
    mov [r14+LD_WK], r12
    mov rax, [rip+g_kv_dim]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    /* wv */
    mov [r14+LD_WV], r12
    mov rax, [rip+g_kv_dim]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    /* wo */
    mov [r14+LD_WO], r12
    mov rax, [rip+g_dim]
    imul rax, [rip+g_stride_q]
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
    mov rax, [rip+g_hidden]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    /* w3 up */
    mov [r14+LD_W3], r12
    mov rax, [rip+g_hidden]
    imul rax, [rip+g_stride_dim]
    add r12, rax

    /* w2 down */
    mov [r14+LD_W2], r12
    mov rax, [rip+g_dim]
    imul rax, [rip+g_stride_hidden]
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
    mov rax, [rip+g_vocab]
    imul rax, [rip+g_stride_dim]
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
    js .sw_parent_done
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
.sw_parent_done:
    xor edi, edi
    call pin_current_cpu
    pop rbx
    ret

detect_workers:
    /* return affinity CPU count in rax, capped to 128; fallback 1 */
    sub rsp,32
    pxor xmm0,xmm0
    movdqu xmmword ptr [rsp],xmm0
    mov eax,SYS_sched_getaffinity
    xor edi,edi
    mov esi,16
    lea rdx,[rsp]
    syscall
    test rax,rax
    js .dw_one
    mov r8,[rsp]
    mov r9,[rsp+8]
    xor ecx,ecx
.dw_l0:
    test r8,r8
    jz .dw_l1
    lea rax,[r8-1]
    and r8,rax
    inc ecx
    jmp .dw_l0
.dw_l1:
    test r9,r9
    jz .dw_done
    lea rax,[r9-1]
    and r9,rax
    inc ecx
    jmp .dw_l1
.dw_done:
    mov eax,ecx
    test rax,rax
    jnz .dw_ret
.dw_one:
    mov eax,1
.dw_ret:
    add rsp,32
    ret

pin_current_cpu:
    /* rdi logical worker id; pin to its ordinal online CPU */
    sub rsp, 48
    mov [rsp+32], rdi
    mov eax, SYS_sched_getaffinity
    xor edi, edi
    mov esi, 16
    lea rdx, [rsp]
    syscall
    test rax, rax
    js .pcc_done
    mov r8, [rsp]
    mov r9, [rsp+8]
    mov r10, [rsp+32]
    xor eax, eax
.pcc_low_scan:
    test r8, r8
    jz .pcc_high_scan
    test r10, r10
    jz .pcc_low_found
    lea rcx, [r8-1]
    and r8, rcx
    dec r10
    jmp .pcc_low_scan
.pcc_low_found:
    bsf rcx, r8
    mov rax, 1
    shl rax, cl
    mov [rsp+16], rax
    mov qword ptr [rsp+24], 0
    jmp .pcc_call
.pcc_high_scan:
    test r9, r9
    jz .pcc_done
    test r10, r10
    jz .pcc_high_found
    lea rcx, [r9-1]
    and r9, rcx
    dec r10
    jmp .pcc_high_scan
.pcc_high_found:
    bsf rcx, r9
    mov rax, 1
    shl rax, cl
    mov qword ptr [rsp+16], 0
    mov [rsp+24], rax
.pcc_call:
    mov eax, SYS_sched_setaffinity
    xor edi, edi
    mov esi, 16
    lea rdx, [rsp+16]
    syscall
.pcc_done:
    add rsp, 48
    ret

worker_loop:
    /* rdi=worker id */
    push r12
    push r13
    push r14
    mov r12, rdi
    mov r13, [rip+g_job]
    mov r14, [r13+JOB_SEQ]
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
    mfence
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
    mulss xmm0,dword ptr [rip+g_att_scale]
    mov rax,[rip+g_scores]
    movss dword ptr [rax+rbp*4],xmm0
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
    call detect_workers
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
