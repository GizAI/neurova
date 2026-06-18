#include "kernels.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

constexpr int LB_MLP_BLOCK = 256;

__device__ __constant__ int LB_MARLIN_INV_PERM[1024] = {
  0, 128, 256, 384, 512, 640, 768, 896, 2, 130, 258, 386, 514, 642, 770, 898,
  4, 132, 260, 388, 516, 644, 772, 900, 6, 134, 262, 390, 518, 646, 774, 902,
  32, 160, 288, 416, 544, 672, 800, 928, 34, 162, 290, 418, 546, 674, 802, 930,
  36, 164, 292, 420, 548, 676, 804, 932, 38, 166, 294, 422, 550, 678, 806, 934,
  64, 192, 320, 448, 576, 704, 832, 960, 66, 194, 322, 450, 578, 706, 834, 962,
  68, 196, 324, 452, 580, 708, 836, 964, 70, 198, 326, 454, 582, 710, 838, 966,
  96, 224, 352, 480, 608, 736, 864, 992, 98, 226, 354, 482, 610, 738, 866, 994,
  100, 228, 356, 484, 612, 740, 868, 996, 102, 230, 358, 486, 614, 742, 870, 998,
  1, 129, 257, 385, 513, 641, 769, 897, 3, 131, 259, 387, 515, 643, 771, 899,
  5, 133, 261, 389, 517, 645, 773, 901, 7, 135, 263, 391, 519, 647, 775, 903,
  33, 161, 289, 417, 545, 673, 801, 929, 35, 163, 291, 419, 547, 675, 803, 931,
  37, 165, 293, 421, 549, 677, 805, 933, 39, 167, 295, 423, 551, 679, 807, 935,
  65, 193, 321, 449, 577, 705, 833, 961, 67, 195, 323, 451, 579, 707, 835, 963,
  69, 197, 325, 453, 581, 709, 837, 965, 71, 199, 327, 455, 583, 711, 839, 967,
  97, 225, 353, 481, 609, 737, 865, 993, 99, 227, 355, 483, 611, 739, 867, 995,
  101, 229, 357, 485, 613, 741, 869, 997, 103, 231, 359, 487, 615, 743, 871, 999,
  8, 136, 264, 392, 520, 648, 776, 904, 10, 138, 266, 394, 522, 650, 778, 906,
  12, 140, 268, 396, 524, 652, 780, 908, 14, 142, 270, 398, 526, 654, 782, 910,
  40, 168, 296, 424, 552, 680, 808, 936, 42, 170, 298, 426, 554, 682, 810, 938,
  44, 172, 300, 428, 556, 684, 812, 940, 46, 174, 302, 430, 558, 686, 814, 942,
  72, 200, 328, 456, 584, 712, 840, 968, 74, 202, 330, 458, 586, 714, 842, 970,
  76, 204, 332, 460, 588, 716, 844, 972, 78, 206, 334, 462, 590, 718, 846, 974,
  104, 232, 360, 488, 616, 744, 872, 1000, 106, 234, 362, 490, 618, 746, 874, 1002,
  108, 236, 364, 492, 620, 748, 876, 1004, 110, 238, 366, 494, 622, 750, 878, 1006,
  9, 137, 265, 393, 521, 649, 777, 905, 11, 139, 267, 395, 523, 651, 779, 907,
  13, 141, 269, 397, 525, 653, 781, 909, 15, 143, 271, 399, 527, 655, 783, 911,
  41, 169, 297, 425, 553, 681, 809, 937, 43, 171, 299, 427, 555, 683, 811, 939,
  45, 173, 301, 429, 557, 685, 813, 941, 47, 175, 303, 431, 559, 687, 815, 943,
  73, 201, 329, 457, 585, 713, 841, 969, 75, 203, 331, 459, 587, 715, 843, 971,
  77, 205, 333, 461, 589, 717, 845, 973, 79, 207, 335, 463, 591, 719, 847, 975,
  105, 233, 361, 489, 617, 745, 873, 1001, 107, 235, 363, 491, 619, 747, 875, 1003,
  109, 237, 365, 493, 621, 749, 877, 1005, 111, 239, 367, 495, 623, 751, 879, 1007,
  16, 144, 272, 400, 528, 656, 784, 912, 18, 146, 274, 402, 530, 658, 786, 914,
  20, 148, 276, 404, 532, 660, 788, 916, 22, 150, 278, 406, 534, 662, 790, 918,
  48, 176, 304, 432, 560, 688, 816, 944, 50, 178, 306, 434, 562, 690, 818, 946,
  52, 180, 308, 436, 564, 692, 820, 948, 54, 182, 310, 438, 566, 694, 822, 950,
  80, 208, 336, 464, 592, 720, 848, 976, 82, 210, 338, 466, 594, 722, 850, 978,
  84, 212, 340, 468, 596, 724, 852, 980, 86, 214, 342, 470, 598, 726, 854, 982,
  112, 240, 368, 496, 624, 752, 880, 1008, 114, 242, 370, 498, 626, 754, 882, 1010,
  116, 244, 372, 500, 628, 756, 884, 1012, 118, 246, 374, 502, 630, 758, 886, 1014,
  17, 145, 273, 401, 529, 657, 785, 913, 19, 147, 275, 403, 531, 659, 787, 915,
  21, 149, 277, 405, 533, 661, 789, 917, 23, 151, 279, 407, 535, 663, 791, 919,
  49, 177, 305, 433, 561, 689, 817, 945, 51, 179, 307, 435, 563, 691, 819, 947,
  53, 181, 309, 437, 565, 693, 821, 949, 55, 183, 311, 439, 567, 695, 823, 951,
  81, 209, 337, 465, 593, 721, 849, 977, 83, 211, 339, 467, 595, 723, 851, 979,
  85, 213, 341, 469, 597, 725, 853, 981, 87, 215, 343, 471, 599, 727, 855, 983,
  113, 241, 369, 497, 625, 753, 881, 1009, 115, 243, 371, 499, 627, 755, 883, 1011,
  117, 245, 373, 501, 629, 757, 885, 1013, 119, 247, 375, 503, 631, 759, 887, 1015,
  24, 152, 280, 408, 536, 664, 792, 920, 26, 154, 282, 410, 538, 666, 794, 922,
  28, 156, 284, 412, 540, 668, 796, 924, 30, 158, 286, 414, 542, 670, 798, 926,
  56, 184, 312, 440, 568, 696, 824, 952, 58, 186, 314, 442, 570, 698, 826, 954,
  60, 188, 316, 444, 572, 700, 828, 956, 62, 190, 318, 446, 574, 702, 830, 958,
  88, 216, 344, 472, 600, 728, 856, 984, 90, 218, 346, 474, 602, 730, 858, 986,
  92, 220, 348, 476, 604, 732, 860, 988, 94, 222, 350, 478, 606, 734, 862, 990,
  120, 248, 376, 504, 632, 760, 888, 1016, 122, 250, 378, 506, 634, 762, 890, 1018,
  124, 252, 380, 508, 636, 764, 892, 1020, 126, 254, 382, 510, 638, 766, 894, 1022,
  25, 153, 281, 409, 537, 665, 793, 921, 27, 155, 283, 411, 539, 667, 795, 923,
  29, 157, 285, 413, 541, 669, 797, 925, 31, 159, 287, 415, 543, 671, 799, 927,
  57, 185, 313, 441, 569, 697, 825, 953, 59, 187, 315, 443, 571, 699, 827, 955,
  61, 189, 317, 445, 573, 701, 829, 957, 63, 191, 319, 447, 575, 703, 831, 959,
  89, 217, 345, 473, 601, 729, 857, 985, 91, 219, 347, 475, 603, 731, 859, 987,
  93, 221, 349, 477, 605, 733, 861, 989, 95, 223, 351, 479, 607, 735, 863, 991,
  121, 249, 377, 505, 633, 761, 889, 1017, 123, 251, 379, 507, 635, 763, 891, 1019,
  125, 253, 381, 509, 637, 765, 893, 1021, 127, 255, 383, 511, 639, 767, 895, 1023
};

__device__ __constant__ int LB_MARLIN_INV_SCALE_PERM[64] = {
  0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57,
  2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
  4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
  6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63
};

__device__ __constant__ int LB_MARLIN_INV_SCALE_PERM_SINGLE[32] = {
  0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
  4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31
};


__device__ __forceinline__ float lb_mlp_silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float lb_marlin_scale_read(
    const half* __restrict__ scales,
    int group,
    int out_col,
    int prob_n,
    int group_size) {
  if (group_size == -1) {
    int flat = out_col;
    int base = (flat / 32) * 32;
    int local = flat - base;
    return __half2float(scales[base + LB_MARLIN_INV_SCALE_PERM_SINGLE[local]]);
  }
  int flat = group * prob_n + out_col;
  int base = (flat / 64) * 64;
  int local = flat - base;
  return __half2float(scales[base + LB_MARLIN_INV_SCALE_PERM[local]]);
}

__device__ __forceinline__ float lb_marlin_weight_read(
    const int* __restrict__ qweight,
    const half* __restrict__ scales,
    int out_col,
    int in_col,
    int prob_n,
    int prob_k,
    int group_size) {
  int kblock = in_col >> 4;
  int kinner = in_col & 15;
  int nblock = out_col >> 4;
  int ninner = out_col & 15;
  int pre_col = nblock * 256 + kinner * 16 + ninner;
  int block_base = (pre_col / 1024) * 1024;
  int local_pre = pre_col - block_base;
  int final_col = block_base + LB_MARLIN_INV_PERM[local_pre];
  int packed_idx = final_col >> 3;
  int nibble = final_col & 7;
  int qcols_int = prob_n * 2;
  uint32_t word = static_cast<uint32_t>(qweight[static_cast<int64_t>(kblock) * qcols_int + packed_idx]);
  int q = static_cast<int>((word >> (4 * nibble)) & 0x0f) - 8;
  int group = group_size == -1 ? 0 : (in_col / group_size);
  float scale = lb_marlin_scale_read(scales, group, out_col, prob_n, group_size);
  return static_cast<float>(q) * scale;
}

__global__ void lowbit_marlin_mlp_streaming_kernel(
    const int* __restrict__ gate_up_qweight,
    const half* __restrict__ gate_up_scales,
    const int* __restrict__ down_qweight,
    const half* __restrict__ down_scales,
    const half* __restrict__ x,
    half* __restrict__ out,
    float* __restrict__ accum,
    int* __restrict__ sync,
    int epoch,
    int batch,
    int hidden,
    int intermediate,
    int gate_group_size,
    int down_group_size) {
  int i = blockIdx.x;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= batch || i >= intermediate) return;

  int* done = sync;
  int* ready = sync + batch;
  if (i == 0) {
    for (int o = tid; o < hidden; o += blockDim.x) {
      accum[static_cast<int64_t>(row) * hidden + o] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
      done[row] = 0;
      __threadfence();
      ready[row] = epoch;
    }
  }
  if (tid == 0) {
    while (atomicAdd(&ready[row], 0) != epoch) {}
  }
  __syncthreads();

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  const half* x_row = x + static_cast<int64_t>(row) * hidden;
  #pragma unroll 1
  for (int h = tid; h < hidden; h += blockDim.x) {
    float xv = __half2float(x_row[h]);
    float wg = lb_marlin_weight_read(gate_up_qweight, gate_up_scales, i, h, 2 * intermediate, hidden, gate_group_size);
    float wu = lb_marlin_weight_read(gate_up_qweight, gate_up_scales, intermediate + i, h, 2 * intermediate, hidden, gate_group_size);
    gate_acc += wg * xv;
    up_acc += wu * xv;
  }

  __shared__ float red_gate[LB_MLP_BLOCK];
  __shared__ float red_up[LB_MLP_BLOCK];
  red_gate[tid] = gate_acc;
  red_up[tid] = up_acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_gate[tid] += red_gate[tid + stride];
      red_up[tid] += red_up[tid + stride];
    }
    __syncthreads();
  }
  float act = lb_mlp_silu(red_gate[0]) * red_up[0];

  for (int o = tid; o < hidden; o += blockDim.x) {
    float wd = lb_marlin_weight_read(down_qweight, down_scales, o, i, hidden, intermediate, down_group_size);
    atomicAdd(&accum[static_cast<int64_t>(row) * hidden + o], wd * act);
  }
  __threadfence();
  if (tid == 0) {
    atomicAdd(&done[row], 1);
  }

  if (i == 0) {
    if (tid == 0) {
      while (atomicAdd(&done[row], 0) < intermediate) {}
    }
    __syncthreads();
    for (int o = tid; o < hidden; o += blockDim.x) {
      out[static_cast<int64_t>(row) * hidden + o] = __float2half_rn(accum[static_cast<int64_t>(row) * hidden + o]);
    }
  }
}

void lowbit_marlin_mlp_streaming_out(
    torch::Tensor gate_up_qweight,
    torch::Tensor gate_up_scales,
    torch::Tensor down_qweight,
    torch::Tensor down_scales,
    torch::Tensor x,
    torch::Tensor out,
    torch::Tensor accum,
    torch::Tensor sync,
    int64_t epoch_i,
    int64_t hidden_i,
    int64_t intermediate_i,
    int64_t gate_group_size_i,
    int64_t down_group_size_i) {
  QB_CHECK_CUDA(gate_up_qweight); QB_CHECK_CUDA(gate_up_scales); QB_CHECK_CUDA(down_qweight); QB_CHECK_CUDA(down_scales);
  QB_CHECK_CUDA(x); QB_CHECK_CUDA(out); QB_CHECK_CUDA(accum); QB_CHECK_CUDA(sync);
  QB_CHECK_CONTIGUOUS(gate_up_qweight); QB_CHECK_CONTIGUOUS(gate_up_scales); QB_CHECK_CONTIGUOUS(down_qweight); QB_CHECK_CONTIGUOUS(down_scales);
  QB_CHECK_CONTIGUOUS(x); QB_CHECK_CONTIGUOUS(out); QB_CHECK_CONTIGUOUS(accum); QB_CHECK_CONTIGUOUS(sync);
  TORCH_CHECK(gate_up_qweight.scalar_type() == at::kInt && down_qweight.scalar_type() == at::kInt, "Marlin qweight must be int32");
  QB_CHECK_HALF(gate_up_scales); QB_CHECK_HALF(down_scales); QB_CHECK_HALF(x); QB_CHECK_HALF(out);
  QB_CHECK_FLOAT(accum);
  TORCH_CHECK(sync.scalar_type() == at::kInt, "sync must be int32");
  TORCH_CHECK(x.dim() == 2, "x must be [batch, hidden]");
  int batch = static_cast<int>(x.size(0));
  int hidden = static_cast<int>(hidden_i);
  int intermediate = static_cast<int>(intermediate_i);
  TORCH_CHECK(hidden > 0 && intermediate > 0, "hidden/intermediate must be positive");
  TORCH_CHECK(hidden % 16 == 0 && intermediate % 16 == 0, "hidden/intermediate must be divisible by 16");
  TORCH_CHECK(x.size(1) == hidden, "x hidden mismatch");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == batch && out.size(1) == hidden, "out must be [batch, hidden]");
  TORCH_CHECK(accum.dim() == 2 && accum.size(0) == batch && accum.size(1) == hidden, "accum must be [batch, hidden]");
  TORCH_CHECK(sync.dim() == 1 && sync.numel() >= 2 * batch, "sync must have at least 2*batch int32 elements");
  TORCH_CHECK(gate_up_qweight.dim() == 2 && gate_up_qweight.size(0) == hidden / 16 && gate_up_qweight.size(1) == 4 * intermediate,
              "gate_up qweight must be [hidden/16, (2*intermediate)*2]");
  TORCH_CHECK(down_qweight.dim() == 2 && down_qweight.size(0) == intermediate / 16 && down_qweight.size(1) == 2 * hidden,
              "down qweight must be [intermediate/16, hidden*2]");
  int gate_group = static_cast<int>(gate_group_size_i);
  int down_group = static_cast<int>(down_group_size_i);
  if (gate_group <= 0 || gate_group == hidden) gate_group = -1;
  if (down_group <= 0 || down_group == intermediate) down_group = -1;
  TORCH_CHECK(gate_group == -1 || gate_group == 128, "gate_up Marlin group_size must be -1 or 128");
  TORCH_CHECK(down_group == -1 || down_group == 128, "down Marlin group_size must be -1 or 128");
  int gate_groups = gate_group == -1 ? 1 : hidden / gate_group;
  int down_groups = down_group == -1 ? 1 : intermediate / down_group;
  TORCH_CHECK(gate_up_scales.dim() == 2 && gate_up_scales.size(0) == gate_groups && gate_up_scales.size(1) == 2 * intermediate,
              "gate_up scales shape mismatch");
  TORCH_CHECK(down_scales.dim() == 2 && down_scales.size(0) == down_groups && down_scales.size(1) == hidden,
              "down scales shape mismatch");
  TORCH_CHECK(epoch_i > 0, "epoch must be positive");

  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  dim3 grid(intermediate, batch);
  lowbit_marlin_mlp_streaming_kernel<<<grid, LB_MLP_BLOCK, 0, stream>>>(
      gate_up_qweight.data_ptr<int>(),
      reinterpret_cast<const half*>(gate_up_scales.data_ptr<at::Half>()),
      down_qweight.data_ptr<int>(),
      reinterpret_cast<const half*>(down_scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      accum.data_ptr<float>(),
      sync.data_ptr<int>(),
      static_cast<int>(epoch_i),
      batch,
      hidden,
      intermediate,
      gate_group,
      down_group);
  QB_CUDA_CHECK(cudaGetLastError());
}
