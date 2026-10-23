//! Apple Metal kernels for pinned MACE FP32 runtime paths.
//!
//! This module is intentionally feature-gated and narrow: it is not a generic
//! MACE/e3nn backend. The MP0b kernels are the performance anchor, and small
//! generic boundaries such as node embedding are shared with Osaka/OAM so later
//! slices can extend the same command/buffer path into fused layer kernels.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

const MP0B_METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Mp0bNodeEmbeddingParams {
    uint num_nodes;
    uint out_features;
    float path_weight;
};

struct Mp0bDense64VjpParams {
    uint num_edges;
    uint out_dim;
    float inv_norm;
};

struct Mp0bEdgeFeatVjpParams {
    uint num_edges;
    uint num_bessel;
    float inv_norm;
    uint flags;
};

struct Mp0bDensityVjpEdgeFeatParams {
    uint num_edges;
    uint num_bessel;
};

struct Mp0bDenseTo64Params {
    uint num_edges;
    uint in_dim;
    float inv_norm;
    float silu_scale;
};

struct Mp0bDense64ForwardParams {
    uint num_edges;
    uint out_dim;
    float inv_norm;
    float silu_scale;
};

struct Mp0bLayer1HiddenVjpParams {
    uint num_nodes;
};

struct Mp0bLayer1ConvTpVjpParams {
    uint num_nodes;
    uint num_edges;
    uint flags;
};

struct Mp0bLayer1TpTemplate {
    uint x_base;
    uint x_dim;
    uint y_base;
    uint out_base;
    uint out_dim;
    uint weight_base;
};

struct Mp0bLayer1Cg {
    uint i_m;
    uint j_m;
    uint k_m;
    float scale;
};

struct Mp0bLayer1WeightLane {
    uint template_index;
    uint feature;
};

struct Mp0bLayer1InputLane {
    uint template_index;
    uint feature;
    uint input_component;
};

struct Mp0bVectorU1Entry {
    uint ell;
    uint param;
    float value;
};

struct Mp0bVectorU2Entry {
    uint left;
    uint right;
    uint param;
    float value;
};

struct Mp0bVectorU3Entry {
    uint left;
    uint right;
    uint inner;
    uint param;
    float value;
};

struct Mp0bProduct0VectorVjpParams {
    uint num_nodes;
    uint num_elements;
    uint num_features;
    uint num_equivariance;
    uint u1_params;
    uint u2_params;
    uint u3_params;
    uint output_stride;
    uint output_offset;
};

struct Mp0bProduct0ScalarVjpParams {
    uint num_nodes;
    uint u1_len;
    uint u2_len;
    uint u3_len;
    uint flags;
};

struct MaceScalarVjpParams {
    uint num_nodes;
    uint num_elements;
    uint num_features;
    uint num_ell;
    uint u1_params;
    uint u2_params;
    uint u3_params;
    uint u1_len;
    uint u2_len;
    uint u3_len;
};

struct MaceScalarForwardParams {
    uint num_nodes;
    uint num_elements;
    uint num_features;
    uint num_ell;
    uint u1_params;
    uint u2_params;
    uint u3_params;
    uint u1_len;
    uint u2_len;
    uint u3_len;
    uint out_stride;
    uint out_offset;
};

kernel void mp0b_node_embedding_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Mp0bNodeEmbeddingParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * params.out_features;
    if (gid >= total) {
        return;
    }
    uint node = gid / params.out_features;
    uint feature = gid - node * params.out_features;
    uint type_id = type_ids[node];
    output[gid] = weight[type_id * params.out_features + feature]
        * params.path_weight
        * output_mask[feature];
}

kernel void mp0b_dense64_vjp_derivative_f32(
    device const float* bar_out [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* derivative [[buffer(2)]],
    device float* bar_in [[buffer(3)]],
    constant Mp0bDense64VjpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 64;
    uint hidden = gid - edge * 64;
    float acc = 0.0f;
    uint bar_base = edge * params.out_dim;
    uint weight_base = hidden * params.out_dim;
    for (uint out = 0; out < params.out_dim; out++) {
        acc += bar_out[bar_base + out] * weight[weight_base + out];
    }
    bar_in[gid] = acc * params.inv_norm * derivative[gid];
}

kernel void mp0b_apply_derivative64_f32(
    device const float* bar_in [[buffer(0)]],
    device const float* derivative [[buffer(1)]],
    device float* bar_out [[buffer(2)]],
    constant Mp0bDense64VjpParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    bar_out[gid] = bar_in[gid] * derivative[gid];
}

kernel void mp0b_edge_feats_vjp_add_f32(
    device const float* bar_hidden0 [[buffer(0)]],
    device const float* weight0 [[buffer(1)]],
    device float* bar_edge_feats [[buffer(2)]],
    constant Mp0bEdgeFeatVjpParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * params.num_bessel;
    if (gid >= total) {
        return;
    }
    uint edge = gid / params.num_bessel;
    uint basis = gid - edge * params.num_bessel;
    float acc = 0.0f;
    uint hidden_base = edge * 64;
    uint weight_base = basis * 64;
    for (uint hidden = 0; hidden < 64; hidden++) {
        acc += bar_hidden0[hidden_base + hidden] * weight0[weight_base + hidden];
    }
    float value = acc * params.inv_norm;
    if ((params.flags & 1u) != 0u) {
        bar_edge_feats[gid] = value;
    } else {
        bar_edge_feats[gid] += value;
    }
}

kernel void mp0b_density_vjp_edge_feats_add_f32(
    device const float* edge_feats [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const uint* receivers [[buffer(2)]],
    device const float* bar_density [[buffer(3)]],
    device float* bar_edge_feats [[buffer(4)]],
    constant Mp0bDensityVjpEdgeFeatParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * params.num_bessel;
    if (gid >= total) {
        return;
    }
    uint edge = gid / params.num_bessel;
    uint basis = gid - edge * params.num_bessel;
    float raw = 0.0f;
    uint edge_base = edge * params.num_bessel;
    for (uint input_basis = 0; input_basis < params.num_bessel; input_basis++) {
        raw += edge_feats[edge_base + input_basis] * weight[input_basis];
    }
    float edge_density = tanh(raw * raw);
    float bar_linear = bar_density[receivers[edge]]
        * (1.0f - edge_density * edge_density)
        * 2.0f
        * raw;
    bar_edge_feats[gid] += bar_linear * weight[basis];
}

kernel void mp0b_dense_to_64_silu_derivative_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* derivative [[buffer(3)]],
    constant Mp0bDenseTo64Params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 64;
    uint hidden = gid - edge * 64;
    float raw = 0.0f;
    uint input_base = edge * params.in_dim;
    for (uint input_feature = 0; input_feature < params.in_dim; input_feature++) {
        raw += input[input_base + input_feature] * weight[input_feature * 64 + hidden];
    }
    raw *= params.inv_norm;
    float sigmoid = 1.0f / (1.0f + exp(-raw));
    output[gid] = params.silu_scale * raw * sigmoid;
    derivative[gid] = params.silu_scale * sigmoid * (1.0f + raw * (1.0f - sigmoid));
}

kernel void mp0b_dense64_silu_derivative_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* derivative [[buffer(3)]],
    constant Mp0bDense64ForwardParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 64;
    uint hidden = gid - edge * 64;
    float raw = 0.0f;
    uint input_base = edge * 64;
    uint weight_base = hidden;
    for (uint input_feature = 0; input_feature < 64; input_feature++) {
        raw += input[input_base + input_feature] * weight[input_feature * 64 + weight_base];
    }
    raw *= params.inv_norm;
    float sigmoid = 1.0f / (1.0f + exp(-raw));
    output[gid] = params.silu_scale * raw * sigmoid;
    derivative[gid] = params.silu_scale * sigmoid * (1.0f + raw * (1.0f - sigmoid));
}

kernel void mp0b_dense64_output_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Mp0bDense64ForwardParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * params.out_dim;
    if (gid >= total) {
        return;
    }
    uint edge = gid / params.out_dim;
    uint out = gid - edge * params.out_dim;
    float acc = 0.0f;
    uint input_base = edge * 64;
    for (uint hidden = 0; hidden < 64; hidden++) {
        acc += input[input_base + hidden] * weight[hidden * params.out_dim + out];
    }
    output[gid] = acc * params.inv_norm;
}

kernel void mp0b_dense64_output_tiled_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Mp0bDense64ForwardParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_EDGES = 8;
    constexpr uint TILE_OUT = 8;
    constexpr uint HIDDEN = 64;
    threadgroup float input_tile[TILE_EDGES][HIDDEN];
    threadgroup float weight_tile[TILE_OUT][HIDDEN];

    uint out = gid.x;
    uint edge = gid.y;
    uint local_out = tid.x;
    uint local_edge = tid.y;
    uint out_base = out - local_out;
    uint edge_base = edge - local_edge;
    uint local_flat = local_edge * TILE_OUT + local_out;

    for (uint load = local_flat; load < TILE_EDGES * HIDDEN; load += TILE_EDGES * TILE_OUT) {
        uint tile_edge = load / HIDDEN;
        uint hidden = load - tile_edge * HIDDEN;
        uint global_edge = edge_base + tile_edge;
        input_tile[tile_edge][hidden] =
            (global_edge < params.num_edges)
                ? input[global_edge * HIDDEN + hidden]
                : 0.0f;
    }
    for (uint load = local_flat; load < TILE_OUT * HIDDEN; load += TILE_EDGES * TILE_OUT) {
        uint tile_out = load / HIDDEN;
        uint hidden = load - tile_out * HIDDEN;
        uint global_out = out_base + tile_out;
        weight_tile[tile_out][hidden] =
            (global_out < params.out_dim)
                ? weight[hidden * params.out_dim + global_out]
                : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (edge < params.num_edges && out < params.out_dim) {
        float acc = 0.0f;
        for (uint hidden = 0; hidden < HIDDEN; hidden++) {
            acc += input_tile[local_edge][hidden] * weight_tile[local_out][hidden];
        }
        output[edge * params.out_dim + out] = acc * params.inv_norm;
    }
}

kernel void mp0b_layer1_hidden_vjp_density_f32(
    device const float* density [[buffer(0)]],
    device const float* linear_out [[buffer(1)]],
    device const float* bar_out [[buffer(2)]],
    device float* bar_density [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint node [[thread_position_in_grid]]
) {
    if (node >= params.num_nodes) {
        return;
    }
    float density_norm = density[node] + 1.0f;
    uint base = node * 2048;
    float acc = 0.0f;
    for (uint index = 0; index < 2048; index++) {
        acc -= bar_out[base + index] * linear_out[base + index] / density_norm;
    }
    bar_density[node] = acc;
}

inline uint mp0b_layer1_hidden_to_feature_major_index(uint hidden_index) {
    if (hidden_index < 128) {
        return hidden_index * 16;
    }
    if (hidden_index < 512) {
        uint relative = hidden_index - 128;
        uint feature = relative / 3;
        uint component = relative - feature * 3;
        return feature * 16 + 1 + component;
    }
    if (hidden_index < 1152) {
        uint relative = hidden_index - 512;
        uint feature = relative / 5;
        uint component = relative - feature * 5;
        return feature * 16 + 4 + component;
    }
    uint relative = hidden_index - 1152;
    uint feature = relative / 7;
    uint component = relative - feature * 7;
    return feature * 16 + 9 + component;
}

kernel void mp0b_layer1_hidden_vjp_density_feature_major_f32(
    device const float* density [[buffer(0)]],
    device const float* linear_out [[buffer(1)]],
    device const float* bar_out_feature_major [[buffer(2)]],
    device float* bar_density [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint node [[thread_position_in_grid]]
) {
    if (node >= params.num_nodes) {
        return;
    }
    float density_norm = density[node] + 1.0f;
    uint base = node * 2048;
    float acc = 0.0f;
    for (uint hidden_index = 0; hidden_index < 2048; hidden_index++) {
        uint feature_major_index = mp0b_layer1_hidden_to_feature_major_index(hidden_index);
        acc -= bar_out_feature_major[base + feature_major_index]
            * linear_out[base + hidden_index]
            / density_norm;
    }
    bar_density[node] = acc;
}

kernel void mp0b_layer1_hidden_vjp_input_f32(
    device const float* density [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device const float* bar_out [[buffer(3)]],
    device float* bar_input [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 5120;
    if (gid >= total) {
        return;
    }
    uint node = gid / 5120;
    uint lane = gid - node * 5120;
    uint block = 0;
    uint input_offset = 0;
    uint output_offset = 0;
    uint component_dim = 1;
    if (lane < 128) {
        block = 0; input_offset = 0; output_offset = 0; component_dim = 1;
    } else if (lane < 256) {
        block = 1; input_offset = 128; output_offset = 0; component_dim = 1;
    } else if (lane < 640) {
        block = 2; input_offset = 256; output_offset = 128; component_dim = 3;
    } else if (lane < 1024) {
        block = 3; input_offset = 640; output_offset = 128; component_dim = 3;
    } else if (lane < 1408) {
        block = 4; input_offset = 1024; output_offset = 128; component_dim = 3;
    } else if (lane < 2048) {
        block = 5; input_offset = 1408; output_offset = 512; component_dim = 5;
    } else if (lane < 2688) {
        block = 6; input_offset = 2048; output_offset = 512; component_dim = 5;
    } else if (lane < 3328) {
        block = 7; input_offset = 2688; output_offset = 512; component_dim = 5;
    } else if (lane < 4224) {
        block = 8; input_offset = 3328; output_offset = 1152; component_dim = 7;
    } else {
        block = 9; input_offset = 4224; output_offset = 1152; component_dim = 7;
    }
    uint relative = lane - input_offset;
    uint in_feature = relative / component_dim;
    uint component = relative - in_feature * component_dim;
    float density_norm = density[node] + 1.0f;
    uint bar_base = node * 2048;
    uint weight_base = block * 16384 + in_feature * 128;
    float acc = 0.0f;
    for (uint out_feature = 0; out_feature < 128; out_feature++) {
        uint out_index = output_offset + out_feature * component_dim + component;
        acc += bar_out[bar_base + out_index]
            * output_mask[out_index]
            * weight[weight_base + out_feature]
            / density_norm;
    }
    bar_input[gid] = acc;
}

kernel void mp0b_layer1_hidden_vjp_input_feature_major_f32(
    device const float* density [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device const float* bar_out_feature_major [[buffer(3)]],
    device float* bar_input [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 5120;
    if (gid >= total) {
        return;
    }
    uint node = gid / 5120;
    uint lane = gid - node * 5120;
    uint block = 0;
    uint input_offset = 0;
    uint output_offset = 0;
    uint component_offset = 0;
    uint component_dim = 1;
    if (lane < 128) {
        block = 0; input_offset = 0; output_offset = 0; component_offset = 0; component_dim = 1;
    } else if (lane < 256) {
        block = 1; input_offset = 128; output_offset = 0; component_offset = 0; component_dim = 1;
    } else if (lane < 640) {
        block = 2; input_offset = 256; output_offset = 128; component_offset = 1; component_dim = 3;
    } else if (lane < 1024) {
        block = 3; input_offset = 640; output_offset = 128; component_offset = 1; component_dim = 3;
    } else if (lane < 1408) {
        block = 4; input_offset = 1024; output_offset = 128; component_offset = 1; component_dim = 3;
    } else if (lane < 2048) {
        block = 5; input_offset = 1408; output_offset = 512; component_offset = 4; component_dim = 5;
    } else if (lane < 2688) {
        block = 6; input_offset = 2048; output_offset = 512; component_offset = 4; component_dim = 5;
    } else if (lane < 3328) {
        block = 7; input_offset = 2688; output_offset = 512; component_offset = 4; component_dim = 5;
    } else if (lane < 4224) {
        block = 8; input_offset = 3328; output_offset = 1152; component_offset = 9; component_dim = 7;
    } else {
        block = 9; input_offset = 4224; output_offset = 1152; component_offset = 9; component_dim = 7;
    }
    uint relative = lane - input_offset;
    uint in_feature = relative / component_dim;
    uint component = relative - in_feature * component_dim;
    float density_norm = density[node] + 1.0f;
    uint bar_base = node * 2048;
    uint weight_base = block * 16384 + in_feature * 128;
    float acc = 0.0f;
    for (uint out_feature = 0; out_feature < 128; out_feature++) {
        uint hidden_out_index = output_offset + out_feature * component_dim + component;
        uint feature_major_out_index = out_feature * 16 + component_offset + component;
        acc += bar_out_feature_major[bar_base + feature_major_out_index]
            * output_mask[hidden_out_index]
            * weight[weight_base + out_feature]
            / density_norm;
    }
    bar_input[gid] = acc;
}

kernel void mp0b_layer0_hidden_vjp_input_f32(
    device const float* density [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device const float* bar_out [[buffer(3)]],
    device float* bar_input [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 2048;
    if (gid >= total) {
        return;
    }
    uint node = gid / 2048;
    uint lane = gid - node * 2048;
    uint block = 0;
    uint offset = 0;
    uint component_dim = 1;
    if (lane < 128) {
        block = 0; offset = 0; component_dim = 1;
    } else if (lane < 512) {
        block = 1; offset = 128; component_dim = 3;
    } else if (lane < 1152) {
        block = 2; offset = 512; component_dim = 5;
    } else {
        block = 3; offset = 1152; component_dim = 7;
    }
    uint relative = lane - offset;
    uint in_feature = relative / component_dim;
    uint component = relative - in_feature * component_dim;
    float density_norm = density[node] + 1.0f;
    uint base = node * 2048;
    uint weight_base = block * 16384 + in_feature * 128;
    const float path_weight = 0.08838834764831845f;
    float acc = 0.0f;
    for (uint out_feature = 0; out_feature < 128; out_feature++) {
        uint out_index = offset + out_feature * component_dim + component;
        acc += bar_out[base + out_index]
            * output_mask[out_index]
            * weight[weight_base + out_feature]
            * path_weight
            / density_norm;
    }
    bar_input[gid] = acc;
}

kernel void mp0b_hidden_message_selector_vjp_input_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device const float* bar_out_feature_major [[buffer(3)]],
    device float* bar_input [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 2048;
    if (gid >= total) {
        return;
    }
    uint node = gid / 2048;
    uint lane = gid - node * 2048;
    uint block = 0;
    uint block_offset = 0;
    uint component_offset = 0;
    uint component_dim = 1;
    float path_weight = 0.009369146112448268f;
    if (lane < 128) {
        block = 0; block_offset = 0; component_offset = 0; component_dim = 1;
        path_weight = 0.009369146112448268f;
    } else if (lane < 512) {
        block = 1; block_offset = 128; component_offset = 1; component_dim = 3;
        path_weight = 0.016227837f;
    } else if (lane < 1152) {
        block = 2; block_offset = 512; component_offset = 4; component_dim = 5;
        path_weight = 0.020950048f;
    } else {
        block = 3; block_offset = 1152; component_offset = 9; component_dim = 7;
        path_weight = 0.02478843f;
    }
    uint relative = lane - block_offset;
    uint in_feature = relative / component_dim;
    uint component = relative - in_feature * component_dim;
    uint type_id = type_ids[node];
    uint weight_base = block * 1458176 + (in_feature * 89 + type_id) * 128;
    uint bar_base = node * 2048;
    float block_scale = path_weight * rsqrt(float(component_dim));
    float acc = 0.0f;
    for (uint out_feature = 0; out_feature < 128; out_feature++) {
        uint hidden_index = block_offset + out_feature * component_dim + component;
        uint feature_major_index = out_feature * 16 + component_offset + component;
        acc += bar_out_feature_major[bar_base + feature_major_index]
            * block_scale
            * output_mask[hidden_index]
            * weight[weight_base + out_feature];
    }
    bar_input[gid] = acc;
}

kernel void mp0b_hidden_message_selector_to_feature_major_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* output_mask [[buffer(3)]],
    device float* output_feature_major [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    constexpr uint NUM_FEATURES = 128;
    constexpr uint NUM_ELEMENTS = 89;
    constexpr uint FEATURE_ROW_LEN = 16;
    constexpr uint HIDDEN_ROW_LEN = 2048;
    constexpr uint BLOCK_WEIGHTS = NUM_FEATURES * NUM_ELEMENTS * NUM_FEATURES;
    uint total = params.num_nodes * HIDDEN_ROW_LEN;
    if (gid >= total) {
        return;
    }
    uint node = gid / HIDDEN_ROW_LEN;
    uint lane = gid - node * HIDDEN_ROW_LEN;
    uint out_feature = lane / FEATURE_ROW_LEN;
    uint component_index = lane - out_feature * FEATURE_ROW_LEN;

    uint block = 0;
    uint block_offset = 0;
    uint component_offset = 0;
    uint component_dim = 1;
    float path_weight = 0.009369146112448268f;
    if (component_index < 1) {
        block = 0; block_offset = 0; component_offset = 0; component_dim = 1;
        path_weight = 0.009369146112448268f;
    } else if (component_index < 4) {
        block = 1; block_offset = 128; component_offset = 1; component_dim = 3;
        path_weight = 0.016227837f;
    } else if (component_index < 9) {
        block = 2; block_offset = 512; component_offset = 4; component_dim = 5;
        path_weight = 0.020950048f;
    } else {
        block = 3; block_offset = 1152; component_offset = 9; component_dim = 7;
        path_weight = 0.02478843f;
    }

    uint component = component_index - component_offset;
    uint type_id = type_ids[node];
    uint input_base = node * HIDDEN_ROW_LEN;
    uint weight_base = block * BLOCK_WEIGHTS + (type_id * NUM_FEATURES) + out_feature;
    float acc = 0.0f;
    for (uint in_feature = 0; in_feature < NUM_FEATURES; in_feature++) {
        uint input_index = input_base + block_offset + in_feature * component_dim + component;
        uint weight_index = weight_base + in_feature * NUM_ELEMENTS * NUM_FEATURES;
        acc += input[input_index] * weight[weight_index];
    }
    uint hidden_index = block_offset + out_feature * component_dim + component;
    float block_scale = path_weight * rsqrt(float(component_dim));
    output_feature_major[gid] = acc * block_scale * output_mask[hidden_index];
}

kernel void mp0b_scalar_vector_product_linear_forward_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 512;
    if (gid >= total) {
        return;
    }
    uint node = gid / 512;
    uint lane = gid - node * 512;
    uint row_base = node * 512;
    float acc = 0.0f;
    if (lane < 128) {
        uint out_feature = lane;
        for (uint in_feature = 0; in_feature < 128; in_feature++) {
            acc += input[row_base + in_feature] * weight[in_feature * 128 + out_feature];
        }
    } else {
        uint relative = lane - 128;
        uint out_feature = relative / 3;
        uint component = relative - out_feature * 3;
        uint weight_base = 16384 + out_feature;
        for (uint in_feature = 0; in_feature < 128; in_feature++) {
            uint in_index = row_base + 128 + in_feature * 3 + component;
            acc += input[in_index] * weight[weight_base + in_feature * 128];
        }
    }
    output[gid] = acc * output_mask[lane];
}

kernel void mp0b_scalar_vector_product_linear_vjp_input_f32(
    device const float* weight [[buffer(0)]],
    device const float* output_mask [[buffer(1)]],
    device const float* bar_out [[buffer(2)]],
    device float* bar_input [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 512;
    if (gid >= total) {
        return;
    }
    uint node = gid / 512;
    uint lane = gid - node * 512;
    uint row_base = node * 512;
    float acc = 0.0f;
    if (lane < 128) {
        uint in_feature = lane;
        uint weight_base = in_feature * 128;
        for (uint out_feature = 0; out_feature < 128; out_feature++) {
            acc += bar_out[row_base + out_feature]
                * output_mask[out_feature]
                * weight[weight_base + out_feature];
        }
    } else {
        uint relative = lane - 128;
        uint in_feature = relative / 3;
        uint component = relative - in_feature * 3;
        uint weight_base = 16384 + in_feature * 128;
        for (uint out_feature = 0; out_feature < 128; out_feature++) {
            uint out_index = row_base + 128 + out_feature * 3 + component;
            uint mask_index = 128 + out_feature * 3 + component;
            acc += bar_out[out_index] * output_mask[mask_index] * weight[weight_base + out_feature];
        }
    }
    bar_input[gid] += acc;
}

kernel void mp0b_scalar_vector_product_linear_forward_packed_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* output_mask [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 128;
    if (gid >= total) {
        return;
    }
    uint node = gid / 128;
    uint out_feature = gid - node * 128;
    uint row_base = node * 512;

    float scalar_acc = 0.0f;
    float vector_acc0 = 0.0f;
    float vector_acc1 = 0.0f;
    float vector_acc2 = 0.0f;
    for (uint in_feature = 0; in_feature < 128; in_feature++) {
        scalar_acc += input[row_base + in_feature] * weight[in_feature * 128 + out_feature];
        float vector_weight = weight[16384 + in_feature * 128 + out_feature];
        uint input_vector_base = row_base + 128 + in_feature * 3;
        vector_acc0 += input[input_vector_base] * vector_weight;
        vector_acc1 += input[input_vector_base + 1] * vector_weight;
        vector_acc2 += input[input_vector_base + 2] * vector_weight;
    }

    output[row_base + out_feature] = scalar_acc * output_mask[out_feature];
    uint output_vector_base = row_base + 128 + out_feature * 3;
    uint mask_vector_base = 128 + out_feature * 3;
    output[output_vector_base] = vector_acc0 * output_mask[mask_vector_base];
    output[output_vector_base + 1] = vector_acc1 * output_mask[mask_vector_base + 1];
    output[output_vector_base + 2] = vector_acc2 * output_mask[mask_vector_base + 2];
}

kernel void mp0b_scalar_vector_product_linear_vjp_input_packed_f32(
    device const float* weight [[buffer(0)]],
    device const float* output_mask [[buffer(1)]],
    device const float* bar_out [[buffer(2)]],
    device float* bar_input [[buffer(3)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 128;
    if (gid >= total) {
        return;
    }
    uint node = gid / 128;
    uint in_feature = gid - node * 128;
    uint row_base = node * 512;

    float scalar_acc = 0.0f;
    float vector_acc0 = 0.0f;
    float vector_acc1 = 0.0f;
    float vector_acc2 = 0.0f;
    uint scalar_weight_base = in_feature * 128;
    uint vector_weight_base = 16384 + in_feature * 128;
    for (uint out_feature = 0; out_feature < 128; out_feature++) {
        scalar_acc += bar_out[row_base + out_feature]
            * output_mask[out_feature]
            * weight[scalar_weight_base + out_feature];
        float vector_weight = weight[vector_weight_base + out_feature];
        uint out_vector_base = row_base + 128 + out_feature * 3;
        uint mask_vector_base = 128 + out_feature * 3;
        vector_acc0 += bar_out[out_vector_base]
            * output_mask[mask_vector_base]
            * vector_weight;
        vector_acc1 += bar_out[out_vector_base + 1]
            * output_mask[mask_vector_base + 1]
            * vector_weight;
        vector_acc2 += bar_out[out_vector_base + 2]
            * output_mask[mask_vector_base + 2]
            * vector_weight;
    }

    bar_input[row_base + in_feature] += scalar_acc;
    uint input_vector_base = row_base + 128 + in_feature * 3;
    bar_input[input_vector_base] += vector_acc0;
    bar_input[input_vector_base + 1] += vector_acc1;
    bar_input[input_vector_base + 2] += vector_acc2;
}

// OAM layer1 currently uses the same 512/16/1280/5120 layout and math as MP0b.
// Keep this body in lockstep with oam_layer1_conv_tp_forward_f32; the separate
// entry points are for pipeline selection and diagnostics, not divergent math.
kernel void mp0b_layer1_conv_tp_forward_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* tp_weights [[buffer(2)]],
    device const uint* receiver_offsets [[buffer(3)]],
    device const uint* receiver_edges [[buffer(4)]],
    device const uint* senders [[buffer(5)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(6)]],
    device const uint* cg_offsets [[buffer(7)]],
    device const Mp0bLayer1Cg* cgs [[buffer(8)]],
    device float* scatter_message [[buffer(9)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 5120;
    if (gid >= total) {
        return;
    }
    uint receiver = gid / 5120;
    uint lane = gid - receiver * 5120;
    uint template_index = 0;
    uint feature = 0;
    uint output_component = 0;
    bool found = false;
    for (uint index = 0; index < 10; index++) {
        Mp0bLayer1TpTemplate tpl = templates[index];
        uint begin = tpl.out_base;
        uint end = tpl.out_base + 128 * tpl.out_dim;
        if (lane >= begin && lane < end) {
            uint relative = lane - begin;
            feature = relative / tpl.out_dim;
            output_component = relative - feature * tpl.out_dim;
            template_index = index;
            found = true;
            break;
        }
    }
    if (!found) {
        scatter_message[gid] = 0.0f;
        return;
    }

    Mp0bLayer1TpTemplate tpl = templates[template_index];
    uint grouped_cg_offset = template_index * 8 + output_component;
    uint cg_begin = cg_offsets[grouped_cg_offset];
    uint cg_end = cg_offsets[grouped_cg_offset + 1];
    float acc = 0.0f;
    for (uint edge_slot = receiver_offsets[receiver]; edge_slot < receiver_offsets[receiver + 1]; edge_slot++) {
        uint edge = receiver_edges[edge_slot];
        uint sender = senders[edge];
        uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
        uint y_base = edge * 16 + tpl.y_base;
        uint weight_index = edge * 1280 + tpl.weight_base + feature;
        float weight = tp_weights[weight_index];
        for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
            Mp0bLayer1Cg cg = cgs[cg_index];
            acc += weight
                * cg.scale
                * linear_up[x_base + cg.i_m]
                * edge_attrs[y_base + cg.j_m];
        }
    }
    scatter_message[gid] = acc;
}

// MACE-OAM Medium currently shares the pinned MP0b layer-1 scalar+vector
// conv-TP layout. Keep a distinct symbol so future OAM-specific artifacts can
// diverge only after adding a fixture-pinned kernel update.
kernel void oam_layer1_conv_tp_forward_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* tp_weights [[buffer(2)]],
    device const uint* receiver_offsets [[buffer(3)]],
    device const uint* receiver_edges [[buffer(4)]],
    device const uint* senders [[buffer(5)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(6)]],
    device const uint* cg_offsets [[buffer(7)]],
    device const Mp0bLayer1Cg* cgs [[buffer(8)]],
    device float* scatter_message [[buffer(9)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 5120;
    if (gid >= total) {
        return;
    }
    uint receiver = gid / 5120;
    uint lane = gid - receiver * 5120;
    uint template_index = 0;
    uint feature = 0;
    uint output_component = 0;
    bool found = false;
    for (uint index = 0; index < 10; index++) {
        Mp0bLayer1TpTemplate tpl = templates[index];
        uint begin = tpl.out_base;
        uint end = tpl.out_base + 128 * tpl.out_dim;
        if (lane >= begin && lane < end) {
            uint relative = lane - begin;
            feature = relative / tpl.out_dim;
            output_component = relative - feature * tpl.out_dim;
            template_index = index;
            found = true;
            break;
        }
    }
    if (!found) {
        scatter_message[gid] = 0.0f;
        return;
    }

    Mp0bLayer1TpTemplate tpl = templates[template_index];
    uint grouped_cg_offset = template_index * 8 + output_component;
    uint cg_begin = cg_offsets[grouped_cg_offset];
    uint cg_end = cg_offsets[grouped_cg_offset + 1];
    float acc = 0.0f;
    for (uint edge_slot = receiver_offsets[receiver]; edge_slot < receiver_offsets[receiver + 1]; edge_slot++) {
        uint edge = receiver_edges[edge_slot];
        uint sender = senders[edge];
        uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
        uint y_base = edge * 16 + tpl.y_base;
        uint weight_index = edge * 1280 + tpl.weight_base + feature;
        float weight = tp_weights[weight_index];
        for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
            Mp0bLayer1Cg cg = cgs[cg_index];
            acc += weight
                * cg.scale
                * linear_up[x_base + cg.i_m]
                * edge_attrs[y_base + cg.j_m];
        }
    }
    scatter_message[gid] = acc;
}

kernel void mp0b_layer1_conv_tp_vjp_linear_up_f32(
    device const float* edge_attrs [[buffer(0)]],
    device const float* tp_weights [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* sender_offsets [[buffer(3)]],
    device const uint* sender_edges [[buffer(4)]],
    device const uint* receivers [[buffer(5)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(6)]],
    device const uint* cg_offsets [[buffer(7)]],
    device const Mp0bLayer1Cg* cgs [[buffer(8)]],
    device const uint* input_lane_offsets [[buffer(9)]],
    device const Mp0bLayer1InputLane* input_lane_entries [[buffer(10)]],
    device float* bar_linear_up [[buffer(11)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 512;
    if (gid >= total) {
        return;
    }
    uint sender = gid / 512;
    uint lane = gid - sender * 512;
    float acc = 0.0f;
    uint entry_begin = input_lane_offsets[lane];
    uint entry_end = input_lane_offsets[lane + 1];
    for (uint entry_index = entry_begin; entry_index < entry_end; entry_index++) {
        Mp0bLayer1InputLane lane_info = input_lane_entries[entry_index];
        uint template_index = lane_info.template_index;
        uint feature = lane_info.feature;
        uint input_component = lane_info.input_component;
        Mp0bLayer1TpTemplate tpl = templates[template_index];
        uint grouped_cg_offset = template_index * 8 + input_component;
        uint cg_begin = cg_offsets[grouped_cg_offset];
        uint cg_end = cg_offsets[grouped_cg_offset + 1];
        for (uint edge_slot = sender_offsets[sender]; edge_slot < sender_offsets[sender + 1]; edge_slot++) {
            uint edge = sender_edges[edge_slot];
            uint receiver = receivers[edge];
            float weight = tp_weights[edge * 1280 + tpl.weight_base + feature];
            uint y_base = edge * 16 + tpl.y_base;
            uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
            for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                Mp0bLayer1Cg cg = cgs[cg_index];
                acc += cg.scale
                    * weight
                    * edge_attrs[y_base + cg.j_m]
                    * bar_scatter_message[out_base + cg.k_m];
            }
        }
    }
    bar_linear_up[gid] = acc;
}

kernel void mp0b_layer1_conv_tp_vjp_edge_attrs_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* tp_weights [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* senders [[buffer(3)]],
    device const uint* receivers [[buffer(4)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(5)]],
    device const uint* cg_offsets [[buffer(6)]],
    device const Mp0bLayer1Cg* cgs [[buffer(7)]],
    device float* bar_edge_attrs [[buffer(8)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 16;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 16;
    uint y_component = gid - edge * 16;
    uint sender = senders[edge];
    uint receiver = receivers[edge];
    float acc = 0.0f;
    for (uint template_index = 0; template_index < 10; template_index++) {
        Mp0bLayer1TpTemplate tpl = templates[template_index];
        uint cg_begin = cg_offsets[template_index];
        uint cg_end = cg_offsets[template_index + 1];
        for (uint feature = 0; feature < 128; feature++) {
            float weight = tp_weights[edge * 1280 + tpl.weight_base + feature];
            uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
            uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
            for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                Mp0bLayer1Cg cg = cgs[cg_index];
                if (tpl.y_base + cg.j_m == y_component) {
                    acc += cg.scale
                        * weight
                        * linear_up[x_base + cg.i_m]
                        * bar_scatter_message[out_base + cg.k_m];
                }
            }
        }
    }
    bar_edge_attrs[gid] = acc;
}

kernel void mp0b_layer1_conv_tp_vjp_edge_attrs_by_edge_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* tp_weights [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* senders [[buffer(3)]],
    device const uint* receivers [[buffer(4)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(5)]],
    device const uint* cg_offsets [[buffer(6)]],
    device const Mp0bLayer1Cg* cgs [[buffer(7)]],
    device float* bar_edge_attrs [[buffer(8)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(9)]],
    uint edge [[thread_position_in_grid]]
) {
    if (edge >= params.num_edges) {
        return;
    }
    uint sender = senders[edge];
    uint receiver = receivers[edge];
    float acc[16];
    for (uint component = 0; component < 16; component++) {
        acc[component] = 0.0f;
    }
    for (uint template_index = 0; template_index < 10; template_index++) {
        Mp0bLayer1TpTemplate tpl = templates[template_index];
        uint cg_begin = cg_offsets[template_index];
        uint cg_end = cg_offsets[template_index + 1];
        for (uint feature = 0; feature < 128; feature++) {
            float weight = tp_weights[edge * 1280 + tpl.weight_base + feature];
            uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
            uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
            for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                Mp0bLayer1Cg cg = cgs[cg_index];
                uint y_component = tpl.y_base + cg.j_m;
                acc[y_component] += cg.scale
                    * weight
                    * linear_up[x_base + cg.i_m]
                    * bar_scatter_message[out_base + cg.k_m];
            }
        }
    }
    uint edge_base = edge * 16;
    for (uint component = 0; component < 16; component++) {
        bar_edge_attrs[edge_base + component] = acc[component];
    }
}

kernel void mp0b_layer1_conv_tp_vjp_edge_attrs_feature_parallel_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* tp_weights [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* senders [[buffer(3)]],
    device const uint* receivers [[buffer(4)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(5)]],
    device const uint* cg_offsets [[buffer(6)]],
    device const Mp0bLayer1Cg* cgs [[buffer(7)]],
    device float* bar_edge_attrs [[buffer(8)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint NUM_FEATURES = 128;
    constexpr uint EDGE_ATTR_ROW_LEN = 16;
    threadgroup float partial[NUM_FEATURES][EDGE_ATTR_ROW_LEN];

    uint edge = gid.y;
    uint feature = tid.x;
    float acc[EDGE_ATTR_ROW_LEN];
    for (uint component = 0; component < EDGE_ATTR_ROW_LEN; component++) {
        acc[component] = 0.0f;
    }
    if (edge < params.num_edges && feature < NUM_FEATURES) {
        uint sender = senders[edge];
        uint receiver = receivers[edge];
        for (uint template_index = 0; template_index < 10; template_index++) {
            Mp0bLayer1TpTemplate tpl = templates[template_index];
            uint cg_begin = cg_offsets[template_index];
            uint cg_end = cg_offsets[template_index + 1];
            float weight = tp_weights[edge * 1280 + tpl.weight_base + feature];
            uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
            uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
            for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                Mp0bLayer1Cg cg = cgs[cg_index];
                uint y_component = tpl.y_base + cg.j_m;
                acc[y_component] += cg.scale
                    * weight
                    * linear_up[x_base + cg.i_m]
                    * bar_scatter_message[out_base + cg.k_m];
            }
        }
    }
    for (uint component = 0; component < EDGE_ATTR_ROW_LEN; component++) {
        partial[feature][component] = acc[component];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (edge < params.num_edges && feature < EDGE_ATTR_ROW_LEN) {
        float sum = 0.0f;
        for (uint input_feature = 0; input_feature < NUM_FEATURES; input_feature++) {
            sum += partial[input_feature][feature];
        }
        bar_edge_attrs[edge * EDGE_ATTR_ROW_LEN + feature] = sum;
    }
}

kernel void mp0b_layer1_conv_tp_vjp_hidden2_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* layer3_weight [[buffer(2)]],
    device const float* bar_scatter_message [[buffer(3)]],
    device const uint* senders [[buffer(4)]],
    device const uint* receivers [[buffer(5)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(6)]],
    device const uint* cg_offsets [[buffer(7)]],
    device const Mp0bLayer1Cg* cgs [[buffer(8)]],
    device float* bar_hidden2 [[buffer(9)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 64;
    uint hidden = gid - edge * 64;
    uint sender = senders[edge];
    uint receiver = receivers[edge];
    float acc = 0.0f;
    for (uint template_index = 0; template_index < 10; template_index++) {
        Mp0bLayer1TpTemplate tpl = templates[template_index];
        uint cg_begin = cg_offsets[template_index];
        uint cg_end = cg_offsets[template_index + 1];
        for (uint feature = 0; feature < 128; feature++) {
            uint weight_index = tpl.weight_base + feature;
            uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
            uint y_base = edge * 16 + tpl.y_base;
            uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
            float bar_weight = 0.0f;
            for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                Mp0bLayer1Cg cg = cgs[cg_index];
                bar_weight += cg.scale
                    * linear_up[x_base + cg.i_m]
                    * edge_attrs[y_base + cg.j_m]
                    * bar_scatter_message[out_base + cg.k_m];
            }
            acc += layer3_weight[hidden * 1280 + weight_index] * bar_weight;
        }
    }
    bar_hidden2[gid] = acc * 0.125f;
}

kernel void mp0b_layer1_conv_tp_vjp_bar_tp_weights_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* senders [[buffer(3)]],
    device const uint* receivers [[buffer(4)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(5)]],
    device const uint* cg_offsets [[buffer(6)]],
    device const Mp0bLayer1Cg* cgs [[buffer(7)]],
    device const Mp0bLayer1WeightLane* weight_lanes [[buffer(8)]],
    device float* bar_tp_weights [[buffer(9)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 1280;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 1280;
    uint lane = gid - edge * 1280;
    uint sender = senders[edge];
    uint receiver = receivers[edge];
    Mp0bLayer1WeightLane lane_info = weight_lanes[lane];
    uint template_index = lane_info.template_index;
    uint feature = lane_info.feature;
    Mp0bLayer1TpTemplate tpl = templates[template_index];
    uint cg_begin = cg_offsets[template_index];
    uint cg_end = cg_offsets[template_index + 1];
    uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
    uint y_base = edge * 16 + tpl.y_base;
    uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
    float acc = 0.0f;
    for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
        Mp0bLayer1Cg cg = cgs[cg_index];
        acc += cg.scale
            * linear_up[x_base + cg.i_m]
            * edge_attrs[y_base + cg.j_m]
            * bar_scatter_message[out_base + cg.k_m];
    }
    bar_tp_weights[gid] = acc;
}

kernel void mp0b_layer1_conv_tp_vjp_hidden2_from_weights_f32(
    device const float* layer3_weight [[buffer(0)]],
    device const float* bar_tp_weights [[buffer(1)]],
    device float* bar_hidden2 [[buffer(2)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_edges * 64;
    if (gid >= total) {
        return;
    }
    uint edge = gid / 64;
    uint hidden = gid - edge * 64;
    uint bar_weight_base = edge * 1280;
    uint layer_weight_base = hidden * 1280;
    float acc = 0.0f;
    for (uint weight_index = 0; weight_index < 1280; weight_index++) {
        acc += layer3_weight[layer_weight_base + weight_index]
            * bar_tp_weights[bar_weight_base + weight_index];
    }
    bar_hidden2[gid] = acc * 0.125f;
}

kernel void mp0b_layer1_conv_tp_vjp_hidden2_from_weights_tiled_f32(
    device const float* layer3_weight [[buffer(0)]],
    device const float* bar_tp_weights [[buffer(1)]],
    device float* bar_hidden2 [[buffer(2)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_EDGES = 8;
    constexpr uint TILE_HIDDEN = 8;
    constexpr uint TILE_K = 32;
    threadgroup float bar_tile[TILE_EDGES][TILE_K];
    threadgroup float weight_tile[TILE_HIDDEN][TILE_K];

    uint hidden = gid.x;
    uint edge = gid.y;
    uint local_hidden = tid.x;
    uint local_edge = tid.y;
    uint edge_base = edge - local_edge;
    uint hidden_base = hidden - local_hidden;
    uint local_flat = local_edge * TILE_HIDDEN + local_hidden;
    float acc = 0.0f;

    for (uint k0 = 0; k0 < 1280; k0 += TILE_K) {
        for (uint load = local_flat; load < TILE_EDGES * TILE_K; load += TILE_EDGES * TILE_HIDDEN) {
            uint tile_edge = load / TILE_K;
            uint tile_k = load - tile_edge * TILE_K;
            uint global_edge = edge_base + tile_edge;
            bar_tile[tile_edge][tile_k] =
                (global_edge < params.num_edges)
                    ? bar_tp_weights[global_edge * 1280 + k0 + tile_k]
                    : 0.0f;
        }
        for (uint load = local_flat; load < TILE_HIDDEN * TILE_K; load += TILE_EDGES * TILE_HIDDEN) {
            uint tile_hidden = load / TILE_K;
            uint tile_k = load - tile_hidden * TILE_K;
            uint global_hidden = hidden_base + tile_hidden;
            weight_tile[tile_hidden][tile_k] =
                (global_hidden < 64)
                    ? layer3_weight[global_hidden * 1280 + k0 + tile_k]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (edge < params.num_edges && hidden < 64) {
            for (uint tile_k = 0; tile_k < TILE_K; tile_k++) {
                acc += bar_tile[local_edge][tile_k] * weight_tile[local_hidden][tile_k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (edge < params.num_edges && hidden < 64) {
        bar_hidden2[edge * 64 + hidden] = acc * 0.125f;
    }
}

kernel void mp0b_layer1_conv_tp_vjp_hidden2_fused_tiled_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* layer3_weight [[buffer(2)]],
    device const float* bar_scatter_message [[buffer(3)]],
    device const uint* senders [[buffer(4)]],
    device const uint* receivers [[buffer(5)]],
    device const Mp0bLayer1TpTemplate* templates [[buffer(6)]],
    device const uint* cg_offsets [[buffer(7)]],
    device const Mp0bLayer1Cg* cgs [[buffer(8)]],
    device const Mp0bLayer1WeightLane* weight_lanes [[buffer(9)]],
    device float* bar_hidden2 [[buffer(10)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_EDGES = 4;
    constexpr uint TILE_HIDDEN = 64;
    constexpr uint TILE_K = 32;
    threadgroup float bar_tile[TILE_EDGES][TILE_K];

    uint hidden = gid.x;
    uint edge = gid.y;
    uint local_hidden = tid.x;
    uint local_edge = tid.y;
    uint edge_base = edge - local_edge;
    uint local_flat = local_edge * TILE_HIDDEN + local_hidden;
    float acc = 0.0f;

    for (uint k0 = 0; k0 < 1280; k0 += TILE_K) {
        if (local_flat < TILE_EDGES * TILE_K) {
            uint tile_edge = local_flat / TILE_K;
            uint tile_k = local_flat - tile_edge * TILE_K;
            uint global_edge = edge_base + tile_edge;
            float bar_weight = 0.0f;
            if (global_edge < params.num_edges) {
                uint lane = k0 + tile_k;
                uint sender = senders[global_edge];
                uint receiver = receivers[global_edge];
                Mp0bLayer1WeightLane lane_info = weight_lanes[lane];
                uint template_index = lane_info.template_index;
                uint feature = lane_info.feature;
                Mp0bLayer1TpTemplate tpl = templates[template_index];
                uint cg_begin = cg_offsets[template_index];
                uint cg_end = cg_offsets[template_index + 1];
                uint x_base = sender * 512 + tpl.x_base + feature * tpl.x_dim;
                uint y_base = global_edge * 16 + tpl.y_base;
                uint out_base = receiver * 5120 + tpl.out_base + feature * tpl.out_dim;
                for (uint cg_index = cg_begin; cg_index < cg_end; cg_index++) {
                    Mp0bLayer1Cg cg = cgs[cg_index];
                    bar_weight += cg.scale
                        * linear_up[x_base + cg.i_m]
                        * edge_attrs[y_base + cg.j_m]
                        * bar_scatter_message[out_base + cg.k_m];
                }
            }
            bar_tile[tile_edge][tile_k] = bar_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (edge < params.num_edges && hidden < 64) {
            for (uint tile_k = 0; tile_k < TILE_K; tile_k++) {
                acc += bar_tile[local_edge][tile_k]
                    * layer3_weight[(k0 + tile_k) * 64 + hidden];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (edge < params.num_edges && hidden < 64) {
        bar_hidden2[edge * 64 + hidden] = acc * 0.125f;
    }
}

kernel void mp0b_layer0_conv_tp_vjp_edge_attrs_feature_parallel_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* tp_weights [[buffer(1)]],
    device const float* bar_scatter_message [[buffer(2)]],
    device const uint* senders [[buffer(3)]],
    device const uint* receivers [[buffer(4)]],
    device float* bar_edge_attrs [[buffer(5)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint NUM_FEATURES = 128;
    constexpr uint EDGE_ATTR_ROW_LEN = 16;
    constexpr uint OUTPUT_ROW_LEN = 2048;
    threadgroup float partial[NUM_FEATURES][EDGE_ATTR_ROW_LEN];

    uint edge = gid.y;
    uint feature = tid.x;
    float acc[EDGE_ATTR_ROW_LEN];
    for (uint component = 0; component < EDGE_ATTR_ROW_LEN; component++) {
        acc[component] = 0.0f;
    }
    if (edge < params.num_edges && feature < NUM_FEATURES) {
        uint sender = senders[edge];
        uint receiver = receivers[edge];
        float x = linear_up[sender * NUM_FEATURES + feature];
        uint weight_base = edge * (4 * NUM_FEATURES) + feature;
        uint bar_base = receiver * OUTPUT_ROW_LEN;

        float w0 = tp_weights[weight_base];
        acc[0] += w0 * x * bar_scatter_message[bar_base + feature];

        float w1 = tp_weights[weight_base + NUM_FEATURES];
        uint out_l1 = bar_base + NUM_FEATURES + feature * 3;
        acc[1] += w1 * x * bar_scatter_message[out_l1];
        acc[2] += w1 * x * bar_scatter_message[out_l1 + 1];
        acc[3] += w1 * x * bar_scatter_message[out_l1 + 2];

        float w2 = tp_weights[weight_base + 2 * NUM_FEATURES];
        uint out_l2 = bar_base + NUM_FEATURES + 3 * NUM_FEATURES + feature * 5;
        for (uint component = 0; component < 5; component++) {
            acc[4 + component] += w2 * x * bar_scatter_message[out_l2 + component];
        }

        float w3 = tp_weights[weight_base + 3 * NUM_FEATURES];
        uint out_l3 = bar_base + NUM_FEATURES + 3 * NUM_FEATURES + 5 * NUM_FEATURES + feature * 7;
        for (uint component = 0; component < 7; component++) {
            acc[9 + component] += w3 * x * bar_scatter_message[out_l3 + component];
        }
    }
    for (uint component = 0; component < EDGE_ATTR_ROW_LEN; component++) {
        partial[feature][component] = acc[component];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (edge < params.num_edges && feature < EDGE_ATTR_ROW_LEN) {
        float sum = 0.0f;
        for (uint input_feature = 0; input_feature < NUM_FEATURES; input_feature++) {
            sum += partial[input_feature][feature];
        }
        if ((params.flags & 1u) != 0u) {
            bar_edge_attrs[edge * EDGE_ATTR_ROW_LEN + feature] += sum;
        } else {
            bar_edge_attrs[edge * EDGE_ATTR_ROW_LEN + feature] = sum;
        }
    }
}

kernel void mp0b_layer0_conv_tp_vjp_hidden2_fused_tiled_f32(
    device const float* linear_up [[buffer(0)]],
    device const float* edge_attrs [[buffer(1)]],
    device const float* layer3_weight [[buffer(2)]],
    device const float* bar_scatter_message [[buffer(3)]],
    device const uint* senders [[buffer(4)]],
    device const uint* receivers [[buffer(5)]],
    device float* bar_hidden2 [[buffer(6)]],
    constant Mp0bLayer1ConvTpVjpParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_EDGES = 4;
    constexpr uint TILE_HIDDEN = 64;
    constexpr uint TILE_K = 64;
    constexpr uint NUM_FEATURES = 128;
    constexpr uint EDGE_ATTR_ROW_LEN = 16;
    constexpr uint OUTPUT_ROW_LEN = 2048;
    constexpr uint WEIGHT_ROW_LEN = 4 * NUM_FEATURES;
    threadgroup float bar_tile[TILE_EDGES][TILE_K];

    uint hidden = gid.x;
    uint edge = gid.y;
    uint local_hidden = tid.x;
    uint local_edge = tid.y;
    uint edge_base = edge - local_edge;
    uint local_flat = local_edge * TILE_HIDDEN + local_hidden;
    float acc = 0.0f;

    for (uint k0 = 0; k0 < WEIGHT_ROW_LEN; k0 += TILE_K) {
        if (local_flat < TILE_EDGES * TILE_K) {
            uint tile_edge = local_flat / TILE_K;
            uint tile_k = local_flat - tile_edge * TILE_K;
            uint global_edge = edge_base + tile_edge;
            float bar_weight = 0.0f;
            if (global_edge < params.num_edges) {
                uint lane = k0 + tile_k;
                uint block = lane / NUM_FEATURES;
                uint feature = lane - block * NUM_FEATURES;
                uint sender = senders[global_edge];
                uint receiver = receivers[global_edge];
                float x = linear_up[sender * NUM_FEATURES + feature];
                uint y_base = global_edge * EDGE_ATTR_ROW_LEN;
                uint bar_base = receiver * OUTPUT_ROW_LEN;
                float dot = 0.0f;
                if (block == 0) {
                    dot = edge_attrs[y_base] * bar_scatter_message[bar_base + feature];
                } else if (block == 1) {
                    uint out_l1 = bar_base + NUM_FEATURES + feature * 3;
                    dot = edge_attrs[y_base + 1] * bar_scatter_message[out_l1]
                        + edge_attrs[y_base + 2] * bar_scatter_message[out_l1 + 1]
                        + edge_attrs[y_base + 3] * bar_scatter_message[out_l1 + 2];
                } else if (block == 2) {
                    uint out_l2 = bar_base + NUM_FEATURES + 3 * NUM_FEATURES + feature * 5;
                    for (uint component = 0; component < 5; component++) {
                        dot += edge_attrs[y_base + 4 + component]
                            * bar_scatter_message[out_l2 + component];
                    }
                } else {
                    uint out_l3 = bar_base + NUM_FEATURES + 3 * NUM_FEATURES + 5 * NUM_FEATURES + feature * 7;
                    for (uint component = 0; component < 7; component++) {
                        dot += edge_attrs[y_base + 9 + component]
                            * bar_scatter_message[out_l3 + component];
                    }
                }
                bar_weight = x * dot;
            }
            bar_tile[tile_edge][tile_k] = bar_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (edge < params.num_edges && hidden < TILE_HIDDEN) {
            for (uint tile_k = 0; tile_k < TILE_K; tile_k++) {
                acc += bar_tile[local_edge][tile_k]
                    * layer3_weight[(k0 + tile_k) * TILE_HIDDEN + hidden];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (edge < params.num_edges && hidden < TILE_HIDDEN) {
        bar_hidden2[edge * TILE_HIDDEN + hidden] = acc * 0.125f;
    }
}

kernel void mp0b_product0_vector_contraction_vjp_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device const float* weights_u1 [[buffer(3)]],
    device const float* weights_u2 [[buffer(4)]],
    device const float* weights_u3 [[buffer(5)]],
    device const uint* u1_offsets [[buffer(6)]],
    device const uint* u2_offsets [[buffer(7)]],
    device const uint* u3_offsets [[buffer(8)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(9)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(10)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(11)]],
    device float* grad_x [[buffer(12)]],
    constant Mp0bProduct0VectorVjpParams& params [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * params.num_features;
    if (gid >= total) {
        return;
    }
    uint node = gid / params.num_features;
    uint feature = gid - node * params.num_features;
    uint element = type_ids[node];
    uint x_base = (node * params.num_features + feature) * 16;
    uint grad_out_base = (node * params.num_features + feature) * params.num_equivariance;
    float x_values[16];
    float grad_values[16];
    for (uint ell = 0; ell < 16; ell++) {
        x_values[ell] = x[x_base + ell];
        grad_values[ell] = 0.0f;
    }
    for (uint equiv = 0; equiv < params.num_equivariance; equiv++) {
        float bar = grad_out[grad_out_base + equiv];
        for (uint index = u1_offsets[equiv]; index < u1_offsets[equiv + 1]; index++) {
            Mp0bVectorU1Entry entry = u1_entries[index];
            float weight = weights_u1[(element * params.u1_params + entry.param) * params.num_features + feature];
            float derivative = bar * weight * entry.value;
            grad_values[entry.ell] += derivative;
        }
        for (uint index = u2_offsets[equiv]; index < u2_offsets[equiv + 1]; index++) {
            Mp0bVectorU2Entry entry = u2_entries[index];
            float weight = weights_u2[(element * params.u2_params + entry.param) * params.num_features + feature];
            float derivative = bar * weight * entry.value;
            float left = x_values[entry.left];
            float right = x_values[entry.right];
            grad_values[entry.left] += derivative * right;
            grad_values[entry.right] += derivative * left;
        }
        for (uint index = u3_offsets[equiv]; index < u3_offsets[equiv + 1]; index++) {
            Mp0bVectorU3Entry entry = u3_entries[index];
            float weight = weights_u3[(element * params.u3_params + entry.param) * params.num_features + feature];
            float derivative = bar * weight * entry.value;
            float left = x_values[entry.left];
            float right = x_values[entry.right];
            float inner = x_values[entry.inner];
            if (entry.left == entry.right) {
                if (entry.right == entry.inner) {
                    grad_values[entry.left] += 3.0f * derivative * left * left;
                } else {
                    grad_values[entry.left] += 2.0f * derivative * left * inner;
                    grad_values[entry.inner] += derivative * left * left;
                }
            } else if (entry.right == entry.inner) {
                grad_values[entry.left] += derivative * right * right;
                grad_values[entry.right] += 2.0f * derivative * left * right;
            } else {
                grad_values[entry.left] += derivative * right * inner;
                grad_values[entry.right] += derivative * left * inner;
                grad_values[entry.inner] += derivative * left * right;
            }
        }
    }
    for (uint ell = 0; ell < 16; ell++) {
        grad_x[x_base + ell] += grad_values[ell];
    }
}

kernel void mp0b_product0_scalar_contraction_vjp_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device const float* weights_u1 [[buffer(3)]],
    device const float* weights_u2 [[buffer(4)]],
    device const float* weights_u3 [[buffer(5)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(6)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(7)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(8)]],
    device float* grad_x [[buffer(9)]],
    constant Mp0bProduct0ScalarVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 128;
    if (gid >= total) {
        return;
    }
    uint node = gid / 128;
    uint feature = gid - node * 128;
    uint element = type_ids[node];
    uint x_base = (node * 128 + feature) * 16;
    float bar = grad_out[node * 128 + feature];
    float weight_u1[1];
    float weight_u2[4];
    float weight_u3[23];
    for (uint param = 0; param < 1; param++) {
        weight_u1[param] = weights_u1[(element * 1 + param) * 128 + feature];
    }
    for (uint param = 0; param < 4; param++) {
        weight_u2[param] = weights_u2[(element * 4 + param) * 128 + feature];
    }
    for (uint param = 0; param < 23; param++) {
        weight_u3[param] = weights_u3[(element * 23 + param) * 128 + feature];
    }
    float x_values[16];
    float grad_values[16];
    for (uint ell = 0; ell < 16; ell++) {
        x_values[ell] = x[x_base + ell];
        grad_values[ell] = 0.0f;
    }
    for (uint index = 0; index < params.u1_len; index++) {
        Mp0bVectorU1Entry entry = u1_entries[index];
        float weight = weight_u1[entry.param];
        grad_values[entry.ell] += bar * weight * entry.value;
    }
    for (uint index = 0; index < params.u2_len; index++) {
        Mp0bVectorU2Entry entry = u2_entries[index];
        float weight = weight_u2[entry.param];
        float derivative = bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        grad_values[entry.left] += derivative * right;
        grad_values[entry.right] += derivative * left;
    }
    for (uint index = 0; index < params.u3_len; index++) {
        Mp0bVectorU3Entry entry = u3_entries[index];
        float weight = weight_u3[entry.param];
        float derivative = bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        float inner = x_values[entry.inner];
        if (entry.left == entry.right) {
            if (entry.right == entry.inner) {
                grad_values[entry.left] += 3.0f * derivative * left * left;
            } else {
                grad_values[entry.left] += 2.0f * derivative * left * inner;
                grad_values[entry.inner] += derivative * left * left;
            }
        } else if (entry.right == entry.inner) {
            grad_values[entry.left] += derivative * right * right;
            grad_values[entry.right] += 2.0f * derivative * left * right;
        } else {
            grad_values[entry.left] += derivative * right * inner;
            grad_values[entry.right] += derivative * left * inner;
            grad_values[entry.inner] += derivative * left * right;
        }
    }
    for (uint ell = 0; ell < 16; ell++) {
        grad_x[x_base + ell] += grad_values[ell];
    }
}

kernel void mace_scalar_contraction_vjp_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device const float* weights_u1 [[buffer(3)]],
    device const float* weights_u2 [[buffer(4)]],
    device const float* weights_u3 [[buffer(5)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(6)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(7)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(8)]],
    device float* grad_x [[buffer(9)]],
    constant MaceScalarVjpParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * params.num_features;
    if (gid >= total || params.num_ell != 16) {
        return;
    }
    uint node = gid / params.num_features;
    uint feature = gid - node * params.num_features;
    uint element = type_ids[node];
    if (element >= params.num_elements) {
        return;
    }
    uint x_base = (node * params.num_features + feature) * params.num_ell;
    float bar = grad_out[node * params.num_features + feature];
    float x_values[16];
    float grad_values[16];
    for (uint ell = 0; ell < 16; ell++) {
        x_values[ell] = x[x_base + ell];
        grad_values[ell] = 0.0f;
    }
    for (uint index = 0; index < params.u1_len; index++) {
        Mp0bVectorU1Entry entry = u1_entries[index];
        float weight =
            weights_u1[(element * params.u1_params + entry.param) * params.num_features + feature];
        grad_values[entry.ell] += bar * weight * entry.value;
    }
    for (uint index = 0; index < params.u2_len; index++) {
        Mp0bVectorU2Entry entry = u2_entries[index];
        float weight =
            weights_u2[(element * params.u2_params + entry.param) * params.num_features + feature];
        float derivative = bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        grad_values[entry.left] += derivative * right;
        grad_values[entry.right] += derivative * left;
    }
    for (uint index = 0; index < params.u3_len; index++) {
        Mp0bVectorU3Entry entry = u3_entries[index];
        float weight =
            weights_u3[(element * params.u3_params + entry.param) * params.num_features + feature];
        float derivative = bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        float inner = x_values[entry.inner];
        if (entry.left == entry.right) {
            if (entry.right == entry.inner) {
                grad_values[entry.left] += 3.0f * derivative * left * left;
            } else {
                grad_values[entry.left] += 2.0f * derivative * left * inner;
                grad_values[entry.inner] += derivative * left * left;
            }
        } else if (entry.right == entry.inner) {
            grad_values[entry.left] += derivative * right * right;
            grad_values[entry.right] += 2.0f * derivative * left * right;
        } else {
            grad_values[entry.left] += derivative * right * inner;
            grad_values[entry.right] += derivative * left * inner;
            grad_values[entry.inner] += derivative * left * right;
        }
    }
    for (uint ell = 0; ell < 16; ell++) {
        grad_x[x_base + ell] += grad_values[ell];
    }
}

kernel void mace_scalar_contraction_forward_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weights_u1 [[buffer(2)]],
    device const float* weights_u2 [[buffer(3)]],
    device const float* weights_u3 [[buffer(4)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(5)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(6)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant MaceScalarForwardParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * params.num_features;
    if (gid >= total) {
        return;
    }
    uint node = gid / params.num_features;
    uint feature = gid - node * params.num_features;
    uint element = type_ids[node];
    uint x_base = (node * params.num_features + feature) * params.num_ell;
    float acc = 0.0f;
    for (uint index = 0; index < params.u1_len; index++) {
        Mp0bVectorU1Entry entry = u1_entries[index];
        float weight =
            weights_u1[(element * params.u1_params + entry.param) * params.num_features + feature];
        acc += weight * entry.value * x[x_base + entry.ell];
    }
    for (uint index = 0; index < params.u2_len; index++) {
        Mp0bVectorU2Entry entry = u2_entries[index];
        float weight =
            weights_u2[(element * params.u2_params + entry.param) * params.num_features + feature];
        acc += weight * entry.value * x[x_base + entry.left] * x[x_base + entry.right];
    }
    for (uint index = 0; index < params.u3_len; index++) {
        Mp0bVectorU3Entry entry = u3_entries[index];
        float weight =
            weights_u3[(element * params.u3_params + entry.param) * params.num_features + feature];
        acc += weight
            * entry.value
            * x[x_base + entry.left]
            * x[x_base + entry.right]
            * x[x_base + entry.inner];
    }
    out[node * params.out_stride + params.out_offset + feature] = acc;
}

kernel void mp0b_product0_linear_scalar_vector_contraction_vjp_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* bar_product_out [[buffer(2)]],
    device const float* product_linear_weight [[buffer(3)]],
    device const float* product_linear_output_mask [[buffer(4)]],
    device const float* scalar_weights_u1 [[buffer(5)]],
    device const float* scalar_weights_u2 [[buffer(6)]],
    device const float* scalar_weights_u3 [[buffer(7)]],
    device const Mp0bVectorU1Entry* scalar_u1_entries [[buffer(8)]],
    device const Mp0bVectorU2Entry* scalar_u2_entries [[buffer(9)]],
    device const Mp0bVectorU3Entry* scalar_u3_entries [[buffer(10)]],
    device const float* vector_weights_u1 [[buffer(11)]],
    device const float* vector_weights_u2 [[buffer(12)]],
    device const float* vector_weights_u3 [[buffer(13)]],
    device const uint* vector_u1_offsets [[buffer(14)]],
    device const uint* vector_u2_offsets [[buffer(15)]],
    device const uint* vector_u3_offsets [[buffer(16)]],
    device const Mp0bVectorU1Entry* vector_u1_entries [[buffer(17)]],
    device const Mp0bVectorU2Entry* vector_u2_entries [[buffer(18)]],
    device const Mp0bVectorU3Entry* vector_u3_entries [[buffer(19)]],
    device float* grad_x [[buffer(20)]],
    constant Mp0bProduct0ScalarVjpParams& params [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 128;
    if (gid >= total) {
        return;
    }
    uint node = gid / 128;
    uint feature = gid - node * 128;
    uint element = type_ids[node];
    uint product_base = node * 512;

    bool output_mask_all_ones = (params.flags & 2u) != 0u;
    float scalar_bar = 0.0f;
    float vector_bar0 = 0.0f;
    float vector_bar1 = 0.0f;
    float vector_bar2 = 0.0f;
    uint scalar_weight_base = feature * 128;
    uint vector_weight_base = 16384 + feature * 128;
    if (output_mask_all_ones) {
        for (uint out_feature = 0; out_feature < 128; out_feature++) {
            scalar_bar += bar_product_out[product_base + out_feature]
                * product_linear_weight[scalar_weight_base + out_feature];
            float vector_weight = product_linear_weight[vector_weight_base + out_feature];
            uint out_vector_base = product_base + 128 + out_feature * 3;
            vector_bar0 += bar_product_out[out_vector_base] * vector_weight;
            vector_bar1 += bar_product_out[out_vector_base + 1] * vector_weight;
            vector_bar2 += bar_product_out[out_vector_base + 2] * vector_weight;
        }
    } else {
        for (uint out_feature = 0; out_feature < 128; out_feature++) {
            scalar_bar += bar_product_out[product_base + out_feature]
                * product_linear_output_mask[out_feature]
                * product_linear_weight[scalar_weight_base + out_feature];
            float vector_weight = product_linear_weight[vector_weight_base + out_feature];
            uint out_vector_base = product_base + 128 + out_feature * 3;
            uint mask_vector_base = 128 + out_feature * 3;
            vector_bar0 += bar_product_out[out_vector_base]
                * product_linear_output_mask[mask_vector_base]
                * vector_weight;
            vector_bar1 += bar_product_out[out_vector_base + 1]
                * product_linear_output_mask[mask_vector_base + 1]
                * vector_weight;
            vector_bar2 += bar_product_out[out_vector_base + 2]
                * product_linear_output_mask[mask_vector_base + 2]
                * vector_weight;
        }
    }

    uint x_base = (node * 128 + feature) * 16;
    float x_values[16];
    float grad_values[16];
    for (uint ell = 0; ell < 16; ell++) {
        x_values[ell] = x[x_base + ell];
        grad_values[ell] = 0.0f;
    }

    for (uint index = 0; index < params.u1_len; index++) {
        Mp0bVectorU1Entry entry = scalar_u1_entries[index];
        float weight = scalar_weights_u1[(element * 1 + entry.param) * 128 + feature];
        grad_values[entry.ell] += scalar_bar * weight * entry.value;
    }
    for (uint index = 0; index < params.u2_len; index++) {
        Mp0bVectorU2Entry entry = scalar_u2_entries[index];
        float weight = scalar_weights_u2[(element * 4 + entry.param) * 128 + feature];
        float derivative = scalar_bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        grad_values[entry.left] += derivative * right;
        grad_values[entry.right] += derivative * left;
    }
    for (uint index = 0; index < params.u3_len; index++) {
        Mp0bVectorU3Entry entry = scalar_u3_entries[index];
        float weight = scalar_weights_u3[(element * 23 + entry.param) * 128 + feature];
        float derivative = scalar_bar * weight * entry.value;
        float left = x_values[entry.left];
        float right = x_values[entry.right];
        float inner = x_values[entry.inner];
        if (entry.left == entry.right) {
            if (entry.right == entry.inner) {
                grad_values[entry.left] += 3.0f * derivative * left * left;
            } else {
                grad_values[entry.left] += 2.0f * derivative * left * inner;
                grad_values[entry.inner] += derivative * left * left;
            }
        } else if (entry.right == entry.inner) {
            grad_values[entry.left] += derivative * right * right;
            grad_values[entry.right] += 2.0f * derivative * left * right;
        } else {
            grad_values[entry.left] += derivative * right * inner;
            grad_values[entry.right] += derivative * left * inner;
            grad_values[entry.inner] += derivative * left * right;
        }
    }

    float vector_bars[3] = {vector_bar0, vector_bar1, vector_bar2};
    for (uint equiv = 0; equiv < 3; equiv++) {
        float bar = vector_bars[equiv];
        for (uint index = vector_u1_offsets[equiv]; index < vector_u1_offsets[equiv + 1]; index++) {
            Mp0bVectorU1Entry entry = vector_u1_entries[index];
            float weight = vector_weights_u1[(element * 1 + entry.param) * 128 + feature];
            grad_values[entry.ell] += bar * weight * entry.value;
        }
        for (uint index = vector_u2_offsets[equiv]; index < vector_u2_offsets[equiv + 1]; index++) {
            Mp0bVectorU2Entry entry = vector_u2_entries[index];
            float weight = vector_weights_u2[(element * 6 + entry.param) * 128 + feature];
            float derivative = bar * weight * entry.value;
            float left = x_values[entry.left];
            float right = x_values[entry.right];
            grad_values[entry.left] += derivative * right;
            grad_values[entry.right] += derivative * left;
        }
        for (uint index = vector_u3_offsets[equiv]; index < vector_u3_offsets[equiv + 1]; index++) {
            Mp0bVectorU3Entry entry = vector_u3_entries[index];
            float weight = vector_weights_u3[(element * 51 + entry.param) * 128 + feature];
            float derivative = bar * weight * entry.value;
            float left = x_values[entry.left];
            float right = x_values[entry.right];
            float inner = x_values[entry.inner];
            if (entry.left == entry.right) {
                if (entry.right == entry.inner) {
                    grad_values[entry.left] += 3.0f * derivative * left * left;
                } else {
                    grad_values[entry.left] += 2.0f * derivative * left * inner;
                    grad_values[entry.inner] += derivative * left * left;
                }
            } else if (entry.right == entry.inner) {
                grad_values[entry.left] += derivative * right * right;
                grad_values[entry.right] += 2.0f * derivative * left * right;
            } else {
                grad_values[entry.left] += derivative * right * inner;
                grad_values[entry.right] += derivative * left * inner;
                grad_values[entry.inner] += derivative * left * right;
            }
        }
    }

    if ((params.flags & 1u) != 0u) {
        for (uint ell = 0; ell < 16; ell++) {
            grad_x[x_base + ell] = grad_values[ell];
        }
    } else {
        for (uint ell = 0; ell < 16; ell++) {
            grad_x[x_base + ell] += grad_values[ell];
        }
    }
}

kernel void mp0b_product0_scalar_contraction_forward_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weights_u1 [[buffer(2)]],
    device const float* weights_u2 [[buffer(3)]],
    device const float* weights_u3 [[buffer(4)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(5)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(6)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(7)]],
    device float* product_symmetric [[buffer(8)]],
    constant Mp0bProduct0ScalarVjpParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 128;
    if (gid >= total) {
        return;
    }
    uint node = gid / 128;
    uint feature = gid - node * 128;
    uint element = type_ids[node];
    uint x_base = (node * 128 + feature) * 16;
    float weight_u1[1];
    float weight_u2[4];
    float weight_u3[23];
    for (uint param = 0; param < 1; param++) {
        weight_u1[param] = weights_u1[(element * 1 + param) * 128 + feature];
    }
    for (uint param = 0; param < 4; param++) {
        weight_u2[param] = weights_u2[(element * 4 + param) * 128 + feature];
    }
    for (uint param = 0; param < 23; param++) {
        weight_u3[param] = weights_u3[(element * 23 + param) * 128 + feature];
    }
    float acc = 0.0f;
    for (uint index = 0; index < params.u1_len; index++) {
        Mp0bVectorU1Entry entry = u1_entries[index];
        float weight = weight_u1[entry.param];
        acc += weight * entry.value * x[x_base + entry.ell];
    }
    for (uint index = 0; index < params.u2_len; index++) {
        Mp0bVectorU2Entry entry = u2_entries[index];
        float weight = weight_u2[entry.param];
        acc += weight * entry.value * x[x_base + entry.left] * x[x_base + entry.right];
    }
    for (uint index = 0; index < params.u3_len; index++) {
        Mp0bVectorU3Entry entry = u3_entries[index];
        float weight = weight_u3[entry.param];
        acc += weight
            * entry.value
            * x[x_base + entry.left]
            * x[x_base + entry.right]
            * x[x_base + entry.inner];
    }
    product_symmetric[node * 512 + feature] = acc;
}

kernel void mp0b_product0_vector_contraction_forward_f32(
    device const uint* type_ids [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* weights_u1 [[buffer(2)]],
    device const float* weights_u2 [[buffer(3)]],
    device const float* weights_u3 [[buffer(4)]],
    device const uint* u1_offsets [[buffer(5)]],
    device const uint* u2_offsets [[buffer(6)]],
    device const uint* u3_offsets [[buffer(7)]],
    device const Mp0bVectorU1Entry* u1_entries [[buffer(8)]],
    device const Mp0bVectorU2Entry* u2_entries [[buffer(9)]],
    device const Mp0bVectorU3Entry* u3_entries [[buffer(10)]],
    device float* product_symmetric [[buffer(11)]],
    constant Mp0bProduct0VectorVjpParams& params [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * params.num_features * params.num_equivariance;
    if (gid >= total) {
        return;
    }
    uint row_len = params.num_features * params.num_equivariance;
    uint node = gid / row_len;
    uint local = gid - node * row_len;
    uint feature = local / params.num_equivariance;
    uint equiv = local - feature * params.num_equivariance;
    uint element = type_ids[node];
    uint x_base = (node * params.num_features + feature) * 16;
    float acc = 0.0f;
    for (uint index = u1_offsets[equiv]; index < u1_offsets[equiv + 1]; index++) {
        Mp0bVectorU1Entry entry = u1_entries[index];
        float weight = weights_u1[(element * params.u1_params + entry.param) * params.num_features + feature];
        acc += weight * entry.value * x[x_base + entry.ell];
    }
    for (uint index = u2_offsets[equiv]; index < u2_offsets[equiv + 1]; index++) {
        Mp0bVectorU2Entry entry = u2_entries[index];
        float weight = weights_u2[(element * params.u2_params + entry.param) * params.num_features + feature];
        acc += weight * entry.value * x[x_base + entry.left] * x[x_base + entry.right];
    }
    for (uint index = u3_offsets[equiv]; index < u3_offsets[equiv + 1]; index++) {
        Mp0bVectorU3Entry entry = u3_entries[index];
        float weight = weights_u3[(element * params.u3_params + entry.param) * params.num_features + feature];
        acc += weight
            * entry.value
            * x[x_base + entry.left]
            * x[x_base + entry.right]
            * x[x_base + entry.inner];
    }
    product_symmetric[node * params.output_stride
        + params.output_offset
        + feature * params.num_equivariance
        + equiv] = acc;
}

kernel void mp0b_layer1_hidden_forward_f32(
    device const float* input [[buffer(0)]],
    device const float* density [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* output_mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 2048;
    if (gid >= total) {
        return;
    }
    uint node = gid / 2048;
    uint lane = gid - node * 2048;
    uint output_offset = 0;
    uint component_dim = 1;
    uint source0 = 0;
    uint source1 = 1;
    uint source2 = 10;
    uint source_count = 2;
    if (lane < 128) {
        output_offset = 0; component_dim = 1;
        source0 = 0; source1 = 1; source_count = 2;
    } else if (lane < 512) {
        output_offset = 128; component_dim = 3;
        source0 = 2; source1 = 3; source2 = 4; source_count = 3;
    } else if (lane < 1152) {
        output_offset = 512; component_dim = 5;
        source0 = 5; source1 = 6; source2 = 7; source_count = 3;
    } else {
        output_offset = 1152; component_dim = 7;
        source0 = 8; source1 = 9; source_count = 2;
    }
    uint relative = lane - output_offset;
    uint out_feature = relative / component_dim;
    uint component = relative - out_feature * component_dim;
    uint source_blocks[3] = { source0, source1, source2 };
    uint input_offsets[10] = { 0, 128, 256, 640, 1024, 1408, 2048, 2688, 3328, 4224 };
    uint input_base = node * 5120;
    float acc = 0.0f;
    for (uint source_index = 0; source_index < source_count; source_index++) {
        uint block = source_blocks[source_index];
        uint input_offset = input_offsets[block];
        uint weight_base = block * 16384 + out_feature;
        for (uint in_feature = 0; in_feature < 128; in_feature++) {
            uint input_index = input_base + input_offset + in_feature * component_dim + component;
            acc += input[input_index] * weight[weight_base + in_feature * 128];
        }
    }
    output[gid] = acc * output_mask[lane] / (density[node] + 1.0f);
}

kernel void mp0b_layer0_hidden_forward_f32(
    device const float* input [[buffer(0)]],
    device const float* density [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* output_mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Mp0bLayer1HiddenVjpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.num_nodes * 2048;
    if (gid >= total) {
        return;
    }
    uint node = gid / 2048;
    uint lane = gid - node * 2048;
    uint block = 0;
    uint offset = 0;
    uint component_dim = 1;
    if (lane < 128) {
        block = 0; offset = 0; component_dim = 1;
    } else if (lane < 512) {
        block = 1; offset = 128; component_dim = 3;
    } else if (lane < 1152) {
        block = 2; offset = 512; component_dim = 5;
    } else {
        block = 3; offset = 1152; component_dim = 7;
    }
    uint relative = lane - offset;
    uint out_feature = relative / component_dim;
    uint component = relative - out_feature * component_dim;
    uint input_base = node * 2048;
    uint weight_base = block * 16384 + out_feature;
    const float path_weight = 0.08838834764831845f;
    float acc = 0.0f;
    for (uint in_feature = 0; in_feature < 128; in_feature++) {
        uint input_index = input_base + offset + in_feature * component_dim + component;
        acc += input[input_index] * path_weight * weight[weight_base + in_feature * 128];
    }
    output[gid] = acc * output_mask[lane] / (density[node] + 1.0f);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bNodeEmbeddingParams {
    num_nodes: u32,
    out_features: u32,
    path_weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bDense64VjpParams {
    num_edges: u32,
    out_dim: u32,
    inv_norm: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bEdgeFeatVjpParams {
    num_edges: u32,
    num_bessel: u32,
    inv_norm: f32,
    flags: u32,
}

const MP0B_EDGE_FEATS_VJP_ASSIGN_OUTPUT_FLAG: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bDensityVjpEdgeFeatParams {
    num_edges: u32,
    num_bessel: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bDenseTo64Params {
    num_edges: u32,
    in_dim: u32,
    inv_norm: f32,
    silu_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bDense64ForwardParams {
    num_edges: u32,
    out_dim: u32,
    inv_norm: f32,
    silu_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1HiddenVjpParams {
    num_nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1ConvTpVjpParams {
    num_nodes: u32,
    num_edges: u32,
    flags: u32,
}

const MP0B_CONV_TP_VJP_ACCUMULATE_EDGE_ATTRS_FLAG: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1TpTemplateMetal {
    x_base: u32,
    x_dim: u32,
    y_base: u32,
    out_base: u32,
    out_dim: u32,
    weight_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1CgMetal {
    i_m: u32,
    j_m: u32,
    k_m: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1WeightLaneMetal {
    template_index: u32,
    feature: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bLayer1InputLaneMetal {
    template_index: u32,
    feature: u32,
    input_component: u32,
}

type Layer1TpMetadataMetal = (
    Vec<Mp0bLayer1TpTemplateMetal>,
    Vec<u32>,
    Vec<Mp0bLayer1CgMetal>,
    Vec<u32>,
    Vec<Mp0bLayer1CgMetal>,
    Vec<u32>,
    Vec<Mp0bLayer1CgMetal>,
    Vec<u32>,
    Vec<Mp0bLayer1InputLaneMetal>,
    Vec<Mp0bLayer1WeightLaneMetal>,
);

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bVectorU1EntryMetal {
    ell: u32,
    param: u32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bVectorU2EntryMetal {
    left: u32,
    right: u32,
    param: u32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bVectorU3EntryMetal {
    left: u32,
    right: u32,
    inner: u32,
    param: u32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bProduct0VectorVjpParams {
    num_nodes: u32,
    num_elements: u32,
    num_features: u32,
    num_equivariance: u32,
    u1_params: u32,
    u2_params: u32,
    u3_params: u32,
    output_stride: u32,
    output_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mp0bProduct0ScalarVjpParams {
    num_nodes: u32,
    u1_len: u32,
    u2_len: u32,
    u3_len: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaceScalarVjpParams {
    num_nodes: u32,
    num_elements: u32,
    num_features: u32,
    num_ell: u32,
    u1_params: u32,
    u2_params: u32,
    u3_params: u32,
    u1_len: u32,
    u2_len: u32,
    u3_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaceScalarForwardParams {
    num_nodes: u32,
    num_elements: u32,
    num_features: u32,
    num_ell: u32,
    u1_params: u32,
    u2_params: u32,
    u3_params: u32,
    u1_len: u32,
    u2_len: u32,
    u3_len: u32,
    out_stride: u32,
    out_offset: u32,
}

const MP0B_PRODUCT0_VJP_ASSIGN_OUTPUT_FLAG: u32 = 1;
const MP0B_PRODUCT0_VJP_OUTPUT_MASK_ALL_ONES_FLAG: u32 = 2;

struct Mp0bMetalBackend {
    device: Device,
    queue: CommandQueue,
    layer1_tp_templates_buffer: Buffer,
    layer1_tp_cg_offsets_buffer: Buffer,
    layer1_tp_cgs_buffer: Buffer,
    layer1_tp_forward_cg_offsets_buffer: Buffer,
    layer1_tp_forward_cgs_buffer: Buffer,
    layer1_tp_vjp_i_cg_offsets_buffer: Buffer,
    layer1_tp_vjp_i_cgs_buffer: Buffer,
    layer1_tp_input_lane_offsets_buffer: Buffer,
    layer1_tp_input_lane_entries_buffer: Buffer,
    layer1_tp_weight_lanes_buffer: Buffer,
    node_embedding_pipeline: ComputePipelineState,
    dense64_vjp_derivative_pipeline: ComputePipelineState,
    apply_derivative64_pipeline: ComputePipelineState,
    edge_feats_vjp_add_pipeline: ComputePipelineState,
    density_vjp_edge_feats_add_pipeline: ComputePipelineState,
    dense_to_64_silu_derivative_pipeline: ComputePipelineState,
    dense64_silu_derivative_pipeline: ComputePipelineState,
    dense64_output_tiled_pipeline: ComputePipelineState,
    layer1_hidden_vjp_density_pipeline: ComputePipelineState,
    layer1_hidden_vjp_density_feature_major_pipeline: ComputePipelineState,
    layer1_hidden_vjp_input_pipeline: ComputePipelineState,
    layer1_hidden_vjp_input_feature_major_pipeline: ComputePipelineState,
    layer0_hidden_vjp_input_pipeline: ComputePipelineState,
    selector_vjp_input_pipeline: ComputePipelineState,
    selector_to_feature_major_forward_pipeline: ComputePipelineState,
    scalar_vector_product_linear_forward_packed_pipeline: ComputePipelineState,
    scalar_vector_product_linear_vjp_input_packed_pipeline: ComputePipelineState,
    layer1_conv_tp_forward_pipeline: ComputePipelineState,
    oam_layer1_conv_tp_forward_pipeline: ComputePipelineState,
    layer1_conv_tp_vjp_linear_up_pipeline: ComputePipelineState,
    layer1_conv_tp_vjp_edge_attrs_pipeline: ComputePipelineState,
    layer1_conv_tp_vjp_hidden2_fused_tiled_pipeline: ComputePipelineState,
    layer0_conv_tp_vjp_edge_attrs_pipeline: ComputePipelineState,
    layer0_conv_tp_vjp_hidden2_fused_tiled_pipeline: ComputePipelineState,
    product0_vector_vjp_pipeline: ComputePipelineState,
    product0_scalar_vjp_pipeline: ComputePipelineState,
    mace_scalar_vjp_pipeline: ComputePipelineState,
    mace_scalar_forward_pipeline: ComputePipelineState,
    product0_linear_scalar_vector_vjp_pipeline: ComputePipelineState,
    product0_scalar_forward_pipeline: ComputePipelineState,
    product0_vector_forward_pipeline: ComputePipelineState,
    layer1_hidden_forward_pipeline: ComputePipelineState,
    layer0_hidden_forward_pipeline: ComputePipelineState,
}

impl Mp0bMetalBackend {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "no Metal device found".to_owned())?;
        let library = device
            .new_library_with_source(MP0B_METAL_SOURCE, &CompileOptions::new())
            .map_err(|err| format!("compile Metal library: {err}"))?;
        let function = library
            .get_function("mp0b_node_embedding_f32", None)
            .map_err(|err| format!("load mp0b_node_embedding_f32: {err}"))?;
        let node_embedding_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_node_embedding_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_dense64_vjp_derivative_f32", None)
            .map_err(|err| format!("load mp0b_dense64_vjp_derivative_f32: {err}"))?;
        let dense64_vjp_derivative_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_dense64_vjp_derivative_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_apply_derivative64_f32", None)
            .map_err(|err| format!("load mp0b_apply_derivative64_f32: {err}"))?;
        let apply_derivative64_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_apply_derivative64_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_edge_feats_vjp_add_f32", None)
            .map_err(|err| format!("load mp0b_edge_feats_vjp_add_f32: {err}"))?;
        let edge_feats_vjp_add_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_edge_feats_vjp_add_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_density_vjp_edge_feats_add_f32", None)
            .map_err(|err| format!("load mp0b_density_vjp_edge_feats_add_f32: {err}"))?;
        let density_vjp_edge_feats_add_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_density_vjp_edge_feats_add_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_dense_to_64_silu_derivative_f32", None)
            .map_err(|err| format!("load mp0b_dense_to_64_silu_derivative_f32: {err}"))?;
        let dense_to_64_silu_derivative_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_dense_to_64_silu_derivative_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_dense64_silu_derivative_f32", None)
            .map_err(|err| format!("load mp0b_dense64_silu_derivative_f32: {err}"))?;
        let dense64_silu_derivative_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_dense64_silu_derivative_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_dense64_output_tiled_f32", None)
            .map_err(|err| format!("load mp0b_dense64_output_tiled_f32: {err}"))?;
        let dense64_output_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_dense64_output_tiled_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_layer1_hidden_vjp_density_f32", None)
            .map_err(|err| format!("load mp0b_layer1_hidden_vjp_density_f32: {err}"))?;
        let layer1_hidden_vjp_density_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer1_hidden_vjp_density_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_layer1_hidden_vjp_density_feature_major_f32", None)
            .map_err(|err| {
                format!("load mp0b_layer1_hidden_vjp_density_feature_major_f32: {err}")
            })?;
        let layer1_hidden_vjp_density_feature_major_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_layer1_hidden_vjp_density_feature_major_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_layer1_hidden_vjp_input_f32", None)
            .map_err(|err| format!("load mp0b_layer1_hidden_vjp_input_f32: {err}"))?;
        let layer1_hidden_vjp_input_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer1_hidden_vjp_input_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_layer1_hidden_vjp_input_feature_major_f32", None)
            .map_err(|err| format!("load mp0b_layer1_hidden_vjp_input_feature_major_f32: {err}"))?;
        let layer1_hidden_vjp_input_feature_major_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_layer1_hidden_vjp_input_feature_major_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_layer0_hidden_vjp_input_f32", None)
            .map_err(|err| format!("load mp0b_layer0_hidden_vjp_input_f32: {err}"))?;
        let layer0_hidden_vjp_input_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer0_hidden_vjp_input_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_hidden_message_selector_vjp_input_f32", None)
            .map_err(|err| format!("load mp0b_hidden_message_selector_vjp_input_f32: {err}"))?;
        let selector_vjp_input_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_hidden_message_selector_vjp_input_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_hidden_message_selector_to_feature_major_f32", None)
            .map_err(|err| {
                format!("load mp0b_hidden_message_selector_to_feature_major_f32: {err}")
            })?;
        let selector_to_feature_major_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_hidden_message_selector_to_feature_major_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_scalar_vector_product_linear_forward_packed_f32", None)
            .map_err(|err| {
                format!("load mp0b_scalar_vector_product_linear_forward_packed_f32: {err}")
            })?;
        let scalar_vector_product_linear_forward_packed_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!(
                    "build mp0b_scalar_vector_product_linear_forward_packed_f32 pipeline: {err}"
                )
            })?;
        let function = library
            .get_function(
                "mp0b_scalar_vector_product_linear_vjp_input_packed_f32",
                None,
            )
            .map_err(|err| {
                format!("load mp0b_scalar_vector_product_linear_vjp_input_packed_f32: {err}")
            })?;
        let scalar_vector_product_linear_vjp_input_packed_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!(
                    "build mp0b_scalar_vector_product_linear_vjp_input_packed_f32 pipeline: {err}"
                )
            })?;
        let function = library
            .get_function("mp0b_layer1_conv_tp_forward_f32", None)
            .map_err(|err| format!("load mp0b_layer1_conv_tp_forward_f32: {err}"))?;
        let layer1_conv_tp_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer1_conv_tp_forward_f32 pipeline: {err}"))?;
        let function = library
            .get_function("oam_layer1_conv_tp_forward_f32", None)
            .map_err(|err| format!("load oam_layer1_conv_tp_forward_f32: {err}"))?;
        let oam_layer1_conv_tp_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build oam_layer1_conv_tp_forward_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_layer1_conv_tp_vjp_linear_up_f32", None)
            .map_err(|err| format!("load mp0b_layer1_conv_tp_vjp_linear_up_f32: {err}"))?;
        let layer1_conv_tp_vjp_linear_up_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_layer1_conv_tp_vjp_linear_up_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function(
                "mp0b_layer1_conv_tp_vjp_edge_attrs_feature_parallel_f32",
                None,
            )
            .map_err(|err| {
                format!("load mp0b_layer1_conv_tp_vjp_edge_attrs_feature_parallel_f32: {err}")
            })?;
        let layer1_conv_tp_vjp_edge_attrs_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!(
                    "build mp0b_layer1_conv_tp_vjp_edge_attrs_feature_parallel_f32 pipeline: {err}"
                )
            })?;
        let function = library
            .get_function("mp0b_layer1_conv_tp_vjp_hidden2_fused_tiled_f32", None)
            .map_err(|err| {
                format!("load mp0b_layer1_conv_tp_vjp_hidden2_fused_tiled_f32: {err}")
            })?;
        let layer1_conv_tp_vjp_hidden2_fused_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_layer1_conv_tp_vjp_hidden2_fused_tiled_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function(
                "mp0b_layer0_conv_tp_vjp_edge_attrs_feature_parallel_f32",
                None,
            )
            .map_err(|err| {
                format!("load mp0b_layer0_conv_tp_vjp_edge_attrs_feature_parallel_f32: {err}")
            })?;
        let layer0_conv_tp_vjp_edge_attrs_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!(
                    "build mp0b_layer0_conv_tp_vjp_edge_attrs_feature_parallel_f32 pipeline: {err}"
                )
            })?;
        let function = library
            .get_function("mp0b_layer0_conv_tp_vjp_hidden2_fused_tiled_f32", None)
            .map_err(|err| {
                format!("load mp0b_layer0_conv_tp_vjp_hidden2_fused_tiled_f32: {err}")
            })?;
        let layer0_conv_tp_vjp_hidden2_fused_tiled_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_layer0_conv_tp_vjp_hidden2_fused_tiled_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_product0_vector_contraction_vjp_f32", None)
            .map_err(|err| format!("load mp0b_product0_vector_contraction_vjp_f32: {err}"))?;
        let product0_vector_vjp_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_product0_vector_contraction_vjp_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_product0_scalar_contraction_vjp_f32", None)
            .map_err(|err| format!("load mp0b_product0_scalar_contraction_vjp_f32: {err}"))?;
        let product0_scalar_vjp_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_product0_scalar_contraction_vjp_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mace_scalar_contraction_vjp_f32", None)
            .map_err(|err| format!("load mace_scalar_contraction_vjp_f32: {err}"))?;
        let mace_scalar_vjp_pipeline =
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|err| format!("build mace_scalar_contraction_vjp_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mace_scalar_contraction_forward_f32", None)
            .map_err(|err| format!("load mace_scalar_contraction_forward_f32: {err}"))?;
        let mace_scalar_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mace_scalar_contraction_forward_f32 pipeline: {err}"))?;
        let function = library
            .get_function(
                "mp0b_product0_linear_scalar_vector_contraction_vjp_f32",
                None,
            )
            .map_err(|err| {
                format!("load mp0b_product0_linear_scalar_vector_contraction_vjp_f32: {err}")
            })?;
        let product0_linear_scalar_vector_vjp_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!(
                    "build mp0b_product0_linear_scalar_vector_contraction_vjp_f32 pipeline: {err}"
                )
            })?;
        let function = library
            .get_function("mp0b_product0_scalar_contraction_forward_f32", None)
            .map_err(|err| format!("load mp0b_product0_scalar_contraction_forward_f32: {err}"))?;
        let product0_scalar_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_product0_scalar_contraction_forward_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_product0_vector_contraction_forward_f32", None)
            .map_err(|err| format!("load mp0b_product0_vector_contraction_forward_f32: {err}"))?;
        let product0_vector_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| {
                format!("build mp0b_product0_vector_contraction_forward_f32 pipeline: {err}")
            })?;
        let function = library
            .get_function("mp0b_layer1_hidden_forward_f32", None)
            .map_err(|err| format!("load mp0b_layer1_hidden_forward_f32: {err}"))?;
        let layer1_hidden_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer1_hidden_forward_f32 pipeline: {err}"))?;
        let function = library
            .get_function("mp0b_layer0_hidden_forward_f32", None)
            .map_err(|err| format!("load mp0b_layer0_hidden_forward_f32: {err}"))?;
        let layer0_hidden_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("build mp0b_layer0_hidden_forward_f32 pipeline: {err}"))?;
        let queue = device.new_command_queue();
        let (
            layer1_tp_templates,
            layer1_tp_cg_offsets,
            layer1_tp_cgs,
            layer1_tp_forward_cg_offsets,
            layer1_tp_forward_cgs,
            layer1_tp_vjp_i_cg_offsets,
            layer1_tp_vjp_i_cgs,
            layer1_tp_input_lane_offsets,
            layer1_tp_input_lane_entries,
            layer1_tp_weight_lanes,
        ) = pack_layer1_tp_metadata()?;
        let options = MTLResourceOptions::StorageModeShared;
        let layer1_tp_templates_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_templates, options);
        let layer1_tp_cg_offsets_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_cg_offsets, options);
        let layer1_tp_cgs_buffer = buffer_with_slice_on_device(&device, &layer1_tp_cgs, options);
        let layer1_tp_forward_cg_offsets_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_forward_cg_offsets, options);
        let layer1_tp_forward_cgs_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_forward_cgs, options);
        let layer1_tp_vjp_i_cg_offsets_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_vjp_i_cg_offsets, options);
        let layer1_tp_vjp_i_cgs_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_vjp_i_cgs, options);
        let layer1_tp_input_lane_offsets_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_input_lane_offsets, options);
        let layer1_tp_input_lane_entries_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_input_lane_entries, options);
        let layer1_tp_weight_lanes_buffer =
            buffer_with_slice_on_device(&device, &layer1_tp_weight_lanes, options);
        Ok(Self {
            device,
            queue,
            layer1_tp_templates_buffer,
            layer1_tp_cg_offsets_buffer,
            layer1_tp_cgs_buffer,
            layer1_tp_forward_cg_offsets_buffer,
            layer1_tp_forward_cgs_buffer,
            layer1_tp_vjp_i_cg_offsets_buffer,
            layer1_tp_vjp_i_cgs_buffer,
            layer1_tp_input_lane_offsets_buffer,
            layer1_tp_input_lane_entries_buffer,
            layer1_tp_weight_lanes_buffer,
            node_embedding_pipeline,
            dense64_vjp_derivative_pipeline,
            apply_derivative64_pipeline,
            edge_feats_vjp_add_pipeline,
            density_vjp_edge_feats_add_pipeline,
            dense_to_64_silu_derivative_pipeline,
            dense64_silu_derivative_pipeline,
            dense64_output_tiled_pipeline,
            layer1_hidden_vjp_density_pipeline,
            layer1_hidden_vjp_density_feature_major_pipeline,
            layer1_hidden_vjp_input_pipeline,
            layer1_hidden_vjp_input_feature_major_pipeline,
            layer0_hidden_vjp_input_pipeline,
            selector_vjp_input_pipeline,
            selector_to_feature_major_forward_pipeline,
            scalar_vector_product_linear_forward_packed_pipeline,
            scalar_vector_product_linear_vjp_input_packed_pipeline,
            layer1_conv_tp_forward_pipeline,
            oam_layer1_conv_tp_forward_pipeline,
            layer1_conv_tp_vjp_linear_up_pipeline,
            layer1_conv_tp_vjp_edge_attrs_pipeline,
            layer1_conv_tp_vjp_hidden2_fused_tiled_pipeline,
            layer0_conv_tp_vjp_edge_attrs_pipeline,
            layer0_conv_tp_vjp_hidden2_fused_tiled_pipeline,
            product0_vector_vjp_pipeline,
            product0_scalar_vjp_pipeline,
            mace_scalar_vjp_pipeline,
            mace_scalar_forward_pipeline,
            product0_linear_scalar_vector_vjp_pipeline,
            product0_scalar_forward_pipeline,
            product0_vector_forward_pipeline,
            layer1_hidden_forward_pipeline,
            layer0_hidden_forward_pipeline,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn node_embedding_f32(
        &self,
        type_ids: &[u32],
        weight: &[f32],
        output_mask: &[f32],
        out_features: usize,
        path_weight: f32,
        out: &mut [f32],
    ) -> Result<(), String> {
        let num_nodes = type_ids.len();
        if out.len() != num_nodes * out_features {
            return Err("node embedding output shape mismatch".to_owned());
        }
        if output_mask.len() != out_features {
            return Err("node embedding output mask shape mismatch".to_owned());
        }
        if !weight.len().is_multiple_of(out_features) {
            return Err("node embedding weight shape mismatch".to_owned());
        }
        let params = Mp0bNodeEmbeddingParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            out_features: u32::try_from(out_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            path_weight,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let weight_buffer = self.buffer_with_slice(weight, options);
        let mask_buffer = self.buffer_with_slice(output_mask, options);
        let output_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(out) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.node_embedding_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&mask_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);

        let total_threads = out.len() as u64;
        let execution_width = self.node_embedding_pipeline.thread_execution_width().max(1);
        let max_threads = self
            .node_embedding_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        let threads_per_group = execution_width.min(max_threads).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }

        // Safety: `output_buffer` was allocated in StorageModeShared with
        // exactly `out.len() * size_of::<f32>()` bytes and the command buffer
        // has completed before we copy from its contents pointer.
        let output_ptr = output_buffer.contents().cast::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(output_ptr, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_forward_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        output_dim: usize,
        silu_scale: f32,
        edge_feats: &[f32],
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        weight3: &[f32],
        read_output: bool,
    ) -> Result<Mp0bConvTpWeightsForwardMetal, String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if edge_feats.len() != num_edges * num_bessel {
            return Err("conv-TP weight forward edge feature shape mismatch".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight forward layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight forward layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight forward layer2 weight shape mismatch".to_owned());
        }
        if weight3.len() != HIDDEN_DIM * output_dim {
            return Err("conv-TP weight forward layer3 weight shape mismatch".to_owned());
        }
        if num_bessel == 0 || output_dim == 0 {
            return Err("conv-TP weight forward requires nonzero dimensions".to_owned());
        }
        let num_edges_u32 = u32::try_from(num_edges)
            .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?;
        let layer0_params = Mp0bDenseTo64Params {
            num_edges: num_edges_u32,
            in_dim: u32::try_from(num_bessel)
                .map_err(|_| "bessel dimension exceeds Metal u32 range".to_owned())?,
            inv_norm: 1.0 / (num_bessel as f32).sqrt(),
            silu_scale,
        };
        let hidden_params = Mp0bDense64ForwardParams {
            num_edges: num_edges_u32,
            out_dim: HIDDEN_DIM as u32,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
            silu_scale,
        };
        let output_params = Mp0bDense64ForwardParams {
            num_edges: num_edges_u32,
            out_dim: u32::try_from(output_dim)
                .map_err(|_| "output dimension exceeds Metal u32 range".to_owned())?,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
            silu_scale,
        };
        let dense_layer3_vjp_params = Mp0bDense64VjpParams {
            num_edges: num_edges_u32,
            out_dim: output_params.out_dim,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
        };
        let dense64_vjp_params = Mp0bDense64VjpParams {
            num_edges: num_edges_u32,
            out_dim: HIDDEN_DIM as u32,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
        };
        let edge_feat_vjp_params = Mp0bEdgeFeatVjpParams {
            num_edges: num_edges_u32,
            num_bessel: layer0_params.in_dim,
            inv_norm: 1.0 / (num_bessel as f32).sqrt(),
            flags: 0,
        };
        let edge_feat_vjp_assign_params = Mp0bEdgeFeatVjpParams {
            flags: MP0B_EDGE_FEATS_VJP_ASSIGN_OUTPUT_FLAG,
            ..edge_feat_vjp_params
        };

        let options = MTLResourceOptions::StorageModeShared;
        let edge_feats_buffer = self.buffer_with_slice(edge_feats, options);
        let weight0_buffer = self.buffer_with_slice(weight0, options);
        let weight1_buffer = self.buffer_with_slice(weight1, options);
        let weight2_buffer = self.buffer_with_slice(weight2, options);
        let weight3_buffer = self.buffer_with_slice(weight3, options);
        let hidden0_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let hidden1_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let hidden2_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let derivative0_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let derivative1_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let derivative2_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let output_len = num_edges * output_dim;
        let output_buffer = self
            .device
            .new_buffer((output_len * std::mem::size_of::<f32>()) as u64, options);
        let layer0_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&layer0_params), options);
        let hidden_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&hidden_params), options);
        let output_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&output_params), options);
        let dense_layer3_vjp_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&dense_layer3_vjp_params), options);
        let dense64_vjp_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&dense64_vjp_params), options);
        let edge_feat_vjp_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&edge_feat_vjp_params), options);
        let edge_feat_vjp_assign_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&edge_feat_vjp_assign_params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        self.encode_dense_to_64_silu_derivative(
            &encoder,
            &edge_feats_buffer,
            &weight0_buffer,
            &hidden0_buffer,
            &derivative0_buffer,
            &layer0_params_buffer,
            hidden_threads,
        );
        self.encode_dense64_silu_derivative(
            &encoder,
            &hidden0_buffer,
            &weight1_buffer,
            &hidden1_buffer,
            &derivative1_buffer,
            &hidden_params_buffer,
            hidden_threads,
        );
        self.encode_dense64_silu_derivative(
            &encoder,
            &hidden1_buffer,
            &weight2_buffer,
            &hidden2_buffer,
            &derivative2_buffer,
            &hidden_params_buffer,
            hidden_threads,
        );
        encoder.set_compute_pipeline_state(&self.dense64_output_tiled_pipeline);
        encoder.set_buffer(0, Some(&hidden2_buffer), 0);
        encoder.set_buffer(1, Some(&weight3_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&output_params_buffer), 0);
        let tiled_output_dim = output_dim.div_ceil(8) * 8;
        let tiled_num_edges = num_edges.div_ceil(8) * 8;
        encoder.dispatch_threads(
            MTLSize {
                width: tiled_output_dim as u64,
                height: tiled_num_edges as u64,
                depth: 1,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }

        let mut output = if read_output {
            vec![0.0_f32; output_len]
        } else {
            Vec::new()
        };
        let mut hidden0_derivative = vec![0.0_f32; hidden_len];
        let mut hidden1_derivative = vec![0.0_f32; hidden_len];
        let mut hidden2_derivative = vec![0.0_f32; hidden_len];
        // Safety: all buffers are StorageModeShared and sized exactly from the
        // destination slice lengths. The command buffer has completed.
        unsafe {
            if read_output {
                std::ptr::copy_nonoverlapping(
                    output_buffer.contents().cast::<f32>(),
                    output.as_mut_ptr(),
                    output.len(),
                );
            }
            std::ptr::copy_nonoverlapping(
                derivative0_buffer.contents().cast::<f32>(),
                hidden0_derivative.as_mut_ptr(),
                hidden0_derivative.len(),
            );
            std::ptr::copy_nonoverlapping(
                derivative1_buffer.contents().cast::<f32>(),
                hidden1_derivative.as_mut_ptr(),
                hidden1_derivative.len(),
            );
            std::ptr::copy_nonoverlapping(
                derivative2_buffer.contents().cast::<f32>(),
                hidden2_derivative.as_mut_ptr(),
                hidden2_derivative.len(),
            );
        }
        Ok(Mp0bConvTpWeightsForwardMetal {
            output,
            hidden0_derivative,
            hidden1_derivative,
            hidden2_derivative,
            metal_cache: Mp0bConvTpWeightCacheMetal {
                hidden_len,
                output_len,
                output_buffer,
                edge_feats_buffer,
                derivative0_buffer,
                derivative1_buffer,
                derivative2_buffer,
                weight0_buffer,
                weight1_buffer,
                weight2_buffer,
                weight3_buffer,
                dense_layer3_vjp_params_buffer,
                dense64_vjp_params_buffer,
                edge_feat_vjp_params_buffer,
                edge_feat_vjp_assign_params_buffer,
                edge_feat_len: edge_feats.len(),
                bar_hidden2_scratch_buffer: self
                    .device
                    .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options),
                bar_hidden1_scratch_buffer: self
                    .device
                    .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options),
                bar_hidden0_scratch_buffer: self
                    .device
                    .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options),
                bar_edge_feats_scratch_buffer: self
                    .device
                    .new_buffer(std::mem::size_of_val(edge_feats) as u64, options),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer1_hidden_linear_vjp_input_density_f32(
        &self,
        density: &[f32],
        linear_out: &[f32],
        linear_out_metal_cache: Option<&Mp0bF32BufferMetal>,
        weight: &[f32],
        output_mask: &[f32],
        static_cache: Option<&Mp0bLayer1HiddenWeightsMetal>,
        bar_out: &[f32],
        bar_input: &mut [f32],
        bar_density: &mut [f32],
        read_bar_input: bool,
        read_bar_density: bool,
        bar_out_feature_major: bool,
    ) -> Result<Option<Mp0bLayer1HiddenVjpMetal>, String> {
        const INPUT_ROW_LEN: usize = 5120;
        const OUTPUT_ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 10 * 128 * 128;
        let num_nodes = density.len();
        let bar_input_len = num_nodes * INPUT_ROW_LEN;
        let linear_out_len = num_nodes * OUTPUT_ROW_LEN;
        if let Some(cache) = linear_out_metal_cache {
            if cache.len != linear_out_len
                || (!linear_out.is_empty() && linear_out.len() != linear_out_len)
            {
                return Err("layer1 hidden VJP linear output Metal shape mismatch".to_owned());
            }
        } else if linear_out.len() != linear_out_len {
            return Err("layer1 hidden VJP shape mismatch".to_owned());
        }
        if bar_out.len() != linear_out_len
            || (read_bar_density && bar_density.len() != num_nodes)
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != OUTPUT_ROW_LEN
        {
            return Err("layer1 hidden VJP shape mismatch".to_owned());
        }
        if read_bar_input && bar_input.len() != bar_input_len {
            return Err("layer1 hidden VJP bar input shape mismatch".to_owned());
        }
        if !read_bar_input && !bar_input.is_empty() && bar_input.len() != bar_input_len {
            return Err("layer1 hidden VJP bar input shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let density_buffer = self.buffer_with_slice(density, options);
        let owned_linear_out_buffer;
        let linear_out_buffer = if let Some(cache) = linear_out_metal_cache {
            &cache.buffer
        } else {
            owned_linear_out_buffer = self.buffer_with_slice(linear_out, options);
            &owned_linear_out_buffer
        };
        let owned_weight_buffer;
        let weight_buffer = if let Some(cache) = static_cache {
            if cache.weight_len != WEIGHT_LEN || cache.output_mask_len != OUTPUT_ROW_LEN {
                return Err("layer1 hidden VJP static cache shape mismatch".to_owned());
            }
            &cache.weight_buffer
        } else {
            owned_weight_buffer = self.buffer_with_slice(weight, options);
            &owned_weight_buffer
        };
        let owned_output_mask_buffer;
        let output_mask_buffer = if let Some(cache) = static_cache {
            &cache.output_mask_buffer
        } else {
            owned_output_mask_buffer = self.buffer_with_slice(output_mask, options);
            &owned_output_mask_buffer
        };
        let bar_out_buffer = self.buffer_with_slice(bar_out, options);
        let bar_input_buffer = self
            .device
            .new_buffer((bar_input_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_density_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(density) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        if bar_out_feature_major {
            encoder
                .set_compute_pipeline_state(&self.layer1_hidden_vjp_density_feature_major_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
        }
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&linear_out_buffer), 0);
        encoder.set_buffer(2, Some(&bar_out_buffer), 0);
        encoder.set_buffer(3, Some(&bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: num_nodes as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                height: 1,
                depth: 1,
            },
        );

        if bar_out_feature_major {
            encoder
                .set_compute_pipeline_state(&self.layer1_hidden_vjp_input_feature_major_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_input_pipeline);
        }
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_out_buffer), 0);
        encoder.set_buffer(4, Some(&bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_input_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();

        let needs_cpu_readback = read_bar_input || read_bar_density;
        if needs_cpu_readback {
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
        }
        // Safety: both output buffers are StorageModeShared and sized exactly
        // from the destination slice lengths. The command buffer completed
        // when CPU readback is requested; otherwise returned Metal buffers
        // stay ordered behind this command buffer on the same queue.
        unsafe {
            if read_bar_input {
                std::ptr::copy_nonoverlapping(
                    bar_input_buffer.contents().cast::<f32>(),
                    bar_input.as_mut_ptr(),
                    bar_input.len(),
                );
            }
            if read_bar_density {
                std::ptr::copy_nonoverlapping(
                    bar_density_buffer.contents().cast::<f32>(),
                    bar_density.as_mut_ptr(),
                    bar_density.len(),
                );
            }
        }
        Ok((!read_bar_input).then_some(Mp0bLayer1HiddenVjpMetal {
            bar_input: Mp0bF32BufferMetal {
                len: bar_input_len,
                buffer: bar_input_buffer,
            },
            bar_density: Mp0bF32BufferMetal {
                len: num_nodes,
                buffer: bar_density_buffer,
            },
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn layer0_hidden_linear_vjp_input_density_f32(
        &self,
        density: &[f32],
        linear_out: &[f32],
        weight: &[f32],
        output_mask: &[f32],
        bar_out: &[f32],
        bar_input: &mut [f32],
        bar_density: &mut [f32],
    ) -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 4 * 128 * 128;
        let num_nodes = density.len();
        if linear_out.len() != num_nodes * ROW_LEN
            || bar_out.len() != linear_out.len()
            || bar_input.len() != linear_out.len()
            || bar_density.len() != num_nodes
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
        {
            return Err("layer0 hidden VJP shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let density_buffer = self.buffer_with_slice(density, options);
        let linear_out_buffer = self.buffer_with_slice(linear_out, options);
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let bar_out_buffer = self.buffer_with_slice(bar_out, options);
        let bar_input_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let bar_density_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_density) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&linear_out_buffer), 0);
        encoder.set_buffer(2, Some(&bar_out_buffer), 0);
        encoder.set_buffer(3, Some(&bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: num_nodes as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer0_hidden_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_out_buffer), 0);
        encoder.set_buffer(4, Some(&bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_input.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer0_hidden_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: both output buffers are StorageModeShared and sized exactly
        // from the destination slice lengths. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_input_buffer.contents().cast::<f32>(),
                bar_input.as_mut_ptr(),
                bar_input.len(),
            );
            std::ptr::copy_nonoverlapping(
                bar_density_buffer.contents().cast::<f32>(),
                bar_density.as_mut_ptr(),
                bar_density.len(),
            );
        }
        Ok(())
    }

    fn hidden_message_selector_vjp_input_from_feature_major_f32(
        &self,
        type_ids: &[u32],
        weight: &[f32],
        output_mask: &[f32],
        bar_out_feature_major: &[f32],
        bar_input: &mut [f32],
    ) -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 4 * 128 * 89 * 128;
        let num_nodes = type_ids.len();
        if weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
            || bar_input.len() != num_nodes * ROW_LEN
            || bar_out_feature_major.len() != bar_input.len()
        {
            return Err("hidden message selector VJP shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let bar_out_buffer = self.buffer_with_slice(bar_out_feature_major, options);
        let bar_input_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.selector_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_out_buffer), 0);
        encoder.set_buffer(4, Some(&bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_input.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.selector_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_input_buffer` is StorageModeShared and was allocated
        // from the destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_input_buffer.contents().cast::<f32>(),
                bar_input.as_mut_ptr(),
                bar_input.len(),
            );
        }
        Ok(())
    }

    fn hidden_message_selector_to_feature_major_f32(
        &self,
        type_ids: &[u32],
        input: &[f32],
        input_metal_cache: Option<&Mp0bF32BufferMetal>,
        weight: &[f32],
        output_mask: &[f32],
        output_feature_major: &mut [f32],
    ) -> Result<Mp0bF32BufferMetal, String> {
        const ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 4 * 128 * 89 * 128;
        let num_nodes = type_ids.len();
        let input_len = if let Some(cache) = input_metal_cache {
            if cache.len != num_nodes * ROW_LEN {
                return Err(
                    "hidden message selector feature-major input Metal cache shape mismatch"
                        .to_owned(),
                );
            }
            if !input.is_empty() && input.len() != cache.len {
                return Err("hidden message selector feature-major shape mismatch".to_owned());
            }
            cache.len
        } else {
            input.len()
        };
        if input_len != num_nodes * ROW_LEN
            || output_feature_major.len() != input_len
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
        {
            return Err("hidden message selector feature-major shape mismatch".to_owned());
        }
        if type_ids.iter().any(|&type_id| type_id as usize >= 89) {
            return Err("hidden message selector feature-major type id out of range".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let owned_input_buffer;
        let input_buffer = if let Some(cache) = input_metal_cache {
            &cache.buffer
        } else {
            owned_input_buffer = self.buffer_with_slice(input, options);
            &owned_input_buffer
        };
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let output_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(output_feature_major) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.selector_to_feature_major_forward_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&output_mask_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: output_feature_major.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.selector_to_feature_major_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `output_buffer` is StorageModeShared and allocated from the
        // destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents().cast::<f32>(),
                output_feature_major.as_mut_ptr(),
                output_feature_major.len(),
            );
        }
        Ok(Mp0bF32BufferMetal {
            len: output_feature_major.len(),
            buffer: output_buffer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer0_selector_hidden_vjp_input_density_f32(
        &self,
        type_ids: &[u32],
        selector_weight: &[f32],
        selector_output_mask: &[f32],
        bar_interaction_out: &[f32],
        density: &[f32],
        linear_out: &[f32],
        hidden_weight: &[f32],
        hidden_output_mask: &[f32],
        bar_input: &mut [f32],
        bar_density: &mut [f32],
    ) -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const SELECTOR_WEIGHT_LEN: usize = 4 * 128 * 89 * 128;
        const HIDDEN_WEIGHT_LEN: usize = 4 * 128 * 128;
        let num_nodes = type_ids.len();
        if selector_weight.len() != SELECTOR_WEIGHT_LEN
            || selector_output_mask.len() != ROW_LEN
            || bar_interaction_out.len() != num_nodes * ROW_LEN
            || density.len() != num_nodes
            || linear_out.len() != num_nodes * ROW_LEN
            || hidden_weight.len() != HIDDEN_WEIGHT_LEN
            || hidden_output_mask.len() != ROW_LEN
            || bar_input.len() != num_nodes * ROW_LEN
            || bar_density.len() != num_nodes
        {
            return Err("layer0 selector/hidden VJP shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let selector_weight_buffer = self.buffer_with_slice(selector_weight, options);
        let selector_output_mask_buffer = self.buffer_with_slice(selector_output_mask, options);
        let bar_interaction_out_buffer = self.buffer_with_slice(bar_interaction_out, options);
        let density_buffer = self.buffer_with_slice(density, options);
        let linear_out_buffer = self.buffer_with_slice(linear_out, options);
        let hidden_weight_buffer = self.buffer_with_slice(hidden_weight, options);
        let hidden_output_mask_buffer = self.buffer_with_slice(hidden_output_mask, options);
        let bar_linear_message_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let bar_input_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let bar_density_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_density) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.selector_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&selector_weight_buffer), 0);
        encoder.set_buffer(2, Some(&selector_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_interaction_out_buffer), 0);
        encoder.set_buffer(4, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_interaction_out.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.selector_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&linear_out_buffer), 0);
        encoder.set_buffer(2, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(3, Some(&bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: num_nodes as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer0_hidden_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&hidden_weight_buffer), 0);
        encoder.set_buffer(2, Some(&hidden_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(4, Some(&bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_input.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer0_hidden_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: both output buffers are StorageModeShared and sized exactly
        // from the destination slice lengths. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_input_buffer.contents().cast::<f32>(),
                bar_input.as_mut_ptr(),
                bar_input.len(),
            );
            std::ptr::copy_nonoverlapping(
                bar_density_buffer.contents().cast::<f32>(),
                bar_density.as_mut_ptr(),
                bar_density.len(),
            );
        }
        Ok(())
    }

    fn scalar_vector_product_linear_forward_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        output_mask: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.scalar_vector_product_linear_forward_with_input_cache_f32(
            input,
            None,
            weight,
            output_mask,
            output,
        )
        .map(|_| ())
    }

    fn scalar_vector_product_linear_forward_with_input_cache_f32(
        &self,
        input: &[f32],
        input_metal_cache: Option<&Mp0bF32BufferMetal>,
        weight: &[f32],
        output_mask: &[f32],
        output: &mut [f32],
    ) -> Result<Mp0bF32BufferMetal, String> {
        const ROW_LEN: usize = 512;
        const WEIGHT_LEN: usize = 2 * 128 * 128;
        let input_len = if let Some(cache) = input_metal_cache {
            if !input.is_empty() && input.len() != cache.len {
                return Err("scalar-vector product linear forward input shape mismatch".to_owned());
            }
            cache.len
        } else {
            input.len()
        };
        if !input_len.is_multiple_of(ROW_LEN)
            || output.len() != input_len
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
        {
            return Err("scalar-vector product linear forward shape mismatch".to_owned());
        }
        let num_nodes = input_len / ROW_LEN;
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let owned_input_buffer;
        let input_buffer = if let Some(cache) = input_metal_cache {
            &cache.buffer
        } else {
            owned_input_buffer = self.buffer_with_slice(input, options);
            &owned_input_buffer
        };
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let output_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(output) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder
            .set_compute_pipeline_state(&self.scalar_vector_product_linear_forward_packed_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * 128) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self
                    .threads_per_group(&self.scalar_vector_product_linear_forward_packed_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `output_buffer` is StorageModeShared and sized exactly from
        // the destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(Mp0bF32BufferMetal {
            len: output.len(),
            buffer: output_buffer,
        })
    }

    fn scalar_vector_product_linear_vjp_input_f32(
        &self,
        weight: &[f32],
        output_mask: &[f32],
        bar_out: &[f32],
        bar_input: &mut [f32],
    ) -> Result<(), String> {
        const ROW_LEN: usize = 512;
        const WEIGHT_LEN: usize = 2 * 128 * 128;
        if !bar_input.len().is_multiple_of(ROW_LEN)
            || bar_out.len() != bar_input.len()
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
        {
            return Err("scalar-vector product linear VJP shape mismatch".to_owned());
        }
        let num_nodes = bar_input.len() / ROW_LEN;
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let bar_out_buffer = self.buffer_with_slice(bar_out, options);
        let bar_input_buffer = self.buffer_with_slice(bar_input, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(
            &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
        );
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&output_mask_buffer), 0);
        encoder.set_buffer(2, Some(&bar_out_buffer), 0);
        encoder.set_buffer(3, Some(&bar_input_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * 128) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(
                    &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
                ),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_input_buffer` is StorageModeShared and was initialized
        // from the destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_input_buffer.contents().cast::<f32>(),
                bar_input.as_mut_ptr(),
                bar_input.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer1_conv_tp_forward_f32(
        &self,
        num_nodes: usize,
        linear_up: &[f32],
        edge_attrs: &[f32],
        tp_weights: &[f32],
        senders: &[usize],
        receivers: &[usize],
        scatter_message: Option<&mut [f32]>,
        tp_weight_cache: Option<&Mp0bConvTpWeightCacheMetal>,
        topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
        read_scatter_message: bool,
        use_oam_pipeline: bool,
    ) -> Result<Mp0bLayer1ConvTpForwardMetal, String> {
        const INPUT_ROW_LEN: usize = 512;
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 1280;
        const OUTPUT_ROW_LEN: usize = 5120;
        let num_edges = senders.len();
        if receivers.len() != num_edges
            || linear_up.len() != num_nodes * INPUT_ROW_LEN
            || edge_attrs.len() != num_edges * EDGE_ATTR_ROW_LEN
            || (tp_weights.len() != num_edges * TP_WEIGHT_ROW_LEN
                && !(tp_weights.is_empty() && tp_weight_cache.is_some()))
        {
            return Err(format!(
                "layer1 conv-TP forward shape mismatch: num_nodes={num_nodes} num_edges={num_edges} linear_up={} expected_linear_up={} edge_attrs={} expected_edge_attrs={} tp_weights={} expected_tp_weights={} has_tp_weight_cache={} pipeline={}",
                linear_up.len(),
                num_nodes * INPUT_ROW_LEN,
                edge_attrs.len(),
                num_edges * EDGE_ATTR_ROW_LEN,
                tp_weights.len(),
                num_edges * TP_WEIGHT_ROW_LEN,
                tp_weight_cache.is_some(),
                if use_oam_pipeline { "oam" } else { "mp0b" }
            ));
        }
        let scatter_len = num_nodes * OUTPUT_ROW_LEN;
        if read_scatter_message && scatter_message.is_none() {
            return Err("layer1 conv-TP forward readback requires output slice".to_owned());
        }
        if let Some(scatter_message) = scatter_message.as_ref() {
            if scatter_message.len() != scatter_len {
                return Err("layer1 conv-TP forward scatter shape mismatch".to_owned());
            }
        }
        let options = MTLResourceOptions::StorageModeShared;
        let owned_topology;
        let topology = if let Some(cache) = topology_cache {
            if cache.num_nodes != num_nodes || cache.num_edges != num_edges {
                return Err("layer1 conv-TP forward topology cache shape mismatch".to_owned());
            }
            cache
        } else {
            owned_topology =
                self.layer1_conv_tp_topology_buffers(num_nodes, senders, receivers, options)?;
            &owned_topology
        };
        let linear_up_buffer = self.buffer_with_slice(linear_up, options);
        let edge_attrs_buffer = self.buffer_with_slice(edge_attrs, options);
        let tp_weights_buffer = if let Some(cache) = tp_weight_cache {
            if !tp_weights.is_empty() && cache.output_len != tp_weights.len() {
                return Err("layer1 conv-TP forward Metal cache shape mismatch".to_owned());
            }
            cache.output_buffer.clone()
        } else {
            self.buffer_with_slice(tp_weights, options)
        };
        let scatter_buffer = self
            .device
            .new_buffer((scatter_len * std::mem::size_of::<f32>()) as u64, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = if use_oam_pipeline {
            &self.oam_layer1_conv_tp_forward_pipeline
        } else {
            &self.layer1_conv_tp_forward_pipeline
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&linear_up_buffer), 0);
        encoder.set_buffer(1, Some(&edge_attrs_buffer), 0);
        encoder.set_buffer(2, Some(&tp_weights_buffer), 0);
        encoder.set_buffer(3, Some(&topology.receiver_offsets_buffer), 0);
        encoder.set_buffer(4, Some(&topology.receiver_edges_buffer), 0);
        encoder.set_buffer(5, Some(&topology.senders_buffer), 0);
        encoder.set_buffer(6, Some(&self.layer1_tp_templates_buffer), 0);
        encoder.set_buffer(7, Some(&self.layer1_tp_forward_cg_offsets_buffer), 0);
        encoder.set_buffer(8, Some(&self.layer1_tp_forward_cgs_buffer), 0);
        encoder.set_buffer(9, Some(&scatter_buffer), 0);
        encoder.set_buffer(10, Some(&topology.forward_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: scatter_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();

        if read_scatter_message {
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            // Safety: `scatter_buffer` is StorageModeShared and sized exactly
            // from the destination slice length. The command buffer completed.
            unsafe {
                if let Some(scatter_message) = scatter_message {
                    std::ptr::copy_nonoverlapping(
                        scatter_buffer.contents().cast::<f32>(),
                        scatter_message.as_mut_ptr(),
                        scatter_message.len(),
                    );
                }
            }
        }
        // Without CPU readback, the returned shared buffer is consumed by later
        // command buffers on the same queue, so queue ordering provides the
        // dependency without a host-side synchronization point.
        Ok(Mp0bLayer1ConvTpForwardMetal {
            topology: topology.clone(),
            scatter_message_metal: Some(Mp0bF32BufferMetal {
                len: scatter_len,
                buffer: scatter_buffer,
            }),
        })
    }

    fn layer1_conv_tp_vjp_fixed_cache_f32(
        &self,
        num_nodes: usize,
        edge_attrs: &[f32],
        layer3_weight: &[f32],
        linear_up_weight: &[f32],
        linear_up_output_mask: &[f32],
    ) -> Result<Mp0bLayer1ConvTpVjpFixedCacheMetal, String> {
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const INPUT_ROW_LEN: usize = 512;
        const TP_WEIGHT_ROW_LEN: usize = 1280;
        const HIDDEN_DIM: usize = 64;
        let num_edges = edge_attrs.len() / EDGE_ATTR_ROW_LEN;
        if edge_attrs.len() != num_edges * EDGE_ATTR_ROW_LEN
            || layer3_weight.len() != HIDDEN_DIM * TP_WEIGHT_ROW_LEN
            || linear_up_weight.len() != 2 * 128 * 128
            || linear_up_output_mask.len() != INPUT_ROW_LEN
        {
            return Err("layer1 conv-TP VJP fixed cache shape mismatch".to_owned());
        }
        let mut layer3_weight_transposed = vec![0.0_f32; layer3_weight.len()];
        for hidden in 0..HIDDEN_DIM {
            for lane in 0..TP_WEIGHT_ROW_LEN {
                layer3_weight_transposed[lane * HIDDEN_DIM + hidden] =
                    layer3_weight[hidden * TP_WEIGHT_ROW_LEN + lane];
            }
        }
        let linear_up_vjp_params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };
        let options = MTLResourceOptions::StorageModeShared;
        Ok(Mp0bLayer1ConvTpVjpFixedCacheMetal {
            num_nodes,
            num_edges,
            edge_attrs_buffer: self.buffer_with_slice(edge_attrs, options),
            layer3_weight_buffer: self.buffer_with_slice(&layer3_weight_transposed, options),
            linear_up_weight_buffer: self.buffer_with_slice(linear_up_weight, options),
            linear_up_output_mask_buffer: self.buffer_with_slice(linear_up_output_mask, options),
            bar_linear_up_buffer: self.device.new_buffer(
                (num_nodes * INPUT_ROW_LEN * std::mem::size_of::<f32>()) as u64,
                options,
            ),
            bar_edge_attrs_buffer: self.device.new_buffer(
                (num_edges * EDGE_ATTR_ROW_LEN * std::mem::size_of::<f32>()) as u64,
                options,
            ),
            bar_hidden2_buffer: self.device.new_buffer(
                (num_edges * HIDDEN_DIM * std::mem::size_of::<f32>()) as u64,
                options,
            ),
            linear_up_vjp_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&linear_up_vjp_params), options),
        })
    }

    fn layer0_conv_tp_vjp_fixed_cache_f32(
        &self,
        num_nodes: usize,
        edge_attrs: &[f32],
        layer3_weight: &[f32],
    ) -> Result<Mp0bLayer0ConvTpVjpFixedCacheMetal, String> {
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 4 * 128;
        const HIDDEN_DIM: usize = 64;
        let num_edges = edge_attrs.len() / EDGE_ATTR_ROW_LEN;
        if edge_attrs.len() != num_edges * EDGE_ATTR_ROW_LEN
            || layer3_weight.len() != HIDDEN_DIM * TP_WEIGHT_ROW_LEN
        {
            return Err("layer0 conv-TP VJP fixed cache shape mismatch".to_owned());
        }
        let mut layer3_weight_transposed = vec![0.0_f32; layer3_weight.len()];
        for hidden in 0..HIDDEN_DIM {
            for lane in 0..TP_WEIGHT_ROW_LEN {
                layer3_weight_transposed[lane * HIDDEN_DIM + hidden] =
                    layer3_weight[hidden * TP_WEIGHT_ROW_LEN + lane];
            }
        }
        let options = MTLResourceOptions::StorageModeShared;
        Ok(Mp0bLayer0ConvTpVjpFixedCacheMetal {
            num_nodes,
            num_edges,
            edge_attrs_buffer: self.buffer_with_slice(edge_attrs, options),
            layer3_weight_buffer: self.buffer_with_slice(&layer3_weight_transposed, options),
            bar_edge_attrs_buffer: self.device.new_buffer(
                (num_edges * EDGE_ATTR_ROW_LEN * std::mem::size_of::<f32>()) as u64,
                options,
            ),
            bar_hidden2_buffer: self.device.new_buffer(
                (num_edges * HIDDEN_DIM * std::mem::size_of::<f32>()) as u64,
                options,
            ),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer0_conv_tp_vjp_hidden2_f32(
        &self,
        num_nodes: usize,
        linear_up: &[f32],
        tp_weights: &[f32],
        tp_weight_cache: &Mp0bConvTpWeightCacheMetal,
        topology: &Mp0bLayer1ConvTpTopologyMetal,
        fixed_cache: &Mp0bLayer0ConvTpVjpFixedCacheMetal,
        bar_scatter_message: &[f32],
        bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
        accumulate_bar_edge_attrs_metal: Option<&Mp0bF32BufferMetal>,
        read_bar_edge_attrs: bool,
    ) -> Result<Mp0bLayer0ConvTpVjpHidden2Metal, String> {
        const INPUT_ROW_LEN: usize = 128;
        const OUTPUT_ROW_LEN: usize = 2048;
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 4 * 128;
        const HIDDEN_DIM: usize = 64;
        let num_edges = topology.num_edges;
        if topology.num_nodes != num_nodes
            || fixed_cache.num_nodes != num_nodes
            || fixed_cache.num_edges != num_edges
            || linear_up.len() != num_nodes * INPUT_ROW_LEN
            || (!tp_weights.is_empty() && tp_weights.len() != num_edges * TP_WEIGHT_ROW_LEN)
            || tp_weight_cache.output_len != num_edges * TP_WEIGHT_ROW_LEN
        {
            return Err("layer0 conv-TP VJP Metal shape mismatch".to_owned());
        }
        let bar_edge_attrs_len = num_edges * EDGE_ATTR_ROW_LEN;
        if let Some(accumulate_target) = accumulate_bar_edge_attrs_metal {
            if accumulate_target.len != bar_edge_attrs_len {
                return Err("layer0 conv-TP VJP accumulate edge attrs shape mismatch".to_owned());
            }
        }
        let bar_scatter_len = num_nodes * OUTPUT_ROW_LEN;
        if let Some(cache) = bar_scatter_message_metal {
            if cache.len != bar_scatter_len {
                return Err("layer0 conv-TP VJP bar scatter Metal cache shape mismatch".to_owned());
            }
            if !bar_scatter_message.is_empty() && bar_scatter_message.len() != bar_scatter_len {
                return Err("layer0 conv-TP VJP bar scatter shape mismatch".to_owned());
            }
        } else if bar_scatter_message.len() != bar_scatter_len {
            return Err("layer0 conv-TP VJP bar scatter shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let linear_up_buffer = self.buffer_with_slice(linear_up, options);
        let owned_bar_scatter_buffer;
        let bar_scatter_buffer = if let Some(cache) = bar_scatter_message_metal {
            &cache.buffer
        } else {
            owned_bar_scatter_buffer = self.buffer_with_slice(bar_scatter_message, options);
            &owned_bar_scatter_buffer
        };
        let tp_weights_buffer = tp_weight_cache.output_buffer.clone();
        let bar_edge_attrs_buffer = if let Some(accumulate_target) = accumulate_bar_edge_attrs_metal
        {
            &accumulate_target.buffer
        } else {
            &fixed_cache.bar_edge_attrs_buffer
        };
        let owned_accumulate_vjp_params_buffer;
        let vjp_params_buffer = if accumulate_bar_edge_attrs_metal.is_some() {
            let vjp_params = Mp0bLayer1ConvTpVjpParams {
                num_nodes: u32::try_from(num_nodes)
                    .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
                num_edges: u32::try_from(num_edges)
                    .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?,
                flags: MP0B_CONV_TP_VJP_ACCUMULATE_EDGE_ATTRS_FLAG,
            };
            owned_accumulate_vjp_params_buffer =
                self.buffer_with_slice(std::slice::from_ref(&vjp_params), options);
            &owned_accumulate_vjp_params_buffer
        } else {
            &topology.vjp_params_buffer
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layer0_conv_tp_vjp_edge_attrs_pipeline);
        encoder.set_buffer(0, Some(&linear_up_buffer), 0);
        encoder.set_buffer(1, Some(&tp_weights_buffer), 0);
        encoder.set_buffer(2, Some(&bar_scatter_buffer), 0);
        encoder.set_buffer(3, Some(&topology.senders_buffer), 0);
        encoder.set_buffer(4, Some(&topology.receivers_buffer), 0);
        encoder.set_buffer(5, Some(bar_edge_attrs_buffer), 0);
        encoder.set_buffer(6, Some(vjp_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: 128,
                height: num_edges as u64,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer0_conv_tp_vjp_hidden2_fused_tiled_pipeline);
        encoder.set_buffer(0, Some(&linear_up_buffer), 0);
        encoder.set_buffer(1, Some(&fixed_cache.edge_attrs_buffer), 0);
        encoder.set_buffer(2, Some(&fixed_cache.layer3_weight_buffer), 0);
        encoder.set_buffer(3, Some(&bar_scatter_buffer), 0);
        encoder.set_buffer(4, Some(&topology.senders_buffer), 0);
        encoder.set_buffer(5, Some(&topology.receivers_buffer), 0);
        encoder.set_buffer(6, Some(&fixed_cache.bar_hidden2_buffer), 0);
        encoder.set_buffer(7, Some(vjp_params_buffer), 0);
        let tiled_edge_grid = num_edges.div_ceil(4) * 4;
        encoder.dispatch_threads(
            MTLSize {
                width: HIDDEN_DIM as u64,
                height: tiled_edge_grid as u64,
                depth: 1,
            },
            MTLSize {
                width: HIDDEN_DIM as u64,
                height: 4,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        if read_bar_edge_attrs {
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
        }
        let mut bar_edge_attrs = if read_bar_edge_attrs {
            vec![0.0_f32; bar_edge_attrs_len]
        } else {
            Vec::new()
        };
        // Safety: the output buffer is StorageModeShared, sized from
        // `num_edges * EDGE_ATTR_ROW_LEN`. The command buffer completed when
        // CPU readback is requested; otherwise the returned Metal buffers stay
        // ordered behind this command buffer on the same queue.
        unsafe {
            if read_bar_edge_attrs {
                std::ptr::copy_nonoverlapping(
                    bar_edge_attrs_buffer.contents().cast::<f32>(),
                    bar_edge_attrs.as_mut_ptr(),
                    bar_edge_attrs.len(),
                );
            }
        }
        Ok(Mp0bLayer0ConvTpVjpHidden2Metal {
            bar_edge_attrs,
            bar_edge_attrs_metal: Some(Mp0bF32BufferMetal {
                len: bar_edge_attrs_len,
                buffer: (*bar_edge_attrs_buffer).clone(),
            }),
            bar_hidden2_metal: Mp0bF32BufferMetal {
                len: num_edges * HIDDEN_DIM,
                buffer: fixed_cache.bar_hidden2_buffer.clone(),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer1_conv_tp_vjp_hidden2_f32(
        &self,
        num_nodes: usize,
        linear_up: &[f32],
        edge_attrs: &[f32],
        tp_weights: &[f32],
        layer3_weight: &[f32],
        bar_scatter_message: &[f32],
        bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
        senders: &[usize],
        receivers: &[usize],
        linear_up_vjp: Option<(&[f32], &[f32], &[f32])>,
        tp_weight_cache: Option<&Mp0bConvTpWeightCacheMetal>,
        topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
        fixed_cache: Option<&Mp0bLayer1ConvTpVjpFixedCacheMetal>,
        read_bar_edge_attrs: bool,
        read_bar_hidden2: bool,
        read_bar_layer0_product_out: bool,
    ) -> Result<Mp0bLayer1ConvTpVjpHidden2Metal, String> {
        const INPUT_ROW_LEN: usize = 512;
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 1280;
        const OUTPUT_ROW_LEN: usize = 5120;
        const HIDDEN_DIM: usize = 64;
        let num_edges = senders.len();
        if receivers.len() != num_edges
            || linear_up.len() != num_nodes * INPUT_ROW_LEN
            || edge_attrs.len() != num_edges * EDGE_ATTR_ROW_LEN
            || (tp_weights.len() != num_edges * TP_WEIGHT_ROW_LEN
                && !(tp_weights.is_empty() && tp_weight_cache.is_some()))
            || layer3_weight.len() != HIDDEN_DIM * TP_WEIGHT_ROW_LEN
        {
            return Err("layer1 conv-TP VJP shape mismatch".to_owned());
        }
        let bar_scatter_len = num_nodes * OUTPUT_ROW_LEN;
        if let Some(cache) = bar_scatter_message_metal {
            if cache.len != bar_scatter_len {
                return Err("layer1 conv-TP VJP bar scatter Metal cache shape mismatch".to_owned());
            }
            if !bar_scatter_message.is_empty() && bar_scatter_message.len() != bar_scatter_len {
                return Err("layer1 conv-TP VJP bar scatter shape mismatch".to_owned());
            }
        } else if bar_scatter_message.len() != bar_scatter_len {
            return Err("layer1 conv-TP VJP bar scatter shape mismatch".to_owned());
        }
        if let Some((weight, output_mask, bar_input)) = linear_up_vjp {
            if weight.len() != 2 * 128 * 128
                || output_mask.len() != INPUT_ROW_LEN
                || bar_input.len() != linear_up.len()
            {
                return Err("layer1 conv-TP fused linear-up VJP shape mismatch".to_owned());
            }
        }
        let read_bar_linear_up = linear_up_vjp.is_none();
        let mut bar_linear_up = if read_bar_linear_up {
            vec![0.0_f32; linear_up.len()]
        } else {
            Vec::new()
        };
        let bar_edge_attrs_len = edge_attrs.len();
        let mut bar_edge_attrs = if read_bar_edge_attrs {
            vec![0.0_f32; bar_edge_attrs_len]
        } else {
            Vec::new()
        };
        let bar_hidden2_len = num_edges * HIDDEN_DIM;
        let mut bar_hidden2 = if read_bar_hidden2 {
            vec![0.0_f32; bar_hidden2_len]
        } else {
            Vec::new()
        };

        let options = MTLResourceOptions::StorageModeShared;
        let owned_topology;
        let topology = if let Some(cache) = topology_cache {
            if cache.num_nodes != num_nodes || cache.num_edges != num_edges {
                return Err("layer1 conv-TP VJP topology cache shape mismatch".to_owned());
            }
            cache
        } else {
            owned_topology =
                self.layer1_conv_tp_topology_buffers(num_nodes, senders, receivers, options)?;
            &owned_topology
        };
        if let Some(cache) = fixed_cache {
            if cache.num_nodes != num_nodes || cache.num_edges != num_edges {
                return Err("layer1 conv-TP VJP fixed cache shape mismatch".to_owned());
            }
        }
        let linear_up_buffer = self.buffer_with_slice(linear_up, options);
        let owned_edge_attrs_buffer;
        let edge_attrs_buffer = if let Some(cache) = fixed_cache {
            &cache.edge_attrs_buffer
        } else {
            owned_edge_attrs_buffer = self.buffer_with_slice(edge_attrs, options);
            &owned_edge_attrs_buffer
        };
        let tp_weights_buffer = if let Some(cache) = tp_weight_cache {
            if !tp_weights.is_empty() && cache.output_len != tp_weights.len() {
                return Err("layer1 conv-TP VJP Metal cache shape mismatch".to_owned());
            }
            cache.output_buffer.clone()
        } else {
            self.buffer_with_slice(tp_weights, options)
        };
        let owned_layer3_weight_buffer;
        let layer3_weight_buffer = if let Some(cache) = fixed_cache {
            &cache.layer3_weight_buffer
        } else {
            let mut layer3_weight_transposed = vec![0.0_f32; layer3_weight.len()];
            for hidden in 0..HIDDEN_DIM {
                for lane in 0..TP_WEIGHT_ROW_LEN {
                    layer3_weight_transposed[lane * HIDDEN_DIM + hidden] =
                        layer3_weight[hidden * TP_WEIGHT_ROW_LEN + lane];
                }
            }
            owned_layer3_weight_buffer = self.buffer_with_slice(&layer3_weight_transposed, options);
            &owned_layer3_weight_buffer
        };
        let owned_bar_scatter_buffer;
        let bar_scatter_buffer = if let Some(cache) = bar_scatter_message_metal {
            &cache.buffer
        } else {
            owned_bar_scatter_buffer = self.buffer_with_slice(bar_scatter_message, options);
            &owned_bar_scatter_buffer
        };
        let owned_bar_linear_up_buffer;
        let bar_linear_up_buffer = if let Some(cache) = fixed_cache {
            &cache.bar_linear_up_buffer
        } else {
            owned_bar_linear_up_buffer = self
                .device
                .new_buffer(std::mem::size_of_val(linear_up) as u64, options);
            &owned_bar_linear_up_buffer
        };
        let owned_bar_edge_attrs_buffer;
        let bar_edge_attrs_buffer = if let Some(cache) = fixed_cache {
            &cache.bar_edge_attrs_buffer
        } else {
            owned_bar_edge_attrs_buffer = self
                .device
                .new_buffer(std::mem::size_of_val(edge_attrs) as u64, options);
            &owned_bar_edge_attrs_buffer
        };
        let owned_bar_hidden2_buffer;
        let bar_hidden2_buffer = if let Some(cache) = fixed_cache {
            &cache.bar_hidden2_buffer
        } else {
            owned_bar_hidden2_buffer = self.device.new_buffer(
                (bar_hidden2_len * std::mem::size_of::<f32>()) as u64,
                options,
            );
            &owned_bar_hidden2_buffer
        };
        let owned_linear_up_vjp_params_buffer;
        let linear_up_vjp_params_buffer = if let Some(cache) = fixed_cache {
            &cache.linear_up_vjp_params_buffer
        } else {
            let linear_up_vjp_params = Mp0bLayer1HiddenVjpParams {
                num_nodes: u32::try_from(topology.num_nodes)
                    .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            };
            owned_linear_up_vjp_params_buffer =
                self.buffer_with_slice(std::slice::from_ref(&linear_up_vjp_params), options);
            &owned_linear_up_vjp_params_buffer
        };
        let mut fused_bar_layer0_product_out = None;
        let mut fused_bar_layer0_product_out_metal = None;
        let linear_up_vjp_buffers = if let Some((weight, output_mask, initial_bar_input)) =
            linear_up_vjp
        {
            let weight_buffer = fixed_cache.map_or_else(
                || self.buffer_with_slice(weight, options),
                |cache| cache.linear_up_weight_buffer.clone(),
            );
            let output_mask_buffer = fixed_cache.map_or_else(
                || self.buffer_with_slice(output_mask, options),
                |cache| cache.linear_up_output_mask_buffer.clone(),
            );
            let bar_layer0_product_out_buffer = self.buffer_with_slice(initial_bar_input, options);
            let buffers = (
                weight_buffer,
                output_mask_buffer,
                bar_layer0_product_out_buffer.clone(),
            );
            if read_bar_layer0_product_out {
                fused_bar_layer0_product_out = Some(initial_bar_input.to_vec());
            }
            fused_bar_layer0_product_out_metal = Some(Mp0bF32BufferMetal {
                len: initial_bar_input.len(),
                buffer: bar_layer0_product_out_buffer,
            });
            Some(buffers)
        } else {
            None
        };
        if mp0b_trace_vjp_kernel_timings_enabled() {
            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layer1_conv_tp_vjp_linear_up_pipeline);
            encoder.set_buffer(0, Some(&edge_attrs_buffer), 0);
            encoder.set_buffer(1, Some(&tp_weights_buffer), 0);
            encoder.set_buffer(2, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(3, Some(&topology.sender_offsets_buffer), 0);
            encoder.set_buffer(4, Some(&topology.sender_edges_buffer), 0);
            encoder.set_buffer(5, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_vjp_i_cg_offsets_buffer), 0);
            encoder.set_buffer(8, Some(&self.layer1_tp_vjp_i_cgs_buffer), 0);
            encoder.set_buffer(9, Some(&self.layer1_tp_input_lane_offsets_buffer), 0);
            encoder.set_buffer(10, Some(&self.layer1_tp_input_lane_entries_buffer), 0);
            encoder.set_buffer(11, Some(bar_linear_up_buffer), 0);
            encoder.set_buffer(12, Some(&topology.vjp_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: linear_up.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.layer1_conv_tp_vjp_linear_up_pipeline),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal layer1_conv_tp_vjp_linear_up_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layer1_conv_tp_vjp_edge_attrs_pipeline);
            encoder.set_buffer(0, Some(&linear_up_buffer), 0);
            encoder.set_buffer(1, Some(&tp_weights_buffer), 0);
            encoder.set_buffer(2, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(3, Some(&topology.senders_buffer), 0);
            encoder.set_buffer(4, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(5, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_cg_offsets_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_cgs_buffer), 0);
            encoder.set_buffer(8, Some(bar_edge_attrs_buffer), 0);
            encoder.set_buffer(9, Some(&topology.vjp_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: 128,
                    height: num_edges as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal layer1_conv_tp_vjp_edge_attrs_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&self.layer1_conv_tp_vjp_hidden2_fused_tiled_pipeline);
            encoder.set_buffer(0, Some(&linear_up_buffer), 0);
            encoder.set_buffer(1, Some(&edge_attrs_buffer), 0);
            encoder.set_buffer(2, Some(&layer3_weight_buffer), 0);
            encoder.set_buffer(3, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(4, Some(&topology.senders_buffer), 0);
            encoder.set_buffer(5, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_cg_offsets_buffer), 0);
            encoder.set_buffer(8, Some(&self.layer1_tp_cgs_buffer), 0);
            encoder.set_buffer(9, Some(&self.layer1_tp_weight_lanes_buffer), 0);
            encoder.set_buffer(10, Some(bar_hidden2_buffer), 0);
            encoder.set_buffer(11, Some(&topology.vjp_params_buffer), 0);
            let tiled_edge_grid = num_edges.div_ceil(4) * 4;
            encoder.dispatch_threads(
                MTLSize {
                    width: HIDDEN_DIM as u64,
                    height: tiled_edge_grid as u64,
                    depth: 1,
                },
                MTLSize {
                    width: HIDDEN_DIM as u64,
                    height: 4,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal layer1_conv_tp_vjp_hidden2_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            if let Some((weight_buffer, output_mask_buffer, bar_layer0_product_out_buffer)) =
                &linear_up_vjp_buffers
            {
                let start = Instant::now();
                let command_buffer = self.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(
                    &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
                );
                encoder.set_buffer(0, Some(weight_buffer), 0);
                encoder.set_buffer(1, Some(output_mask_buffer), 0);
                encoder.set_buffer(2, Some(bar_linear_up_buffer), 0);
                encoder.set_buffer(3, Some(bar_layer0_product_out_buffer), 0);
                encoder.set_buffer(4, Some(linear_up_vjp_params_buffer), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: (num_nodes * 128) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.threads_per_group(
                            &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
                        ),
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let status = command_buffer.status();
                if status != metal::MTLCommandBufferStatus::Completed {
                    return Err(format!("Metal command buffer did not complete: {status:?}"));
                }
                eprintln!(
                    "mace_mp0b_metal layer1_conv_tp_vjp_linear_up_vjp_kernel_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1.0e3
                );
            }
        } else {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layer1_conv_tp_vjp_linear_up_pipeline);
            encoder.set_buffer(0, Some(&edge_attrs_buffer), 0);
            encoder.set_buffer(1, Some(&tp_weights_buffer), 0);
            encoder.set_buffer(2, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(3, Some(&topology.sender_offsets_buffer), 0);
            encoder.set_buffer(4, Some(&topology.sender_edges_buffer), 0);
            encoder.set_buffer(5, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_vjp_i_cg_offsets_buffer), 0);
            encoder.set_buffer(8, Some(&self.layer1_tp_vjp_i_cgs_buffer), 0);
            encoder.set_buffer(9, Some(&self.layer1_tp_input_lane_offsets_buffer), 0);
            encoder.set_buffer(10, Some(&self.layer1_tp_input_lane_entries_buffer), 0);
            encoder.set_buffer(11, Some(bar_linear_up_buffer), 0);
            encoder.set_buffer(12, Some(&topology.vjp_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: linear_up.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.layer1_conv_tp_vjp_linear_up_pipeline),
                    height: 1,
                    depth: 1,
                },
            );

            encoder.set_compute_pipeline_state(&self.layer1_conv_tp_vjp_edge_attrs_pipeline);
            encoder.set_buffer(0, Some(&linear_up_buffer), 0);
            encoder.set_buffer(1, Some(&tp_weights_buffer), 0);
            encoder.set_buffer(2, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(3, Some(&topology.senders_buffer), 0);
            encoder.set_buffer(4, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(5, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_cg_offsets_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_cgs_buffer), 0);
            encoder.set_buffer(8, Some(bar_edge_attrs_buffer), 0);
            encoder.set_buffer(9, Some(&topology.vjp_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: 128,
                    height: num_edges as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );

            encoder
                .set_compute_pipeline_state(&self.layer1_conv_tp_vjp_hidden2_fused_tiled_pipeline);
            encoder.set_buffer(0, Some(&linear_up_buffer), 0);
            encoder.set_buffer(1, Some(&edge_attrs_buffer), 0);
            encoder.set_buffer(2, Some(&layer3_weight_buffer), 0);
            encoder.set_buffer(3, Some(bar_scatter_buffer), 0);
            encoder.set_buffer(4, Some(&topology.senders_buffer), 0);
            encoder.set_buffer(5, Some(&topology.receivers_buffer), 0);
            encoder.set_buffer(6, Some(&self.layer1_tp_templates_buffer), 0);
            encoder.set_buffer(7, Some(&self.layer1_tp_cg_offsets_buffer), 0);
            encoder.set_buffer(8, Some(&self.layer1_tp_cgs_buffer), 0);
            encoder.set_buffer(9, Some(&self.layer1_tp_weight_lanes_buffer), 0);
            encoder.set_buffer(10, Some(bar_hidden2_buffer), 0);
            encoder.set_buffer(11, Some(&topology.vjp_params_buffer), 0);
            let tiled_edge_grid = num_edges.div_ceil(4) * 4;
            encoder.dispatch_threads(
                MTLSize {
                    width: HIDDEN_DIM as u64,
                    height: tiled_edge_grid as u64,
                    depth: 1,
                },
                MTLSize {
                    width: HIDDEN_DIM as u64,
                    height: 4,
                    depth: 1,
                },
            );
            if let Some((weight_buffer, output_mask_buffer, bar_layer0_product_out_buffer)) =
                &linear_up_vjp_buffers
            {
                encoder.set_compute_pipeline_state(
                    &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
                );
                encoder.set_buffer(0, Some(weight_buffer), 0);
                encoder.set_buffer(1, Some(output_mask_buffer), 0);
                encoder.set_buffer(2, Some(bar_linear_up_buffer), 0);
                encoder.set_buffer(3, Some(bar_layer0_product_out_buffer), 0);
                encoder.set_buffer(4, Some(linear_up_vjp_params_buffer), 0);
                encoder.dispatch_threads(
                    MTLSize {
                        width: (num_nodes * 128) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.threads_per_group(
                            &self.scalar_vector_product_linear_vjp_input_packed_pipeline,
                        ),
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            command_buffer.commit();

            let needs_cpu_readback = read_bar_linear_up
                || read_bar_edge_attrs
                || read_bar_hidden2
                || fused_bar_layer0_product_out.is_some();
            if needs_cpu_readback {
                command_buffer.wait_until_completed();

                let status = command_buffer.status();
                if status != metal::MTLCommandBufferStatus::Completed {
                    return Err(format!("Metal command buffer did not complete: {status:?}"));
                }
            }
        }
        // Safety: the output buffers are StorageModeShared and sized exactly
        // from the destination slice lengths. The command buffer completed.
        unsafe {
            if read_bar_linear_up {
                std::ptr::copy_nonoverlapping(
                    bar_linear_up_buffer.contents().cast::<f32>(),
                    bar_linear_up.as_mut_ptr(),
                    bar_linear_up.len(),
                );
            }
            if read_bar_edge_attrs {
                std::ptr::copy_nonoverlapping(
                    bar_edge_attrs_buffer.contents().cast::<f32>(),
                    bar_edge_attrs.as_mut_ptr(),
                    bar_edge_attrs.len(),
                );
            }
            if read_bar_hidden2 {
                std::ptr::copy_nonoverlapping(
                    bar_hidden2_buffer.contents().cast::<f32>(),
                    bar_hidden2.as_mut_ptr(),
                    bar_hidden2.len(),
                );
            }
            if let (Some(output), Some((_, _, bar_layer0_product_out_buffer))) = (
                fused_bar_layer0_product_out.as_mut(),
                linear_up_vjp_buffers.as_ref(),
            ) {
                std::ptr::copy_nonoverlapping(
                    bar_layer0_product_out_buffer.contents().cast::<f32>(),
                    output.as_mut_ptr(),
                    output.len(),
                );
            }
        }

        Ok(Mp0bLayer1ConvTpVjpHidden2Metal {
            bar_linear_up,
            bar_edge_attrs,
            bar_edge_attrs_metal: Some(Mp0bF32BufferMetal {
                len: bar_edge_attrs_len,
                buffer: (*bar_edge_attrs_buffer).clone(),
            }),
            bar_hidden2,
            bar_hidden2_metal: Some(Mp0bF32BufferMetal {
                len: bar_hidden2_len,
                buffer: (*bar_hidden2_buffer).clone(),
            }),
            bar_layer0_product_out: fused_bar_layer0_product_out,
            bar_layer0_product_out_metal: fused_bar_layer0_product_out_metal,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer1_hidden_forward_f32(
        &self,
        input: &[f32],
        input_metal_cache: Option<&Mp0bF32BufferMetal>,
        density: &[f32],
        weight: &[f32],
        output_mask: &[f32],
        static_cache: Option<&Mp0bLayer1HiddenWeightsMetal>,
        output: &mut [f32],
    ) -> Result<Mp0bLayer1HiddenForwardMetal, String> {
        const INPUT_ROW_LEN: usize = 5120;
        const OUTPUT_ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 10 * 128 * 128;
        let num_nodes = density.len();
        let expected_input_len = num_nodes * INPUT_ROW_LEN;
        if weight.len() != WEIGHT_LEN
            || output_mask.len() != OUTPUT_ROW_LEN
            || output.len() != num_nodes * OUTPUT_ROW_LEN
        {
            return Err("layer1 hidden forward shape mismatch".to_owned());
        }
        if let Some(cache) = input_metal_cache {
            if cache.len != expected_input_len {
                return Err("layer1 hidden forward input Metal cache shape mismatch".to_owned());
            }
            if !input.is_empty() && input.len() != expected_input_len {
                return Err("layer1 hidden forward input shape mismatch".to_owned());
            }
        } else if input.len() != expected_input_len {
            return Err("layer1 hidden forward input shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };
        let options = MTLResourceOptions::StorageModeShared;
        let owned_input_buffer;
        let input_buffer = if let Some(cache) = input_metal_cache {
            &cache.buffer
        } else {
            owned_input_buffer = self.buffer_with_slice(input, options);
            &owned_input_buffer
        };
        let density_buffer = self.buffer_with_slice(density, options);
        let owned_weight_buffer;
        let weight_buffer = if let Some(cache) = static_cache {
            if cache.weight_len != WEIGHT_LEN || cache.output_mask_len != OUTPUT_ROW_LEN {
                return Err("layer1 hidden forward static cache shape mismatch".to_owned());
            }
            &cache.weight_buffer
        } else {
            owned_weight_buffer = self.buffer_with_slice(weight, options);
            &owned_weight_buffer
        };
        let owned_output_mask_buffer;
        let output_mask_buffer = if let Some(cache) = static_cache {
            &cache.output_mask_buffer
        } else {
            owned_output_mask_buffer = self.buffer_with_slice(output_mask, options);
            &owned_output_mask_buffer
        };
        let output_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(output) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layer1_hidden_forward_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(&density_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&output_mask_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: output.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `output_buffer` is StorageModeShared and sized exactly from
        // the destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(Mp0bLayer1HiddenForwardMetal {
            linear_message: Mp0bF32BufferMetal {
                len: output.len(),
                buffer: output_buffer,
            },
            density: Mp0bF32BufferMetal {
                len: density.len(),
                buffer: density_buffer,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn vector_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        grad_out: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        num_elements: usize,
        num_features: usize,
        num_equivariance: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        let num_nodes = type_ids.len();
        if num_ell != 16
            || num_features == 0
            || num_equivariance == 0
            || x.len() != num_nodes * num_features * num_ell
            || grad_x.len() != x.len()
            || grad_out.len() != num_nodes * num_features * num_equivariance
            || weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
            || u1_entries_by_equiv.len() != num_equivariance
            || u2_entries_by_equiv.len() != num_equivariance
            || u3_entries_by_equiv.len() != num_equivariance
        {
            return Err("vector contraction VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("vector contraction VJP type id out of range".to_owned());
        }
        let (u1_offsets, u1_entries) = pack_vector_u1_entries(u1_entries_by_equiv)?;
        let (u2_offsets, u2_entries) = pack_vector_u2_entries(u2_entries_by_equiv)?;
        let (u3_offsets, u3_entries) = pack_vector_u3_entries(u3_entries_by_equiv)?;
        let params = Mp0bProduct0VectorVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_equivariance: u32::try_from(num_equivariance)
                .map_err(|_| "equivariance count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "u1 param count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "u2 param count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "u3 param count exceeds Metal u32 range".to_owned())?,
            output_stride: 0,
            output_offset: 0,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let grad_out_buffer = self.buffer_with_slice(grad_out, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_offsets_buffer = self.buffer_with_slice(&u1_offsets, options);
        let u2_offsets_buffer = self.buffer_with_slice(&u2_offsets, options);
        let u3_offsets_buffer = self.buffer_with_slice(&u3_offsets, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_vector_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&u1_offsets_buffer), 0);
        encoder.set_buffer(7, Some(&u2_offsets_buffer), 0);
        encoder.set_buffer(8, Some(&u3_offsets_buffer), 0);
        encoder.set_buffer(9, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(10, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(11, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(12, Some(&grad_x_buffer), 0);
        encoder.set_buffer(13, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * num_features) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_vector_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The command
        // buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_vector_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        grad_out: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        self.vector_contraction_vjp_f32(
            type_ids,
            x,
            grad_out,
            weights_u1,
            weights_u2,
            weights_u3,
            u1_entries_by_equiv,
            u2_entries_by_equiv,
            u3_entries_by_equiv,
            89,
            128,
            3,
            16,
            1,
            6,
            51,
            grad_x,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn vector_contraction_forward_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        num_elements: usize,
        num_features: usize,
        num_equivariance: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        output_stride: usize,
        output_offset: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let num_nodes = type_ids.len();
        let output_row_len = num_features
            .checked_mul(num_equivariance)
            .ok_or_else(|| "vector contraction forward output row overflow".to_owned())?;
        if num_ell != 16
            || num_features == 0
            || num_equivariance == 0
            || output_offset + output_row_len > output_stride
            || x.len() != num_nodes * num_features * num_ell
            || out.len() != num_nodes * output_stride
            || weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
            || u1_entries_by_equiv.len() != num_equivariance
            || u2_entries_by_equiv.len() != num_equivariance
            || u3_entries_by_equiv.len() != num_equivariance
        {
            return Err("vector contraction forward shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("vector contraction forward type id out of range".to_owned());
        }
        let (u1_offsets, u1_entries) = pack_vector_u1_entries(u1_entries_by_equiv)?;
        let (u2_offsets, u2_entries) = pack_vector_u2_entries(u2_entries_by_equiv)?;
        let (u3_offsets, u3_entries) = pack_vector_u3_entries(u3_entries_by_equiv)?;
        let params = Mp0bProduct0VectorVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_equivariance: u32::try_from(num_equivariance)
                .map_err(|_| "equivariance count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "u1 param count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "u2 param count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "u3 param count exceeds Metal u32 range".to_owned())?,
            output_stride: u32::try_from(output_stride)
                .map_err(|_| "output stride exceeds Metal u32 range".to_owned())?,
            output_offset: u32::try_from(output_offset)
                .map_err(|_| "output offset exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_offsets_buffer = self.buffer_with_slice(&u1_offsets, options);
        let u2_offsets_buffer = self.buffer_with_slice(&u2_offsets, options);
        let u3_offsets_buffer = self.buffer_with_slice(&u3_offsets, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let out_buffer = self.buffer_with_slice(out, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_vector_forward_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(5, Some(&u1_offsets_buffer), 0);
        encoder.set_buffer(6, Some(&u2_offsets_buffer), 0);
        encoder.set_buffer(7, Some(&u3_offsets_buffer), 0);
        encoder.set_buffer(8, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(9, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(10, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(11, Some(&out_buffer), 0);
        encoder.set_buffer(12, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * output_row_len) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_vector_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `out_buffer` is StorageModeShared and initialized from the
        // destination slice length. The command buffer completed before
        // readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.contents().cast::<f32>(),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_scalar_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        grad_out: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_ELEMENTS: usize = 89;
        const U1_PARAMS: usize = 1;
        const U2_PARAMS: usize = 4;
        const U3_PARAMS: usize = 23;
        let num_nodes = type_ids.len();
        if x.len() != num_nodes * NUM_FEATURES * NUM_ELL
            || grad_x.len() != x.len()
            || grad_out.len() != num_nodes * NUM_FEATURES
            || weights_u1.len() != NUM_ELEMENTS * U1_PARAMS * NUM_FEATURES
            || weights_u2.len() != NUM_ELEMENTS * U2_PARAMS * NUM_FEATURES
            || weights_u3.len() != NUM_ELEMENTS * U3_PARAMS * NUM_FEATURES
        {
            return Err("product0 scalar contraction VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err("product0 scalar contraction VJP type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: 0,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let grad_out_buffer = self.buffer_with_slice(grad_out, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_scalar_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(7, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(8, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(9, Some(&grad_x_buffer), 0);
        encoder.set_buffer(10, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_scalar_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The command
        // buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn product0_scalar_contraction_vjp_cache_f32(
        &self,
        type_ids: &[u32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
    ) -> Result<Mp0bProduct0ScalarVjpCacheMetal, String> {
        const NUM_ELEMENTS: usize = 89;
        self.product0_scalar_contraction_vjp_cache_for_elements_f32(
            NUM_ELEMENTS,
            type_ids,
            weights_u1,
            weights_u2,
            weights_u3,
            u1_entries,
            u2_entries,
            u3_entries,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_scalar_contraction_vjp_cache_for_elements_f32(
        &self,
        num_elements: usize,
        type_ids: &[u32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
    ) -> Result<Mp0bProduct0ScalarVjpCacheMetal, String> {
        const NUM_FEATURES: usize = 128;
        const U1_PARAMS: usize = 1;
        const U2_PARAMS: usize = 4;
        const U3_PARAMS: usize = 23;
        let num_nodes = type_ids.len();
        if weights_u1.len() != num_elements * U1_PARAMS * NUM_FEATURES
            || weights_u2.len() != num_elements * U2_PARAMS * NUM_FEATURES
            || weights_u3.len() != num_elements * U3_PARAMS * NUM_FEATURES
        {
            return Err("product0 scalar contraction VJP cache shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("product0 scalar contraction VJP cache type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: 0,
        };
        let options = MTLResourceOptions::StorageModeShared;
        Ok(Mp0bProduct0ScalarVjpCacheMetal {
            num_nodes,
            type_ids_buffer: self.buffer_with_slice(type_ids, options),
            weights_u1_buffer: self.buffer_with_slice(weights_u1, options),
            weights_u2_buffer: self.buffer_with_slice(weights_u2, options),
            weights_u3_buffer: self.buffer_with_slice(weights_u3, options),
            u1_entries_buffer: self.buffer_with_slice(&u1_entries, options),
            u2_entries_buffer: self.buffer_with_slice(&u2_entries, options),
            u3_entries_buffer: self.buffer_with_slice(&u3_entries, options),
            params_buffer: self.buffer_with_slice(std::slice::from_ref(&params), options),
        })
    }

    #[allow(dead_code)]
    fn product0_scalar_contraction_vjp_with_cache_f32(
        &self,
        cache: &Mp0bProduct0ScalarVjpCacheMetal,
        x: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        self.product0_scalar_contraction_vjp_with_input_cache_f32(cache, x, None, grad_out, grad_x)
    }

    fn product0_scalar_contraction_vjp_with_input_cache_f32(
        &self,
        cache: &Mp0bProduct0ScalarVjpCacheMetal,
        x: &[f32],
        x_metal_cache: Option<&Mp0bF32BufferMetal>,
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        let num_nodes = cache.num_nodes;
        let expected_x_len = num_nodes * NUM_FEATURES * NUM_ELL;
        if let Some(cache) = x_metal_cache {
            if cache.len != expected_x_len || (!x.is_empty() && x.len() != expected_x_len) {
                return Err(
                    "product0 scalar contraction cached VJP input cache shape mismatch".to_owned(),
                );
            }
        } else if x.len() != expected_x_len {
            return Err("product0 scalar contraction cached VJP shape mismatch".to_owned());
        }
        if grad_x.len() != expected_x_len || grad_out.len() != num_nodes * NUM_FEATURES {
            return Err("product0 scalar contraction cached VJP shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let owned_x_buffer;
        let x_buffer = if let Some(cache) = x_metal_cache {
            &cache.buffer
        } else {
            owned_x_buffer = self.buffer_with_slice(x, options);
            &owned_x_buffer
        };
        let grad_out_buffer = self.buffer_with_slice(grad_out, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_scalar_vjp_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(x_buffer), 0);
        encoder.set_buffer(2, Some(&grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&cache.weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&cache.weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&cache.weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&cache.u1_entries_buffer), 0);
        encoder.set_buffer(7, Some(&cache.u2_entries_buffer), 0);
        encoder.set_buffer(8, Some(&cache.u3_entries_buffer), 0);
        encoder.set_buffer(9, Some(&grad_x_buffer), 0);
        encoder.set_buffer(10, Some(&cache.params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_scalar_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The command
        // buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_contraction_vjp_cache_f32(
        &self,
        type_ids: &[u32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        num_elements: usize,
        num_features: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
    ) -> Result<MaceScalarVjpCacheMetal, String> {
        let num_nodes = type_ids.len();
        if num_ell != 16 {
            return Err("scalar contraction VJP Metal cache requires num_ell=16".to_owned());
        }
        if weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
        {
            return Err("scalar contraction VJP cache shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("scalar contraction VJP cache type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = MaceScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_ell: u32::try_from(num_ell)
                .map_err(|_| "ell count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "U1 parameter count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "U2 parameter count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "U3 parameter count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        Ok(MaceScalarVjpCacheMetal {
            num_nodes,
            num_features,
            num_ell,
            type_ids_buffer: self.buffer_with_slice(type_ids, options),
            weights_u1_buffer: self.buffer_with_slice(weights_u1, options),
            weights_u2_buffer: self.buffer_with_slice(weights_u2, options),
            weights_u3_buffer: self.buffer_with_slice(weights_u3, options),
            u1_entries_buffer: self.buffer_with_slice(&u1_entries, options),
            u2_entries_buffer: self.buffer_with_slice(&u2_entries, options),
            u3_entries_buffer: self.buffer_with_slice(&u3_entries, options),
            params_buffer: self.buffer_with_slice(std::slice::from_ref(&params), options),
        })
    }

    fn scalar_contraction_vjp_with_input_cache_f32(
        &self,
        cache: &MaceScalarVjpCacheMetal,
        x: &[f32],
        x_metal_cache: Option<&Mp0bF32BufferMetal>,
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        let expected_x_len = cache.num_nodes * cache.num_features * cache.num_ell;
        if let Some(cache) = x_metal_cache {
            if cache.len != expected_x_len || (!x.is_empty() && x.len() != expected_x_len) {
                return Err("scalar contraction cached VJP input cache shape mismatch".to_owned());
            }
        } else if x.len() != expected_x_len {
            return Err("scalar contraction cached VJP shape mismatch".to_owned());
        }
        if grad_x.len() != expected_x_len || grad_out.len() != cache.num_nodes * cache.num_features
        {
            return Err("scalar contraction cached VJP shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let owned_x_buffer;
        let x_buffer = if let Some(cache) = x_metal_cache {
            &cache.buffer
        } else {
            owned_x_buffer = self.buffer_with_slice(x, options);
            &owned_x_buffer
        };
        let grad_out_buffer = self.buffer_with_slice(grad_out, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mace_scalar_vjp_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(x_buffer), 0);
        encoder.set_buffer(2, Some(&grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&cache.weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&cache.weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&cache.weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&cache.u1_entries_buffer), 0);
        encoder.set_buffer(7, Some(&cache.u2_entries_buffer), 0);
        encoder.set_buffer(8, Some(&cache.u3_entries_buffer), 0);
        encoder.set_buffer(9, Some(&grad_x_buffer), 0);
        encoder.set_buffer(10, Some(&cache.params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (cache.num_nodes * cache.num_features) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.mace_scalar_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The command
        // buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        grad_out: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        num_elements: usize,
        num_features: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        let num_nodes = type_ids.len();
        if num_ell != 16 {
            return Err("scalar contraction VJP Metal requires num_ell=16".to_owned());
        }
        if x.len() != num_nodes * num_features * num_ell
            || grad_x.len() != x.len()
            || grad_out.len() != num_nodes * num_features
            || weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
        {
            return Err("scalar contraction VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("scalar contraction VJP type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = MaceScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_ell: u32::try_from(num_ell)
                .map_err(|_| "ell count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "U1 parameter count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "U2 parameter count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "U3 parameter count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let grad_out_buffer = self.buffer_with_slice(grad_out, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mace_scalar_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(7, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(8, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(9, Some(&grad_x_buffer), 0);
        encoder.set_buffer(10, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * num_features) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.mace_scalar_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The command
        // buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn scalar_contraction_forward_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        num_elements: usize,
        num_features: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
        output_stride: usize,
        output_offset: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let num_nodes = type_ids.len();
        if output_stride < num_features || output_offset + num_features > output_stride {
            return Err("scalar contraction forward output stride mismatch".to_owned());
        }
        if x.len() != num_nodes * num_features * num_ell
            || out.len() != num_nodes * output_stride
            || weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
        {
            return Err("scalar contraction forward shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("scalar contraction forward type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = MaceScalarForwardParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_ell: u32::try_from(num_ell)
                .map_err(|_| "ell count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "U1 parameter count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "U2 parameter count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "U3 parameter count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            out_stride: u32::try_from(output_stride)
                .map_err(|_| "output stride exceeds Metal u32 range".to_owned())?,
            out_offset: u32::try_from(output_offset)
                .map_err(|_| "output offset exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let out_buffer = self.buffer_with_slice(out, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mace_scalar_forward_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(5, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(6, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(7, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(8, Some(&out_buffer), 0);
        encoder.set_buffer(9, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * num_features) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.mace_scalar_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `out_buffer` is StorageModeShared and initialized from `out`
        // with exactly the destination slice length. The command buffer
        // completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.contents().cast::<f32>(),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_contraction_forward_with_input_cache_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        x_metal_cache: Option<&Mp0bF32BufferMetal>,
        weights_u1: &[f32],
        weights_u2: &[f32],
        weights_u3: &[f32],
        num_elements: usize,
        num_features: usize,
        num_ell: usize,
        u1_params: usize,
        u2_params: usize,
        u3_params: usize,
        u1_entries: &[super::ScalarU1EntryF32],
        u2_entries: &[super::ScalarU2EntryF32],
        u3_entries: &[super::ScalarU3EntryF32],
        output_stride: usize,
        output_offset: usize,
        out: &mut [f32],
    ) -> Result<Mp0bF32BufferMetal, String> {
        let num_nodes = type_ids.len();
        if output_stride < num_features || output_offset + num_features > output_stride {
            return Err("scalar contraction forward output stride mismatch".to_owned());
        }
        let expected_x_len = num_nodes * num_features * num_ell;
        if let Some(cache) = x_metal_cache {
            if cache.len != expected_x_len || (!x.is_empty() && x.len() != expected_x_len) {
                return Err(
                    "scalar contraction forward input Metal cache shape mismatch".to_owned(),
                );
            }
        } else if x.len() != expected_x_len {
            return Err("scalar contraction forward shape mismatch".to_owned());
        }
        if out.len() != num_nodes * output_stride
            || weights_u1.len() != num_elements * u1_params * num_features
            || weights_u2.len() != num_elements * u2_params * num_features
            || weights_u3.len() != num_elements * u3_params * num_features
        {
            return Err("scalar contraction forward shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= num_elements)
        {
            return Err("scalar contraction forward type id out of range".to_owned());
        }
        let u1_entries = pack_scalar_u1_entries(u1_entries)?;
        let u2_entries = pack_scalar_u2_entries(u2_entries)?;
        let u3_entries = pack_scalar_u3_entries(u3_entries)?;
        let params = MaceScalarForwardParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            num_elements: u32::try_from(num_elements)
                .map_err(|_| "element count exceeds Metal u32 range".to_owned())?,
            num_features: u32::try_from(num_features)
                .map_err(|_| "feature count exceeds Metal u32 range".to_owned())?,
            num_ell: u32::try_from(num_ell)
                .map_err(|_| "ell count exceeds Metal u32 range".to_owned())?,
            u1_params: u32::try_from(u1_params)
                .map_err(|_| "U1 parameter count exceeds Metal u32 range".to_owned())?,
            u2_params: u32::try_from(u2_params)
                .map_err(|_| "U2 parameter count exceeds Metal u32 range".to_owned())?,
            u3_params: u32::try_from(u3_params)
                .map_err(|_| "U3 parameter count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            out_stride: u32::try_from(output_stride)
                .map_err(|_| "output stride exceeds Metal u32 range".to_owned())?,
            out_offset: u32::try_from(output_offset)
                .map_err(|_| "output offset exceeds Metal u32 range".to_owned())?,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let x_metal = if let Some(cache) = x_metal_cache {
            cache.clone()
        } else {
            Mp0bF32BufferMetal {
                len: expected_x_len,
                buffer: self.buffer_with_slice(x, options),
            }
        };
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let weights_u1_buffer = self.buffer_with_slice(weights_u1, options);
        let weights_u2_buffer = self.buffer_with_slice(weights_u2, options);
        let weights_u3_buffer = self.buffer_with_slice(weights_u3, options);
        let u1_entries_buffer = self.buffer_with_slice(&u1_entries, options);
        let u2_entries_buffer = self.buffer_with_slice(&u2_entries, options);
        let u3_entries_buffer = self.buffer_with_slice(&u3_entries, options);
        let out_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(out) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mace_scalar_forward_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_metal.buffer), 0);
        encoder.set_buffer(2, Some(&weights_u1_buffer), 0);
        encoder.set_buffer(3, Some(&weights_u2_buffer), 0);
        encoder.set_buffer(4, Some(&weights_u3_buffer), 0);
        encoder.set_buffer(5, Some(&u1_entries_buffer), 0);
        encoder.set_buffer(6, Some(&u2_entries_buffer), 0);
        encoder.set_buffer(7, Some(&u3_entries_buffer), 0);
        encoder.set_buffer(8, Some(&out_buffer), 0);
        encoder.set_buffer(9, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * num_features) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.mace_scalar_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `out_buffer` is StorageModeShared and sized exactly from the
        // destination slice length. The scalar forward kernel writes every
        // output element addressed by the configured stride and offset.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.contents().cast::<f32>(),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(x_metal)
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_scalar_vector_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        scalar_grad_out: &[f32],
        vector_grad_out: &[f32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        const NUM_ELEMENTS: usize = 89;
        let num_nodes = type_ids.len();
        if x.len() != num_nodes * NUM_FEATURES * NUM_ELL
            || grad_x.len() != x.len()
            || scalar_grad_out.len() != num_nodes * NUM_FEATURES
            || vector_grad_out.len() != num_nodes * NUM_FEATURES * NUM_EQUIV
            || scalar_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || scalar_weights_u2.len() != NUM_ELEMENTS * 4 * NUM_FEATURES
            || scalar_weights_u3.len() != NUM_ELEMENTS * 23 * NUM_FEATURES
            || vector_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || vector_weights_u2.len() != NUM_ELEMENTS * 6 * NUM_FEATURES
            || vector_weights_u3.len() != NUM_ELEMENTS * 51 * NUM_FEATURES
            || vector_u1_entries_by_equiv.len() != NUM_EQUIV
            || vector_u2_entries_by_equiv.len() != NUM_EQUIV
            || vector_u3_entries_by_equiv.len() != NUM_EQUIV
        {
            return Err("product0 scalar/vector contraction VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err("product0 scalar/vector contraction VJP type id out of range".to_owned());
        }

        let scalar_u1_entries = pack_scalar_u1_entries(scalar_u1_entries)?;
        let scalar_u2_entries = pack_scalar_u2_entries(scalar_u2_entries)?;
        let scalar_u3_entries = pack_scalar_u3_entries(scalar_u3_entries)?;
        let (vector_u1_offsets, vector_u1_entries) =
            pack_vector_u1_entries(vector_u1_entries_by_equiv)?;
        let (vector_u2_offsets, vector_u2_entries) =
            pack_vector_u2_entries(vector_u2_entries_by_equiv)?;
        let (vector_u3_offsets, vector_u3_entries) =
            pack_vector_u3_entries(vector_u3_entries_by_equiv)?;
        let scalar_params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(scalar_u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(scalar_u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(scalar_u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: 0,
        };
        let vector_params = Mp0bProduct0VectorVjpParams {
            num_nodes: scalar_params.num_nodes,
            num_elements: NUM_ELEMENTS as u32,
            num_features: NUM_FEATURES as u32,
            num_equivariance: NUM_EQUIV as u32,
            u1_params: 1,
            u2_params: 6,
            u3_params: 51,
            output_stride: 512,
            output_offset: 128,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);
        let scalar_grad_out_buffer = self.buffer_with_slice(scalar_grad_out, options);
        let scalar_weights_u1_buffer = self.buffer_with_slice(scalar_weights_u1, options);
        let scalar_weights_u2_buffer = self.buffer_with_slice(scalar_weights_u2, options);
        let scalar_weights_u3_buffer = self.buffer_with_slice(scalar_weights_u3, options);
        let scalar_u1_entries_buffer = self.buffer_with_slice(&scalar_u1_entries, options);
        let scalar_u2_entries_buffer = self.buffer_with_slice(&scalar_u2_entries, options);
        let scalar_u3_entries_buffer = self.buffer_with_slice(&scalar_u3_entries, options);
        let vector_grad_out_buffer = self.buffer_with_slice(vector_grad_out, options);
        let vector_weights_u1_buffer = self.buffer_with_slice(vector_weights_u1, options);
        let vector_weights_u2_buffer = self.buffer_with_slice(vector_weights_u2, options);
        let vector_weights_u3_buffer = self.buffer_with_slice(vector_weights_u3, options);
        let vector_u1_offsets_buffer = self.buffer_with_slice(&vector_u1_offsets, options);
        let vector_u2_offsets_buffer = self.buffer_with_slice(&vector_u2_offsets, options);
        let vector_u3_offsets_buffer = self.buffer_with_slice(&vector_u3_offsets, options);
        let vector_u1_entries_buffer = self.buffer_with_slice(&vector_u1_entries, options);
        let vector_u2_entries_buffer = self.buffer_with_slice(&vector_u2_entries, options);
        let vector_u3_entries_buffer = self.buffer_with_slice(&vector_u3_entries, options);
        let scalar_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&scalar_params), options);
        let vector_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&vector_params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_scalar_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&scalar_grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&scalar_weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&scalar_weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&scalar_weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&scalar_u1_entries_buffer), 0);
        encoder.set_buffer(7, Some(&scalar_u2_entries_buffer), 0);
        encoder.set_buffer(8, Some(&scalar_u3_entries_buffer), 0);
        encoder.set_buffer(9, Some(&grad_x_buffer), 0);
        encoder.set_buffer(10, Some(&scalar_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_scalar_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.product0_vector_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&vector_grad_out_buffer), 0);
        encoder.set_buffer(3, Some(&vector_weights_u1_buffer), 0);
        encoder.set_buffer(4, Some(&vector_weights_u2_buffer), 0);
        encoder.set_buffer(5, Some(&vector_weights_u3_buffer), 0);
        encoder.set_buffer(6, Some(&vector_u1_offsets_buffer), 0);
        encoder.set_buffer(7, Some(&vector_u2_offsets_buffer), 0);
        encoder.set_buffer(8, Some(&vector_u3_offsets_buffer), 0);
        encoder.set_buffer(9, Some(&vector_u1_entries_buffer), 0);
        encoder.set_buffer(10, Some(&vector_u2_entries_buffer), 0);
        encoder.set_buffer(11, Some(&vector_u3_entries_buffer), 0);
        encoder.set_buffer(12, Some(&grad_x_buffer), 0);
        encoder.set_buffer(13, Some(&vector_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_vector_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The scalar and
        // vector VJP dispatches run sequentially in one command buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_linear_scalar_vector_contraction_vjp_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        bar_product_out: &[f32],
        product_linear_weight: &[f32],
        product_linear_output_mask: &[f32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        grad_x: &mut [f32],
    ) -> Result<(), String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        const PRODUCT_ROW_LEN: usize = 512;
        const PRODUCT_LINEAR_WEIGHT_LEN: usize = 2 * NUM_FEATURES * NUM_FEATURES;
        const NUM_ELEMENTS: usize = 89;
        let num_nodes = type_ids.len();
        if x.len() != num_nodes * NUM_FEATURES * NUM_ELL
            || grad_x.len() != x.len()
            || bar_product_out.len() != num_nodes * PRODUCT_ROW_LEN
            || product_linear_weight.len() != PRODUCT_LINEAR_WEIGHT_LEN
            || product_linear_output_mask.len() != PRODUCT_ROW_LEN
            || scalar_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || scalar_weights_u2.len() != NUM_ELEMENTS * 4 * NUM_FEATURES
            || scalar_weights_u3.len() != NUM_ELEMENTS * 23 * NUM_FEATURES
            || vector_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || vector_weights_u2.len() != NUM_ELEMENTS * 6 * NUM_FEATURES
            || vector_weights_u3.len() != NUM_ELEMENTS * 51 * NUM_FEATURES
            || vector_u1_entries_by_equiv.len() != NUM_EQUIV
            || vector_u2_entries_by_equiv.len() != NUM_EQUIV
            || vector_u3_entries_by_equiv.len() != NUM_EQUIV
        {
            return Err("product0 linear scalar/vector contraction VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err(
                "product0 linear scalar/vector contraction VJP type id out of range".to_owned(),
            );
        }

        let scalar_u1_entries = pack_scalar_u1_entries(scalar_u1_entries)?;
        let scalar_u2_entries = pack_scalar_u2_entries(scalar_u2_entries)?;
        let scalar_u3_entries = pack_scalar_u3_entries(scalar_u3_entries)?;
        let (vector_u1_offsets, vector_u1_entries) =
            pack_vector_u1_entries(vector_u1_entries_by_equiv)?;
        let (vector_u2_offsets, vector_u2_entries) =
            pack_vector_u2_entries(vector_u2_entries_by_equiv)?;
        let (vector_u3_offsets, vector_u3_entries) =
            pack_vector_u3_entries(vector_u3_entries_by_equiv)?;
        let mut flags = 0;
        if product_linear_output_mask.iter().all(|&mask| mask == 1.0) {
            flags |= MP0B_PRODUCT0_VJP_OUTPUT_MASK_ALL_ONES_FLAG;
        }
        let params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(scalar_u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(scalar_u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(scalar_u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let bar_product_out_buffer = self.buffer_with_slice(bar_product_out, options);
        let product_linear_weight_buffer = self.buffer_with_slice(product_linear_weight, options);
        let product_linear_output_mask_buffer =
            self.buffer_with_slice(product_linear_output_mask, options);
        let scalar_weights_u1_buffer = self.buffer_with_slice(scalar_weights_u1, options);
        let scalar_weights_u2_buffer = self.buffer_with_slice(scalar_weights_u2, options);
        let scalar_weights_u3_buffer = self.buffer_with_slice(scalar_weights_u3, options);
        let scalar_u1_entries_buffer = self.buffer_with_slice(&scalar_u1_entries, options);
        let scalar_u2_entries_buffer = self.buffer_with_slice(&scalar_u2_entries, options);
        let scalar_u3_entries_buffer = self.buffer_with_slice(&scalar_u3_entries, options);
        let vector_weights_u1_buffer = self.buffer_with_slice(vector_weights_u1, options);
        let vector_weights_u2_buffer = self.buffer_with_slice(vector_weights_u2, options);
        let vector_weights_u3_buffer = self.buffer_with_slice(vector_weights_u3, options);
        let vector_u1_offsets_buffer = self.buffer_with_slice(&vector_u1_offsets, options);
        let vector_u2_offsets_buffer = self.buffer_with_slice(&vector_u2_offsets, options);
        let vector_u3_offsets_buffer = self.buffer_with_slice(&vector_u3_offsets, options);
        let vector_u1_entries_buffer = self.buffer_with_slice(&vector_u1_entries, options);
        let vector_u2_entries_buffer = self.buffer_with_slice(&vector_u2_entries, options);
        let vector_u3_entries_buffer = self.buffer_with_slice(&vector_u3_entries, options);
        let grad_x_buffer = self.buffer_with_slice(grad_x, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_linear_scalar_vector_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&bar_product_out_buffer), 0);
        encoder.set_buffer(3, Some(&product_linear_weight_buffer), 0);
        encoder.set_buffer(4, Some(&product_linear_output_mask_buffer), 0);
        encoder.set_buffer(5, Some(&scalar_weights_u1_buffer), 0);
        encoder.set_buffer(6, Some(&scalar_weights_u2_buffer), 0);
        encoder.set_buffer(7, Some(&scalar_weights_u3_buffer), 0);
        encoder.set_buffer(8, Some(&scalar_u1_entries_buffer), 0);
        encoder.set_buffer(9, Some(&scalar_u2_entries_buffer), 0);
        encoder.set_buffer(10, Some(&scalar_u3_entries_buffer), 0);
        encoder.set_buffer(11, Some(&vector_weights_u1_buffer), 0);
        encoder.set_buffer(12, Some(&vector_weights_u2_buffer), 0);
        encoder.set_buffer(13, Some(&vector_weights_u3_buffer), 0);
        encoder.set_buffer(14, Some(&vector_u1_offsets_buffer), 0);
        encoder.set_buffer(15, Some(&vector_u2_offsets_buffer), 0);
        encoder.set_buffer(16, Some(&vector_u3_offsets_buffer), 0);
        encoder.set_buffer(17, Some(&vector_u1_entries_buffer), 0);
        encoder.set_buffer(18, Some(&vector_u2_entries_buffer), 0);
        encoder.set_buffer(19, Some(&vector_u3_entries_buffer), 0);
        encoder.set_buffer(20, Some(&grad_x_buffer), 0);
        encoder.set_buffer(21, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_linear_scalar_vector_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `grad_x_buffer` is StorageModeShared and initialized from
        // `grad_x` with exactly the destination slice length. The fused kernel
        // completed before this readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                grad_x_buffer.contents().cast::<f32>(),
                grad_x.as_mut_ptr(),
                grad_x.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_layer0_vjp_cache_f32(
        &self,
        type_ids: &[u32],
        product_linear_weight: &[f32],
        product_linear_output_mask: &[f32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        selector_weight: &[f32],
        selector_output_mask: &[f32],
        hidden_weight: &[f32],
        hidden_output_mask: &[f32],
    ) -> Result<Mp0bProduct0Layer0VjpCacheMetal, String> {
        const NUM_FEATURES: usize = 128;
        const NUM_EQUIV: usize = 3;
        const HIDDEN_ROW_LEN: usize = 2048;
        const PRODUCT_ROW_LEN: usize = 512;
        const PRODUCT_LINEAR_WEIGHT_LEN: usize = 2 * NUM_FEATURES * NUM_FEATURES;
        const SELECTOR_WEIGHT_LEN: usize = 4 * NUM_FEATURES * 89 * NUM_FEATURES;
        const HIDDEN_WEIGHT_LEN: usize = 4 * NUM_FEATURES * NUM_FEATURES;
        const NUM_ELEMENTS: usize = 89;
        let num_nodes = type_ids.len();
        if product_linear_weight.len() != PRODUCT_LINEAR_WEIGHT_LEN
            || product_linear_output_mask.len() != PRODUCT_ROW_LEN
            || scalar_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || scalar_weights_u2.len() != NUM_ELEMENTS * 4 * NUM_FEATURES
            || scalar_weights_u3.len() != NUM_ELEMENTS * 23 * NUM_FEATURES
            || vector_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || vector_weights_u2.len() != NUM_ELEMENTS * 6 * NUM_FEATURES
            || vector_weights_u3.len() != NUM_ELEMENTS * 51 * NUM_FEATURES
            || vector_u1_entries_by_equiv.len() != NUM_EQUIV
            || vector_u2_entries_by_equiv.len() != NUM_EQUIV
            || vector_u3_entries_by_equiv.len() != NUM_EQUIV
            || selector_weight.len() != SELECTOR_WEIGHT_LEN
            || selector_output_mask.len() != HIDDEN_ROW_LEN
            || hidden_weight.len() != HIDDEN_WEIGHT_LEN
            || hidden_output_mask.len() != HIDDEN_ROW_LEN
        {
            return Err("product0-to-layer0 VJP cache shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err("product0-to-layer0 VJP cache type id out of range".to_owned());
        }

        let scalar_u1_entries = pack_scalar_u1_entries(scalar_u1_entries)?;
        let scalar_u2_entries = pack_scalar_u2_entries(scalar_u2_entries)?;
        let scalar_u3_entries = pack_scalar_u3_entries(scalar_u3_entries)?;
        let (vector_u1_offsets, vector_u1_entries) =
            pack_vector_u1_entries(vector_u1_entries_by_equiv)?;
        let (vector_u2_offsets, vector_u2_entries) =
            pack_vector_u2_entries(vector_u2_entries_by_equiv)?;
        let (vector_u3_offsets, vector_u3_entries) =
            pack_vector_u3_entries(vector_u3_entries_by_equiv)?;
        let mut product_flags = MP0B_PRODUCT0_VJP_ASSIGN_OUTPUT_FLAG;
        if product_linear_output_mask.iter().all(|&mask| mask == 1.0) {
            product_flags |= MP0B_PRODUCT0_VJP_OUTPUT_MASK_ALL_ONES_FLAG;
        }
        let product_params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(scalar_u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(scalar_u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(scalar_u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: product_flags,
        };
        let hidden_params = Mp0bLayer1HiddenVjpParams {
            num_nodes: product_params.num_nodes,
        };
        let options = MTLResourceOptions::StorageModeShared;
        let hidden_scratch_bytes = (num_nodes * HIDDEN_ROW_LEN * std::mem::size_of::<f32>()) as u64;
        let density_scratch_bytes = (num_nodes * std::mem::size_of::<f32>()) as u64;
        Ok(Mp0bProduct0Layer0VjpCacheMetal {
            num_nodes,
            type_ids_buffer: self.buffer_with_slice(type_ids, options),
            product_linear_weight_buffer: self.buffer_with_slice(product_linear_weight, options),
            product_linear_output_mask_buffer: self
                .buffer_with_slice(product_linear_output_mask, options),
            scalar_weights_u1_buffer: self.buffer_with_slice(scalar_weights_u1, options),
            scalar_weights_u2_buffer: self.buffer_with_slice(scalar_weights_u2, options),
            scalar_weights_u3_buffer: self.buffer_with_slice(scalar_weights_u3, options),
            scalar_u1_entries_buffer: self.buffer_with_slice(&scalar_u1_entries, options),
            scalar_u2_entries_buffer: self.buffer_with_slice(&scalar_u2_entries, options),
            scalar_u3_entries_buffer: self.buffer_with_slice(&scalar_u3_entries, options),
            vector_weights_u1_buffer: self.buffer_with_slice(vector_weights_u1, options),
            vector_weights_u2_buffer: self.buffer_with_slice(vector_weights_u2, options),
            vector_weights_u3_buffer: self.buffer_with_slice(vector_weights_u3, options),
            vector_u1_offsets_buffer: self.buffer_with_slice(&vector_u1_offsets, options),
            vector_u2_offsets_buffer: self.buffer_with_slice(&vector_u2_offsets, options),
            vector_u3_offsets_buffer: self.buffer_with_slice(&vector_u3_offsets, options),
            vector_u1_entries_buffer: self.buffer_with_slice(&vector_u1_entries, options),
            vector_u2_entries_buffer: self.buffer_with_slice(&vector_u2_entries, options),
            vector_u3_entries_buffer: self.buffer_with_slice(&vector_u3_entries, options),
            selector_weight_buffer: self.buffer_with_slice(selector_weight, options),
            selector_output_mask_buffer: self.buffer_with_slice(selector_output_mask, options),
            hidden_weight_buffer: self.buffer_with_slice(hidden_weight, options),
            hidden_output_mask_buffer: self.buffer_with_slice(hidden_output_mask, options),
            product_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&product_params), options),
            hidden_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&hidden_params), options),
            bar_interaction_out_buffer: self.device.new_buffer(hidden_scratch_bytes, options),
            bar_linear_message_buffer: self.device.new_buffer(hidden_scratch_bytes, options),
            bar_input_buffer: self.device.new_buffer(hidden_scratch_bytes, options),
            bar_density_buffer: self.device.new_buffer(density_scratch_bytes, options),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_layer0_vjp_with_cache_f32(
        &self,
        cache: &Mp0bProduct0Layer0VjpCacheMetal,
        x: &[f32],
        x_metal_cache: Option<&Mp0bF32BufferMetal>,
        bar_product_out: &[f32],
        bar_product_out_metal: Option<&Mp0bF32BufferMetal>,
        density: &[f32],
        density_metal_cache: Option<&Mp0bF32BufferMetal>,
        linear_out: &[f32],
        linear_out_metal_cache: Option<&Mp0bF32BufferMetal>,
        bar_input: &mut [f32],
        bar_density: &mut [f32],
        read_bar_input: bool,
        read_bar_density: bool,
    ) -> Result<Mp0bProduct0Layer0VjpMetal, String> {
        const NUM_FEATURES: usize = 128;
        const HIDDEN_ROW_LEN: usize = 2048;
        const PRODUCT_ROW_LEN: usize = 512;
        let num_nodes = cache.num_nodes;
        let x_len = if let Some(cache) = x_metal_cache {
            if cache.len != num_nodes * HIDDEN_ROW_LEN {
                return Err("product0-to-layer0 cached VJP x Metal cache shape mismatch".to_owned());
            }
            if !x.is_empty() && x.len() != cache.len {
                return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
            }
            cache.len
        } else {
            x.len()
        };
        let density_len = if let Some(cache) = density_metal_cache {
            if cache.len != num_nodes {
                return Err(
                    "product0-to-layer0 cached VJP density Metal cache shape mismatch".to_owned(),
                );
            }
            if !density.is_empty() && density.len() != cache.len {
                return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
            }
            cache.len
        } else {
            density.len()
        };
        let linear_out_len = if let Some(cache) = linear_out_metal_cache {
            if cache.len != num_nodes * HIDDEN_ROW_LEN {
                return Err(
                    "product0-to-layer0 cached VJP linear output Metal cache shape mismatch"
                        .to_owned(),
                );
            }
            if !linear_out.is_empty() && linear_out.len() != cache.len {
                return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
            }
            cache.len
        } else {
            linear_out.len()
        };
        if x_len != num_nodes * HIDDEN_ROW_LEN
            || density_len != num_nodes
            || linear_out_len != num_nodes * HIDDEN_ROW_LEN
            || (read_bar_input && bar_input.len() != num_nodes * HIDDEN_ROW_LEN)
            || (read_bar_density && bar_density.len() != num_nodes)
        {
            return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
        }
        let bar_product_out_len = num_nodes * PRODUCT_ROW_LEN;
        if let Some(cache) = bar_product_out_metal {
            if cache.len != bar_product_out_len {
                return Err(
                    "product0-to-layer0 cached VJP bar product Metal cache shape mismatch"
                        .to_owned(),
                );
            }
            if !bar_product_out.is_empty() && bar_product_out.len() != bar_product_out_len {
                return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
            }
        } else if bar_product_out.len() != bar_product_out_len {
            return Err("product0-to-layer0 cached VJP shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let owned_x_buffer;
        let x_buffer = if let Some(cache) = x_metal_cache {
            &cache.buffer
        } else {
            owned_x_buffer = self.buffer_with_slice(x, options);
            &owned_x_buffer
        };
        let owned_bar_product_out_buffer;
        let bar_product_out_buffer = if let Some(cache) = bar_product_out_metal {
            &cache.buffer
        } else {
            owned_bar_product_out_buffer = self.buffer_with_slice(bar_product_out, options);
            &owned_bar_product_out_buffer
        };
        let owned_density_buffer;
        let density_buffer = if let Some(cache) = density_metal_cache {
            &cache.buffer
        } else {
            owned_density_buffer = self.buffer_with_slice(density, options);
            &owned_density_buffer
        };
        let owned_linear_out_buffer;
        let linear_out_buffer = if let Some(cache) = linear_out_metal_cache {
            &cache.buffer
        } else {
            owned_linear_out_buffer = self.buffer_with_slice(linear_out, options);
            &owned_linear_out_buffer
        };

        if mp0b_trace_vjp_kernel_timings_enabled() {
            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.product0_linear_scalar_vector_vjp_pipeline);
            encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
            encoder.set_buffer(1, Some(x_buffer), 0);
            encoder.set_buffer(2, Some(bar_product_out_buffer), 0);
            encoder.set_buffer(3, Some(&cache.product_linear_weight_buffer), 0);
            encoder.set_buffer(4, Some(&cache.product_linear_output_mask_buffer), 0);
            encoder.set_buffer(5, Some(&cache.scalar_weights_u1_buffer), 0);
            encoder.set_buffer(6, Some(&cache.scalar_weights_u2_buffer), 0);
            encoder.set_buffer(7, Some(&cache.scalar_weights_u3_buffer), 0);
            encoder.set_buffer(8, Some(&cache.scalar_u1_entries_buffer), 0);
            encoder.set_buffer(9, Some(&cache.scalar_u2_entries_buffer), 0);
            encoder.set_buffer(10, Some(&cache.scalar_u3_entries_buffer), 0);
            encoder.set_buffer(11, Some(&cache.vector_weights_u1_buffer), 0);
            encoder.set_buffer(12, Some(&cache.vector_weights_u2_buffer), 0);
            encoder.set_buffer(13, Some(&cache.vector_weights_u3_buffer), 0);
            encoder.set_buffer(14, Some(&cache.vector_u1_offsets_buffer), 0);
            encoder.set_buffer(15, Some(&cache.vector_u2_offsets_buffer), 0);
            encoder.set_buffer(16, Some(&cache.vector_u3_offsets_buffer), 0);
            encoder.set_buffer(17, Some(&cache.vector_u1_entries_buffer), 0);
            encoder.set_buffer(18, Some(&cache.vector_u2_entries_buffer), 0);
            encoder.set_buffer(19, Some(&cache.vector_u3_entries_buffer), 0);
            encoder.set_buffer(20, Some(&cache.bar_interaction_out_buffer), 0);
            encoder.set_buffer(21, Some(&cache.product_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (num_nodes * NUM_FEATURES) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.product0_linear_scalar_vector_vjp_pipeline),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal product0_layer0_vjp_product_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.selector_vjp_input_pipeline);
            encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
            encoder.set_buffer(1, Some(&cache.selector_weight_buffer), 0);
            encoder.set_buffer(2, Some(&cache.selector_output_mask_buffer), 0);
            encoder.set_buffer(3, Some(&cache.bar_interaction_out_buffer), 0);
            encoder.set_buffer(4, Some(&cache.bar_linear_message_buffer), 0);
            encoder.set_buffer(5, Some(&cache.hidden_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.selector_vjp_input_pipeline),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal product0_layer0_vjp_selector_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
            encoder.set_buffer(0, Some(density_buffer), 0);
            encoder.set_buffer(1, Some(linear_out_buffer), 0);
            encoder.set_buffer(2, Some(&cache.bar_linear_message_buffer), 0);
            encoder.set_buffer(3, Some(&cache.bar_density_buffer), 0);
            encoder.set_buffer(4, Some(&cache.hidden_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: num_nodes as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal product0_layer0_vjp_density_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            let start = Instant::now();
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layer0_hidden_vjp_input_pipeline);
            encoder.set_buffer(0, Some(density_buffer), 0);
            encoder.set_buffer(1, Some(&cache.hidden_weight_buffer), 0);
            encoder.set_buffer(2, Some(&cache.hidden_output_mask_buffer), 0);
            encoder.set_buffer(3, Some(&cache.bar_linear_message_buffer), 0);
            encoder.set_buffer(4, Some(&cache.bar_input_buffer), 0);
            encoder.set_buffer(5, Some(&cache.hidden_params_buffer), 0);
            encoder.dispatch_threads(
                MTLSize {
                    width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_group(&self.layer0_hidden_vjp_input_pipeline),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
            eprintln!(
                "mace_mp0b_metal product0_layer0_vjp_hidden_input_kernel_ms={:.3}",
                start.elapsed().as_secs_f64() * 1.0e3
            );

            // Safety: output buffers are StorageModeShared and sized exactly
            // from the destination slice lengths. All traced command buffers
            // completed before these readbacks.
            unsafe {
                if read_bar_input {
                    std::ptr::copy_nonoverlapping(
                        cache.bar_input_buffer.contents().cast::<f32>(),
                        bar_input.as_mut_ptr(),
                        bar_input.len(),
                    );
                }
                if read_bar_density {
                    std::ptr::copy_nonoverlapping(
                        cache.bar_density_buffer.contents().cast::<f32>(),
                        bar_density.as_mut_ptr(),
                        bar_density.len(),
                    );
                }
            }
            return Ok(Mp0bProduct0Layer0VjpMetal {
                bar_input: Mp0bF32BufferMetal {
                    len: num_nodes * HIDDEN_ROW_LEN,
                    buffer: cache.bar_input_buffer.clone(),
                },
                bar_density: Mp0bF32BufferMetal {
                    len: num_nodes,
                    buffer: cache.bar_density_buffer.clone(),
                },
            });
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_linear_scalar_vector_vjp_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(x_buffer), 0);
        encoder.set_buffer(2, Some(bar_product_out_buffer), 0);
        encoder.set_buffer(3, Some(&cache.product_linear_weight_buffer), 0);
        encoder.set_buffer(4, Some(&cache.product_linear_output_mask_buffer), 0);
        encoder.set_buffer(5, Some(&cache.scalar_weights_u1_buffer), 0);
        encoder.set_buffer(6, Some(&cache.scalar_weights_u2_buffer), 0);
        encoder.set_buffer(7, Some(&cache.scalar_weights_u3_buffer), 0);
        encoder.set_buffer(8, Some(&cache.scalar_u1_entries_buffer), 0);
        encoder.set_buffer(9, Some(&cache.scalar_u2_entries_buffer), 0);
        encoder.set_buffer(10, Some(&cache.scalar_u3_entries_buffer), 0);
        encoder.set_buffer(11, Some(&cache.vector_weights_u1_buffer), 0);
        encoder.set_buffer(12, Some(&cache.vector_weights_u2_buffer), 0);
        encoder.set_buffer(13, Some(&cache.vector_weights_u3_buffer), 0);
        encoder.set_buffer(14, Some(&cache.vector_u1_offsets_buffer), 0);
        encoder.set_buffer(15, Some(&cache.vector_u2_offsets_buffer), 0);
        encoder.set_buffer(16, Some(&cache.vector_u3_offsets_buffer), 0);
        encoder.set_buffer(17, Some(&cache.vector_u1_entries_buffer), 0);
        encoder.set_buffer(18, Some(&cache.vector_u2_entries_buffer), 0);
        encoder.set_buffer(19, Some(&cache.vector_u3_entries_buffer), 0);
        encoder.set_buffer(20, Some(&cache.bar_interaction_out_buffer), 0);
        encoder.set_buffer(21, Some(&cache.product_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_linear_scalar_vector_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.selector_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&cache.selector_weight_buffer), 0);
        encoder.set_buffer(2, Some(&cache.selector_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&cache.bar_interaction_out_buffer), 0);
        encoder.set_buffer(4, Some(&cache.bar_linear_message_buffer), 0);
        encoder.set_buffer(5, Some(&cache.hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.selector_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
        encoder.set_buffer(0, Some(density_buffer), 0);
        encoder.set_buffer(1, Some(linear_out_buffer), 0);
        encoder.set_buffer(2, Some(&cache.bar_linear_message_buffer), 0);
        encoder.set_buffer(3, Some(&cache.bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&cache.hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: num_nodes as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer0_hidden_vjp_input_pipeline);
        encoder.set_buffer(0, Some(density_buffer), 0);
        encoder.set_buffer(1, Some(&cache.hidden_weight_buffer), 0);
        encoder.set_buffer(2, Some(&cache.hidden_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&cache.bar_linear_message_buffer), 0);
        encoder.set_buffer(4, Some(&cache.bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&cache.hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer0_hidden_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();

        let needs_cpu_readback = read_bar_input || read_bar_density;
        if needs_cpu_readback {
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
        }
        // Safety: output buffers are StorageModeShared and sized exactly from
        // the destination slice lengths. When a CPU readback is requested, the
        // command buffer completed before this copy.
        unsafe {
            if read_bar_input {
                std::ptr::copy_nonoverlapping(
                    cache.bar_input_buffer.contents().cast::<f32>(),
                    bar_input.as_mut_ptr(),
                    bar_input.len(),
                );
            }
            if read_bar_density {
                std::ptr::copy_nonoverlapping(
                    cache.bar_density_buffer.contents().cast::<f32>(),
                    bar_density.as_mut_ptr(),
                    bar_density.len(),
                );
            }
        }
        Ok(Mp0bProduct0Layer0VjpMetal {
            bar_input: Mp0bF32BufferMetal {
                len: num_nodes * HIDDEN_ROW_LEN,
                buffer: cache.bar_input_buffer.clone(),
            },
            bar_density: Mp0bF32BufferMetal {
                len: num_nodes,
                buffer: cache.bar_density_buffer.clone(),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_linear_scalar_vector_contraction_layer0_selector_hidden_vjp_input_density_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        bar_product_out: &[f32],
        product_linear_weight: &[f32],
        product_linear_output_mask: &[f32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        selector_weight: &[f32],
        selector_output_mask: &[f32],
        density: &[f32],
        linear_out: &[f32],
        hidden_weight: &[f32],
        hidden_output_mask: &[f32],
        bar_input: &mut [f32],
        bar_density: &mut [f32],
    ) -> Result<(), String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        const HIDDEN_ROW_LEN: usize = NUM_FEATURES * NUM_ELL;
        const PRODUCT_ROW_LEN: usize = 512;
        const PRODUCT_LINEAR_WEIGHT_LEN: usize = 2 * NUM_FEATURES * NUM_FEATURES;
        const SELECTOR_WEIGHT_LEN: usize = 4 * NUM_FEATURES * 89 * NUM_FEATURES;
        const HIDDEN_WEIGHT_LEN: usize = 4 * NUM_FEATURES * NUM_FEATURES;
        const NUM_ELEMENTS: usize = 89;
        let num_nodes = type_ids.len();
        if x.len() != num_nodes * HIDDEN_ROW_LEN
            || bar_product_out.len() != num_nodes * PRODUCT_ROW_LEN
            || product_linear_weight.len() != PRODUCT_LINEAR_WEIGHT_LEN
            || product_linear_output_mask.len() != PRODUCT_ROW_LEN
            || scalar_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || scalar_weights_u2.len() != NUM_ELEMENTS * 4 * NUM_FEATURES
            || scalar_weights_u3.len() != NUM_ELEMENTS * 23 * NUM_FEATURES
            || vector_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || vector_weights_u2.len() != NUM_ELEMENTS * 6 * NUM_FEATURES
            || vector_weights_u3.len() != NUM_ELEMENTS * 51 * NUM_FEATURES
            || vector_u1_entries_by_equiv.len() != NUM_EQUIV
            || vector_u2_entries_by_equiv.len() != NUM_EQUIV
            || vector_u3_entries_by_equiv.len() != NUM_EQUIV
            || selector_weight.len() != SELECTOR_WEIGHT_LEN
            || selector_output_mask.len() != HIDDEN_ROW_LEN
            || density.len() != num_nodes
            || linear_out.len() != num_nodes * HIDDEN_ROW_LEN
            || hidden_weight.len() != HIDDEN_WEIGHT_LEN
            || hidden_output_mask.len() != HIDDEN_ROW_LEN
            || bar_input.len() != num_nodes * HIDDEN_ROW_LEN
            || bar_density.len() != num_nodes
        {
            return Err("product0-to-layer0 selector/hidden VJP shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err("product0-to-layer0 selector/hidden VJP type id out of range".to_owned());
        }

        let scalar_u1_entries = pack_scalar_u1_entries(scalar_u1_entries)?;
        let scalar_u2_entries = pack_scalar_u2_entries(scalar_u2_entries)?;
        let scalar_u3_entries = pack_scalar_u3_entries(scalar_u3_entries)?;
        let (vector_u1_offsets, vector_u1_entries) =
            pack_vector_u1_entries(vector_u1_entries_by_equiv)?;
        let (vector_u2_offsets, vector_u2_entries) =
            pack_vector_u2_entries(vector_u2_entries_by_equiv)?;
        let (vector_u3_offsets, vector_u3_entries) =
            pack_vector_u3_entries(vector_u3_entries_by_equiv)?;
        let mut product_flags = MP0B_PRODUCT0_VJP_ASSIGN_OUTPUT_FLAG;
        if product_linear_output_mask.iter().all(|&mask| mask == 1.0) {
            product_flags |= MP0B_PRODUCT0_VJP_OUTPUT_MASK_ALL_ONES_FLAG;
        }
        let product_params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(scalar_u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(scalar_u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(scalar_u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: product_flags,
        };
        let hidden_params = Mp0bLayer1HiddenVjpParams {
            num_nodes: product_params.num_nodes,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let type_ids_buffer = self.buffer_with_slice(type_ids, options);
        let x_buffer = self.buffer_with_slice(x, options);
        let bar_product_out_buffer = self.buffer_with_slice(bar_product_out, options);
        let product_linear_weight_buffer = self.buffer_with_slice(product_linear_weight, options);
        let product_linear_output_mask_buffer =
            self.buffer_with_slice(product_linear_output_mask, options);
        let scalar_weights_u1_buffer = self.buffer_with_slice(scalar_weights_u1, options);
        let scalar_weights_u2_buffer = self.buffer_with_slice(scalar_weights_u2, options);
        let scalar_weights_u3_buffer = self.buffer_with_slice(scalar_weights_u3, options);
        let scalar_u1_entries_buffer = self.buffer_with_slice(&scalar_u1_entries, options);
        let scalar_u2_entries_buffer = self.buffer_with_slice(&scalar_u2_entries, options);
        let scalar_u3_entries_buffer = self.buffer_with_slice(&scalar_u3_entries, options);
        let vector_weights_u1_buffer = self.buffer_with_slice(vector_weights_u1, options);
        let vector_weights_u2_buffer = self.buffer_with_slice(vector_weights_u2, options);
        let vector_weights_u3_buffer = self.buffer_with_slice(vector_weights_u3, options);
        let vector_u1_offsets_buffer = self.buffer_with_slice(&vector_u1_offsets, options);
        let vector_u2_offsets_buffer = self.buffer_with_slice(&vector_u2_offsets, options);
        let vector_u3_offsets_buffer = self.buffer_with_slice(&vector_u3_offsets, options);
        let vector_u1_entries_buffer = self.buffer_with_slice(&vector_u1_entries, options);
        let vector_u2_entries_buffer = self.buffer_with_slice(&vector_u2_entries, options);
        let vector_u3_entries_buffer = self.buffer_with_slice(&vector_u3_entries, options);
        let selector_weight_buffer = self.buffer_with_slice(selector_weight, options);
        let selector_output_mask_buffer = self.buffer_with_slice(selector_output_mask, options);
        let density_buffer = self.buffer_with_slice(density, options);
        let linear_out_buffer = self.buffer_with_slice(linear_out, options);
        let hidden_weight_buffer = self.buffer_with_slice(hidden_weight, options);
        let hidden_output_mask_buffer = self.buffer_with_slice(hidden_output_mask, options);
        let product_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&product_params), options);
        let hidden_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&hidden_params), options);

        // The product VJP kernel accumulates into `grad_x`; zero the
        // intermediate on Metal to avoid uploading an edge-sized zero vector.
        let bar_interaction_out_len = num_nodes * HIDDEN_ROW_LEN;
        let bar_interaction_out_buffer = self.device.new_buffer(
            (bar_interaction_out_len * std::mem::size_of::<f32>()) as u64,
            options,
        );
        let bar_linear_message_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let bar_input_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_input) as u64, options);
        let bar_density_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(bar_density) as u64, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_linear_scalar_vector_vjp_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&bar_product_out_buffer), 0);
        encoder.set_buffer(3, Some(&product_linear_weight_buffer), 0);
        encoder.set_buffer(4, Some(&product_linear_output_mask_buffer), 0);
        encoder.set_buffer(5, Some(&scalar_weights_u1_buffer), 0);
        encoder.set_buffer(6, Some(&scalar_weights_u2_buffer), 0);
        encoder.set_buffer(7, Some(&scalar_weights_u3_buffer), 0);
        encoder.set_buffer(8, Some(&scalar_u1_entries_buffer), 0);
        encoder.set_buffer(9, Some(&scalar_u2_entries_buffer), 0);
        encoder.set_buffer(10, Some(&scalar_u3_entries_buffer), 0);
        encoder.set_buffer(11, Some(&vector_weights_u1_buffer), 0);
        encoder.set_buffer(12, Some(&vector_weights_u2_buffer), 0);
        encoder.set_buffer(13, Some(&vector_weights_u3_buffer), 0);
        encoder.set_buffer(14, Some(&vector_u1_offsets_buffer), 0);
        encoder.set_buffer(15, Some(&vector_u2_offsets_buffer), 0);
        encoder.set_buffer(16, Some(&vector_u3_offsets_buffer), 0);
        encoder.set_buffer(17, Some(&vector_u1_entries_buffer), 0);
        encoder.set_buffer(18, Some(&vector_u2_entries_buffer), 0);
        encoder.set_buffer(19, Some(&vector_u3_entries_buffer), 0);
        encoder.set_buffer(20, Some(&bar_interaction_out_buffer), 0);
        encoder.set_buffer(21, Some(&product_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_linear_scalar_vector_vjp_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.selector_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&selector_weight_buffer), 0);
        encoder.set_buffer(2, Some(&selector_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_interaction_out_buffer), 0);
        encoder.set_buffer(4, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(5, Some(&hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.selector_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer1_hidden_vjp_density_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&linear_out_buffer), 0);
        encoder.set_buffer(2, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(3, Some(&bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: num_nodes as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer1_hidden_vjp_density_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.layer0_hidden_vjp_input_pipeline);
        encoder.set_buffer(0, Some(&density_buffer), 0);
        encoder.set_buffer(1, Some(&hidden_weight_buffer), 0);
        encoder.set_buffer(2, Some(&hidden_output_mask_buffer), 0);
        encoder.set_buffer(3, Some(&bar_linear_message_buffer), 0);
        encoder.set_buffer(4, Some(&bar_input_buffer), 0);
        encoder.set_buffer(5, Some(&hidden_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * HIDDEN_ROW_LEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer0_hidden_vjp_input_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: both output buffers are StorageModeShared and sized exactly
        // from the destination slice lengths. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_input_buffer.contents().cast::<f32>(),
                bar_input.as_mut_ptr(),
                bar_input.len(),
            );
            std::ptr::copy_nonoverlapping(
                bar_density_buffer.contents().cast::<f32>(),
                bar_density.as_mut_ptr(),
                bar_density.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_forward_cache_f32(
        &self,
        type_ids: &[u32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    ) -> Result<Mp0bProduct0ForwardCacheMetal, String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELEMENTS: usize = 89;
        const NUM_EQUIV: usize = 3;
        let num_nodes = type_ids.len();
        if scalar_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || scalar_weights_u2.len() != NUM_ELEMENTS * 4 * NUM_FEATURES
            || scalar_weights_u3.len() != NUM_ELEMENTS * 23 * NUM_FEATURES
            || vector_weights_u1.len() != NUM_ELEMENTS * NUM_FEATURES
            || vector_weights_u2.len() != NUM_ELEMENTS * 6 * NUM_FEATURES
            || vector_weights_u3.len() != NUM_ELEMENTS * 51 * NUM_FEATURES
            || vector_u1_entries_by_equiv.len() != NUM_EQUIV
            || vector_u2_entries_by_equiv.len() != NUM_EQUIV
            || vector_u3_entries_by_equiv.len() != NUM_EQUIV
        {
            return Err("product0 contraction forward cache shape mismatch".to_owned());
        }
        if type_ids
            .iter()
            .any(|&type_id| type_id as usize >= NUM_ELEMENTS)
        {
            return Err("product0 contraction forward cache type id out of range".to_owned());
        }
        let scalar_u1_entries = pack_scalar_u1_entries(scalar_u1_entries)?;
        let scalar_u2_entries = pack_scalar_u2_entries(scalar_u2_entries)?;
        let scalar_u3_entries = pack_scalar_u3_entries(scalar_u3_entries)?;
        let (vector_u1_offsets, vector_u1_entries) =
            pack_vector_u1_entries(vector_u1_entries_by_equiv)?;
        let (vector_u2_offsets, vector_u2_entries) =
            pack_vector_u2_entries(vector_u2_entries_by_equiv)?;
        let (vector_u3_offsets, vector_u3_entries) =
            pack_vector_u3_entries(vector_u3_entries_by_equiv)?;
        let scalar_params = Mp0bProduct0ScalarVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
            u1_len: u32::try_from(scalar_u1_entries.len())
                .map_err(|_| "scalar U1 entry count exceeds Metal u32 range".to_owned())?,
            u2_len: u32::try_from(scalar_u2_entries.len())
                .map_err(|_| "scalar U2 entry count exceeds Metal u32 range".to_owned())?,
            u3_len: u32::try_from(scalar_u3_entries.len())
                .map_err(|_| "scalar U3 entry count exceeds Metal u32 range".to_owned())?,
            flags: 0,
        };
        let vector_params = Mp0bProduct0VectorVjpParams {
            num_nodes: scalar_params.num_nodes,
            num_elements: NUM_ELEMENTS as u32,
            num_features: NUM_FEATURES as u32,
            num_equivariance: NUM_EQUIV as u32,
            u1_params: 1,
            u2_params: 6,
            u3_params: 51,
            output_stride: 512,
            output_offset: 128,
        };
        let options = MTLResourceOptions::StorageModeShared;
        Ok(Mp0bProduct0ForwardCacheMetal {
            num_nodes,
            type_ids_buffer: self.buffer_with_slice(type_ids, options),
            scalar_weights_u1_buffer: self.buffer_with_slice(scalar_weights_u1, options),
            scalar_weights_u2_buffer: self.buffer_with_slice(scalar_weights_u2, options),
            scalar_weights_u3_buffer: self.buffer_with_slice(scalar_weights_u3, options),
            scalar_u1_entries_buffer: self.buffer_with_slice(&scalar_u1_entries, options),
            scalar_u2_entries_buffer: self.buffer_with_slice(&scalar_u2_entries, options),
            scalar_u3_entries_buffer: self.buffer_with_slice(&scalar_u3_entries, options),
            vector_weights_u1_buffer: self.buffer_with_slice(vector_weights_u1, options),
            vector_weights_u2_buffer: self.buffer_with_slice(vector_weights_u2, options),
            vector_weights_u3_buffer: self.buffer_with_slice(vector_weights_u3, options),
            vector_u1_offsets_buffer: self.buffer_with_slice(&vector_u1_offsets, options),
            vector_u2_offsets_buffer: self.buffer_with_slice(&vector_u2_offsets, options),
            vector_u3_offsets_buffer: self.buffer_with_slice(&vector_u3_offsets, options),
            vector_u1_entries_buffer: self.buffer_with_slice(&vector_u1_entries, options),
            vector_u2_entries_buffer: self.buffer_with_slice(&vector_u2_entries, options),
            vector_u3_entries_buffer: self.buffer_with_slice(&vector_u3_entries, options),
            scalar_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&scalar_params), options),
            vector_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&vector_params), options),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn product0_contraction_forward_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        product_symmetric: &mut [f32],
    ) -> Result<(), String> {
        self.product0_contraction_forward_with_input_cache_f32(
            type_ids,
            x,
            None,
            scalar_weights_u1,
            scalar_weights_u2,
            scalar_weights_u3,
            scalar_u1_entries,
            scalar_u2_entries,
            scalar_u3_entries,
            vector_weights_u1,
            vector_weights_u2,
            vector_weights_u3,
            vector_u1_entries_by_equiv,
            vector_u2_entries_by_equiv,
            vector_u3_entries_by_equiv,
            None,
            product_symmetric,
            true,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn product0_contraction_forward_with_input_cache_f32(
        &self,
        type_ids: &[u32],
        x: &[f32],
        x_metal_cache: Option<&Mp0bF32BufferMetal>,
        scalar_weights_u1: &[f32],
        scalar_weights_u2: &[f32],
        scalar_weights_u3: &[f32],
        scalar_u1_entries: &[super::ScalarU1EntryF32],
        scalar_u2_entries: &[super::ScalarU2EntryF32],
        scalar_u3_entries: &[super::ScalarU3EntryF32],
        vector_weights_u1: &[f32],
        vector_weights_u2: &[f32],
        vector_weights_u3: &[f32],
        vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
        vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
        vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
        static_cache: Option<&Mp0bProduct0ForwardCacheMetal>,
        product_symmetric: &mut [f32],
        read_product_symmetric: bool,
    ) -> Result<Mp0bF32BufferMetal, String> {
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        const PRODUCT_ROW_LEN: usize = 512;
        let num_nodes = static_cache.map_or(type_ids.len(), |cache| cache.num_nodes);
        let expected_x_len = num_nodes * NUM_FEATURES * NUM_ELL;
        if let Some(cache) = x_metal_cache {
            if cache.len != expected_x_len || (!x.is_empty() && x.len() != expected_x_len) {
                return Err("product0 contraction forward input cache shape mismatch".to_owned());
            }
        } else if x.len() != expected_x_len {
            return Err("product0 contraction forward shape mismatch".to_owned());
        }
        if product_symmetric.len() != num_nodes * PRODUCT_ROW_LEN {
            return Err("product0 contraction forward shape mismatch".to_owned());
        }
        let owned_static_cache;
        let cache = if let Some(cache) = static_cache {
            cache
        } else {
            owned_static_cache = self.product0_forward_cache_f32(
                type_ids,
                scalar_weights_u1,
                scalar_weights_u2,
                scalar_weights_u3,
                scalar_u1_entries,
                scalar_u2_entries,
                scalar_u3_entries,
                vector_weights_u1,
                vector_weights_u2,
                vector_weights_u3,
                vector_u1_entries_by_equiv,
                vector_u2_entries_by_equiv,
                vector_u3_entries_by_equiv,
            )?;
            &owned_static_cache
        };

        let options = MTLResourceOptions::StorageModeShared;
        let owned_x_buffer;
        let x_buffer = if let Some(cache) = x_metal_cache {
            &cache.buffer
        } else {
            owned_x_buffer = self.buffer_with_slice(x, options);
            &owned_x_buffer
        };
        let product_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(product_symmetric) as u64, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.product0_scalar_forward_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&cache.scalar_weights_u1_buffer), 0);
        encoder.set_buffer(3, Some(&cache.scalar_weights_u2_buffer), 0);
        encoder.set_buffer(4, Some(&cache.scalar_weights_u3_buffer), 0);
        encoder.set_buffer(5, Some(&cache.scalar_u1_entries_buffer), 0);
        encoder.set_buffer(6, Some(&cache.scalar_u2_entries_buffer), 0);
        encoder.set_buffer(7, Some(&cache.scalar_u3_entries_buffer), 0);
        encoder.set_buffer(8, Some(&product_buffer), 0);
        encoder.set_buffer(9, Some(&cache.scalar_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_scalar_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.product0_vector_forward_pipeline);
        encoder.set_buffer(0, Some(&cache.type_ids_buffer), 0);
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&cache.vector_weights_u1_buffer), 0);
        encoder.set_buffer(3, Some(&cache.vector_weights_u2_buffer), 0);
        encoder.set_buffer(4, Some(&cache.vector_weights_u3_buffer), 0);
        encoder.set_buffer(5, Some(&cache.vector_u1_offsets_buffer), 0);
        encoder.set_buffer(6, Some(&cache.vector_u2_offsets_buffer), 0);
        encoder.set_buffer(7, Some(&cache.vector_u3_offsets_buffer), 0);
        encoder.set_buffer(8, Some(&cache.vector_u1_entries_buffer), 0);
        encoder.set_buffer(9, Some(&cache.vector_u2_entries_buffer), 0);
        encoder.set_buffer(10, Some(&cache.vector_u3_entries_buffer), 0);
        encoder.set_buffer(11, Some(&product_buffer), 0);
        encoder.set_buffer(12, Some(&cache.vector_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: (num_nodes * NUM_FEATURES * NUM_EQUIV) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.product0_vector_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        if read_product_symmetric {
            // Safety: `product_buffer` is StorageModeShared and sized exactly
            // from the destination slice length. Both forward kernels write
            // disjoint scalar/vector regions for every node before readback.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    product_buffer.contents().cast::<f32>(),
                    product_symmetric.as_mut_ptr(),
                    product_symmetric.len(),
                );
            }
        }
        Ok(Mp0bF32BufferMetal {
            len: product_symmetric.len(),
            buffer: product_buffer,
        })
    }

    fn layer0_hidden_forward_f32(
        &self,
        input: &[f32],
        density: &[f32],
        weight: &[f32],
        output_mask: &[f32],
        output: &mut [f32],
    ) -> Result<Mp0bLayer0HiddenForwardMetal, String> {
        const ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 4 * 128 * 128;
        let num_nodes = density.len();
        if input.len() != num_nodes * ROW_LEN
            || weight.len() != WEIGHT_LEN
            || output_mask.len() != ROW_LEN
            || output.len() != input.len()
        {
            return Err("layer0 hidden forward shape mismatch".to_owned());
        }
        let params = Mp0bLayer1HiddenVjpParams {
            num_nodes: u32::try_from(num_nodes)
                .map_err(|_| "node count exceeds Metal u32 range".to_owned())?,
        };
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = self.buffer_with_slice(input, options);
        let density_buffer = self.buffer_with_slice(density, options);
        let weight_buffer = self.buffer_with_slice(weight, options);
        let output_mask_buffer = self.buffer_with_slice(output_mask, options);
        let output_buffer = self
            .device
            .new_buffer(std::mem::size_of_val(output) as u64, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layer0_hidden_forward_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&density_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&output_mask_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: output.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.layer0_hidden_forward_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `output_buffer` is StorageModeShared and sized exactly from
        // the destination slice length. The command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(Mp0bLayer0HiddenForwardMetal {
            linear_message: Mp0bF32BufferMetal {
                len: output.len(),
                buffer: output_buffer,
            },
            density: Mp0bF32BufferMetal {
                len: density.len(),
                buffer: density_buffer,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        output_dim: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        weight3: &[f32],
        hidden0_derivative: &[f32],
        hidden1_derivative: &[f32],
        hidden2_derivative: &[f32],
        bar_output: &[f32],
        bar_edge_feats: &mut [f32],
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if num_bessel == 0 {
            return Err("conv-TP weight VJP requires nonzero bessel dimension".to_owned());
        }
        if output_dim == 0 {
            return Err("conv-TP weight VJP requires nonzero output dimension".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer2 weight shape mismatch".to_owned());
        }
        if weight3.len() != HIDDEN_DIM * output_dim {
            return Err("conv-TP weight VJP layer3 weight shape mismatch".to_owned());
        }
        if hidden0_derivative.len() != hidden_len
            || hidden1_derivative.len() != hidden_len
            || hidden2_derivative.len() != hidden_len
        {
            return Err("conv-TP weight VJP derivative shape mismatch".to_owned());
        }
        if bar_output.len() != num_edges * output_dim {
            return Err("conv-TP weight VJP bar output shape mismatch".to_owned());
        }
        if bar_edge_feats.len() != num_edges * num_bessel {
            return Err("conv-TP weight VJP bar edge feature shape mismatch".to_owned());
        }

        let dense_layer3 = Mp0bDense64VjpParams {
            num_edges: u32::try_from(num_edges)
                .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?,
            out_dim: u32::try_from(output_dim)
                .map_err(|_| "output dimension exceeds Metal u32 range".to_owned())?,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
        };
        let dense64 = Mp0bDense64VjpParams {
            num_edges: dense_layer3.num_edges,
            out_dim: HIDDEN_DIM as u32,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
        };
        let edge_params = Mp0bEdgeFeatVjpParams {
            num_edges: dense_layer3.num_edges,
            num_bessel: u32::try_from(num_bessel)
                .map_err(|_| "bessel dimension exceeds Metal u32 range".to_owned())?,
            inv_norm: 1.0 / (num_bessel as f32).sqrt(),
            flags: 0,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let bar_output_buffer = self.buffer_with_slice(bar_output, options);
        let weight0_buffer = self.buffer_with_slice(weight0, options);
        let weight1_buffer = self.buffer_with_slice(weight1, options);
        let weight2_buffer = self.buffer_with_slice(weight2, options);
        let weight3_buffer = self.buffer_with_slice(weight3, options);
        let derivative0_buffer = self.buffer_with_slice(hidden0_derivative, options);
        let derivative1_buffer = self.buffer_with_slice(hidden1_derivative, options);
        let derivative2_buffer = self.buffer_with_slice(hidden2_derivative, options);
        let bar_hidden2_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden1_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden0_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_edge_feats_buffer = self.buffer_with_slice(bar_edge_feats, options);
        let dense_layer3_buffer =
            self.buffer_with_slice(std::slice::from_ref(&dense_layer3), options);
        let dense64_buffer = self.buffer_with_slice(std::slice::from_ref(&dense64), options);
        let edge_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&edge_params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        let hidden_threads_per_group =
            self.threads_per_group(&self.dense64_vjp_derivative_pipeline);
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_output_buffer,
            &weight3_buffer,
            &derivative2_buffer,
            &bar_hidden2_buffer,
            &dense_layer3_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden2_buffer,
            &weight2_buffer,
            &derivative1_buffer,
            &bar_hidden1_buffer,
            &dense64_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden1_buffer,
            &weight1_buffer,
            &derivative0_buffer,
            &bar_hidden0_buffer,
            &dense64_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );

        encoder.set_compute_pipeline_state(&self.edge_feats_vjp_add_pipeline);
        encoder.set_buffer(0, Some(&bar_hidden0_buffer), 0);
        encoder.set_buffer(1, Some(&weight0_buffer), 0);
        encoder.set_buffer(2, Some(&bar_edge_feats_buffer), 0);
        encoder.set_buffer(3, Some(&edge_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_edge_feats.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.edge_feats_vjp_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_edge_feats_buffer` is StorageModeShared, initialized
        // from `bar_edge_feats`, and has exactly `bar_edge_feats.len()`
        // contiguous f32 lanes. The command buffer completed before readback.
        let output_ptr = bar_edge_feats_buffer.contents().cast::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_ptr,
                bar_edge_feats.as_mut_ptr(),
                bar_edge_feats.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_with_metal_cache_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        output_dim: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        weight3: &[f32],
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_output: &[f32],
        bar_edge_feats: &mut [f32],
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if cache.hidden_len != hidden_len {
            return Err("conv-TP weight VJP Metal cache shape mismatch".to_owned());
        }
        if num_bessel == 0 {
            return Err("conv-TP weight VJP requires nonzero bessel dimension".to_owned());
        }
        if output_dim == 0 {
            return Err("conv-TP weight VJP requires nonzero output dimension".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight VJP layer2 weight shape mismatch".to_owned());
        }
        if weight3.len() != HIDDEN_DIM * output_dim {
            return Err("conv-TP weight VJP layer3 weight shape mismatch".to_owned());
        }
        if bar_output.len() != num_edges * output_dim {
            return Err("conv-TP weight VJP bar output shape mismatch".to_owned());
        }
        if bar_edge_feats.len() != num_edges * num_bessel {
            return Err("conv-TP weight VJP bar edge feature shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let bar_output_buffer = self.buffer_with_slice(bar_output, options);
        let bar_hidden2_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden1_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden0_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_edge_feats_buffer = self.buffer_with_slice(bar_edge_feats, options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        let hidden_threads_per_group =
            self.threads_per_group(&self.dense64_vjp_derivative_pipeline);
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_output_buffer,
            &cache.weight3_buffer,
            &cache.derivative2_buffer,
            &bar_hidden2_buffer,
            &cache.dense_layer3_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden2_buffer,
            &cache.weight2_buffer,
            &cache.derivative1_buffer,
            &bar_hidden1_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden1_buffer,
            &cache.weight1_buffer,
            &cache.derivative0_buffer,
            &bar_hidden0_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );

        encoder.set_compute_pipeline_state(&self.edge_feats_vjp_add_pipeline);
        encoder.set_buffer(0, Some(&bar_hidden0_buffer), 0);
        encoder.set_buffer(1, Some(&cache.weight0_buffer), 0);
        encoder.set_buffer(2, Some(&bar_edge_feats_buffer), 0);
        encoder.set_buffer(3, Some(&cache.edge_feat_vjp_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_edge_feats.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.edge_feats_vjp_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_edge_feats_buffer` is StorageModeShared, initialized
        // from `bar_edge_feats`, and sized exactly from the destination slice
        // length. The command buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_edge_feats_buffer.contents().cast::<f32>(),
                bar_edge_feats.as_mut_ptr(),
                bar_edge_feats.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_from_hidden2_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        hidden0_derivative: &[f32],
        hidden1_derivative: &[f32],
        hidden2_derivative: &[f32],
        bar_hidden2: &[f32],
        bar_edge_feats: &mut [f32],
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if num_bessel == 0 {
            return Err("conv-TP weight hidden2 VJP requires nonzero bessel dimension".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer2 weight shape mismatch".to_owned());
        }
        if hidden0_derivative.len() != hidden_len
            || hidden1_derivative.len() != hidden_len
            || hidden2_derivative.len() != hidden_len
            || bar_hidden2.len() != hidden_len
        {
            return Err("conv-TP weight hidden2 VJP derivative shape mismatch".to_owned());
        }
        if bar_edge_feats.len() != num_edges * num_bessel {
            return Err("conv-TP weight hidden2 VJP bar edge feature shape mismatch".to_owned());
        }

        let dense64 = Mp0bDense64VjpParams {
            num_edges: u32::try_from(num_edges)
                .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?,
            out_dim: HIDDEN_DIM as u32,
            inv_norm: 1.0 / (HIDDEN_DIM as f32).sqrt(),
        };
        let edge_params = Mp0bEdgeFeatVjpParams {
            num_edges: dense64.num_edges,
            num_bessel: u32::try_from(num_bessel)
                .map_err(|_| "bessel dimension exceeds Metal u32 range".to_owned())?,
            inv_norm: 1.0 / (num_bessel as f32).sqrt(),
            flags: 0,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let bar_hidden2_input_buffer = self.buffer_with_slice(bar_hidden2, options);
        let weight0_buffer = self.buffer_with_slice(weight0, options);
        let weight1_buffer = self.buffer_with_slice(weight1, options);
        let weight2_buffer = self.buffer_with_slice(weight2, options);
        let derivative0_buffer = self.buffer_with_slice(hidden0_derivative, options);
        let derivative1_buffer = self.buffer_with_slice(hidden1_derivative, options);
        let derivative2_buffer = self.buffer_with_slice(hidden2_derivative, options);
        let bar_hidden2_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden1_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_hidden0_buffer = self
            .device
            .new_buffer((hidden_len * std::mem::size_of::<f32>()) as u64, options);
        let bar_edge_feats_buffer = self.buffer_with_slice(bar_edge_feats, options);
        let dense64_buffer = self.buffer_with_slice(std::slice::from_ref(&dense64), options);
        let edge_params_buffer =
            self.buffer_with_slice(std::slice::from_ref(&edge_params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        let hidden_threads_per_group =
            self.threads_per_group(&self.dense64_vjp_derivative_pipeline);
        self.encode_apply_derivative64(
            &encoder,
            &bar_hidden2_input_buffer,
            &derivative2_buffer,
            &bar_hidden2_buffer,
            &dense64_buffer,
            hidden_threads,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden2_buffer,
            &weight2_buffer,
            &derivative1_buffer,
            &bar_hidden1_buffer,
            &dense64_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            &bar_hidden1_buffer,
            &weight1_buffer,
            &derivative0_buffer,
            &bar_hidden0_buffer,
            &dense64_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        encoder.set_compute_pipeline_state(&self.edge_feats_vjp_add_pipeline);
        encoder.set_buffer(0, Some(&bar_hidden0_buffer), 0);
        encoder.set_buffer(1, Some(&weight0_buffer), 0);
        encoder.set_buffer(2, Some(&bar_edge_feats_buffer), 0);
        encoder.set_buffer(3, Some(&edge_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: bar_edge_feats.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.edge_feats_vjp_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_edge_feats_buffer` is StorageModeShared, initialized
        // from `bar_edge_feats`, and sized exactly from the destination slice
        // length. The command buffer completed before readback.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bar_edge_feats_buffer.contents().cast::<f32>(),
                bar_edge_feats.as_mut_ptr(),
                bar_edge_feats.len(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_from_hidden2_with_metal_cache_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_hidden2: &[f32],
        bar_edge_feats: &mut [f32],
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if cache.hidden_len != hidden_len {
            return Err("conv-TP weight hidden2 VJP Metal cache shape mismatch".to_owned());
        }
        if num_bessel == 0 {
            return Err("conv-TP weight hidden2 VJP requires nonzero bessel dimension".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer2 weight shape mismatch".to_owned());
        }
        if bar_hidden2.len() != hidden_len {
            return Err("conv-TP weight hidden2 VJP bar hidden2 shape mismatch".to_owned());
        }
        if bar_edge_feats.len() != num_edges * num_bessel {
            return Err("conv-TP weight hidden2 VJP bar edge feature shape mismatch".to_owned());
        }

        let options = MTLResourceOptions::StorageModeShared;
        let bar_hidden2_input_buffer = self.buffer_with_slice(bar_hidden2, options);
        self.conv_tp_weights_vjp_edge_feats_from_hidden2_buffer_with_metal_cache_f32(
            num_edges,
            num_bessel,
            weight0,
            weight1,
            weight2,
            cache,
            &bar_hidden2_input_buffer,
            bar_edge_feats,
            false,
            true,
            None,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_from_hidden2_metal_buffer_with_metal_cache_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_hidden2: &Mp0bF32BufferMetal,
        bar_edge_feats: &mut [f32],
        assign_bar_edge_feats: bool,
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if bar_hidden2.len != hidden_len {
            return Err("conv-TP weight hidden2 VJP Metal input buffer shape mismatch".to_owned());
        }
        self.conv_tp_weights_vjp_edge_feats_from_hidden2_buffer_with_metal_cache_f32(
            num_edges,
            num_bessel,
            weight0,
            weight1,
            weight2,
            cache,
            &bar_hidden2.buffer,
            bar_edge_feats,
            assign_bar_edge_feats,
            true,
            None,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_vjp_edge_feats_from_hidden2_buffer_with_metal_cache_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_hidden2_input_buffer: &metal::BufferRef,
        bar_edge_feats: &mut [f32],
        assign_bar_edge_feats: bool,
        read_bar_edge_feats: bool,
        output_bar_edge_feats: Option<&Mp0bF32BufferMetal>,
    ) -> Result<Mp0bF32BufferMetal, String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        if cache.hidden_len != hidden_len {
            return Err("conv-TP weight hidden2 VJP Metal cache shape mismatch".to_owned());
        }
        if num_bessel == 0 {
            return Err("conv-TP weight hidden2 VJP requires nonzero bessel dimension".to_owned());
        }
        if weight0.len() != num_bessel * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer0 weight shape mismatch".to_owned());
        }
        if weight1.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer1 weight shape mismatch".to_owned());
        }
        if weight2.len() != HIDDEN_DIM * HIDDEN_DIM {
            return Err("conv-TP weight hidden2 VJP layer2 weight shape mismatch".to_owned());
        }
        let edge_feat_len = num_edges * num_bessel;
        if cache.edge_feat_len != edge_feat_len
            || ((read_bar_edge_feats
                || (!assign_bar_edge_feats && output_bar_edge_feats.is_none()))
                && bar_edge_feats.len() != edge_feat_len)
        {
            return Err("conv-TP weight hidden2 VJP bar edge feature shape mismatch".to_owned());
        }
        if let Some(output) = output_bar_edge_feats {
            if output.len != edge_feat_len {
                return Err(
                    "conv-TP weight hidden2 VJP output edge feature buffer shape mismatch"
                        .to_owned(),
                );
            }
        }

        if !assign_bar_edge_feats && output_bar_edge_feats.is_none() {
            self.copy_slice_to_buffer(bar_edge_feats, &cache.bar_edge_feats_scratch_buffer);
        }
        let bar_hidden2_buffer = &cache.bar_hidden2_scratch_buffer;
        let bar_hidden1_buffer = &cache.bar_hidden1_scratch_buffer;
        let bar_hidden0_buffer = &cache.bar_hidden0_scratch_buffer;
        let bar_edge_feats_buffer = output_bar_edge_feats
            .map(|output| &output.buffer)
            .unwrap_or(&cache.bar_edge_feats_scratch_buffer);
        let edge_feat_vjp_params_buffer = if assign_bar_edge_feats {
            &cache.edge_feat_vjp_assign_params_buffer
        } else {
            &cache.edge_feat_vjp_params_buffer
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        let hidden_threads_per_group =
            self.threads_per_group(&self.dense64_vjp_derivative_pipeline);
        self.encode_apply_derivative64(
            &encoder,
            &bar_hidden2_input_buffer,
            &cache.derivative2_buffer,
            bar_hidden2_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            bar_hidden2_buffer,
            &cache.weight2_buffer,
            &cache.derivative1_buffer,
            bar_hidden1_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            bar_hidden1_buffer,
            &cache.weight1_buffer,
            &cache.derivative0_buffer,
            bar_hidden0_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        encoder.set_compute_pipeline_state(&self.edge_feats_vjp_add_pipeline);
        encoder.set_buffer(0, Some(bar_hidden0_buffer), 0);
        encoder.set_buffer(1, Some(&cache.weight0_buffer), 0);
        encoder.set_buffer(2, Some(bar_edge_feats_buffer), 0);
        encoder.set_buffer(3, Some(edge_feat_vjp_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: edge_feat_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.edge_feats_vjp_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        // Safety: `bar_edge_feats_buffer` is StorageModeShared, initialized
        // from `bar_edge_feats`, and sized exactly from the destination slice
        // length. The command buffer completed before readback.
        unsafe {
            if read_bar_edge_feats {
                std::ptr::copy_nonoverlapping(
                    bar_edge_feats_buffer.contents().cast::<f32>(),
                    bar_edge_feats.as_mut_ptr(),
                    bar_edge_feats.len(),
                );
            }
        }
        Ok(Mp0bF32BufferMetal {
            len: edge_feat_len,
            buffer: (*bar_edge_feats_buffer).clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_tp_weights_and_density_vjp_edge_feats_from_hidden2_metal_buffer_to_metal_with_metal_cache_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight0: &[f32],
        weight1: &[f32],
        weight2: &[f32],
        density_weight: &[f32],
        bar_density: &[f32],
        bar_density_metal: Option<&Mp0bF32BufferMetal>,
        topology: &Mp0bLayer1ConvTpTopologyMetal,
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_hidden2: &Mp0bF32BufferMetal,
        output_bar_edge_feats: Option<&Mp0bF32BufferMetal>,
        assign_bar_edge_feats: bool,
        wait_for_completion: bool,
    ) -> Result<Mp0bF32BufferMetal, String> {
        const HIDDEN_DIM: usize = 64;
        let hidden_len = num_edges * HIDDEN_DIM;
        let edge_feat_len = num_edges * num_bessel;
        if bar_hidden2.len != hidden_len || cache.hidden_len != hidden_len {
            return Err(
                "combined conv-TP/density VJP Metal hidden buffer shape mismatch".to_owned(),
            );
        }
        if topology.num_edges != num_edges || cache.edge_feat_len != edge_feat_len {
            return Err("combined conv-TP/density VJP Metal topology shape mismatch".to_owned());
        }
        if num_bessel == 0 {
            return Err(
                "combined conv-TP/density VJP requires nonzero bessel dimension".to_owned(),
            );
        }
        if weight0.len() != num_bessel * HIDDEN_DIM
            || weight1.len() != HIDDEN_DIM * HIDDEN_DIM
            || weight2.len() != HIDDEN_DIM * HIDDEN_DIM
        {
            return Err("combined conv-TP/density VJP layer weight shape mismatch".to_owned());
        }
        if density_weight.len() != num_bessel {
            return Err("combined conv-TP/density VJP density shape mismatch".to_owned());
        }
        if let Some(cache) = bar_density_metal {
            if cache.len != topology.num_nodes
                || (!bar_density.is_empty() && bar_density.len() != topology.num_nodes)
            {
                return Err("combined conv-TP/density VJP density Metal shape mismatch".to_owned());
            }
        } else if bar_density.len() != topology.num_nodes {
            return Err("combined conv-TP/density VJP density shape mismatch".to_owned());
        }
        if !assign_bar_edge_feats && output_bar_edge_feats.is_none() {
            return Err(
                "combined conv-TP/density VJP add mode requires an output edge feature buffer"
                    .to_owned(),
            );
        }
        if let Some(output) = output_bar_edge_feats {
            if output.len != edge_feat_len {
                return Err(
                    "combined conv-TP/density VJP output edge feature shape mismatch".to_owned(),
                );
            }
        }

        let params = Mp0bDensityVjpEdgeFeatParams {
            num_edges: u32::try_from(num_edges)
                .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?,
            num_bessel: u32::try_from(num_bessel)
                .map_err(|_| "bessel dimension exceeds Metal u32 range".to_owned())?,
        };
        let options = MTLResourceOptions::StorageModeShared;
        let density_weight_buffer = self.buffer_with_slice(density_weight, options);
        let owned_bar_density_buffer;
        let bar_density_buffer = if let Some(cache) = bar_density_metal {
            &cache.buffer
        } else {
            owned_bar_density_buffer = self.buffer_with_slice(bar_density, options);
            &owned_bar_density_buffer
        };
        let density_params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let bar_hidden2_buffer = &cache.bar_hidden2_scratch_buffer;
        let bar_hidden1_buffer = &cache.bar_hidden1_scratch_buffer;
        let bar_hidden0_buffer = &cache.bar_hidden0_scratch_buffer;
        let bar_edge_feats_buffer = output_bar_edge_feats
            .map(|output| &output.buffer)
            .unwrap_or(&cache.bar_edge_feats_scratch_buffer);
        let edge_feat_vjp_params_buffer = if assign_bar_edge_feats {
            &cache.edge_feat_vjp_assign_params_buffer
        } else {
            &cache.edge_feat_vjp_params_buffer
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_threads = hidden_len as u64;
        let hidden_threads_per_group =
            self.threads_per_group(&self.dense64_vjp_derivative_pipeline);
        self.encode_apply_derivative64(
            &encoder,
            &bar_hidden2.buffer,
            &cache.derivative2_buffer,
            bar_hidden2_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            bar_hidden2_buffer,
            &cache.weight2_buffer,
            &cache.derivative1_buffer,
            bar_hidden1_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        self.encode_dense64_vjp_derivative(
            &encoder,
            bar_hidden1_buffer,
            &cache.weight1_buffer,
            &cache.derivative0_buffer,
            bar_hidden0_buffer,
            &cache.dense64_vjp_params_buffer,
            hidden_threads,
            hidden_threads_per_group,
        );
        encoder.set_compute_pipeline_state(&self.edge_feats_vjp_add_pipeline);
        encoder.set_buffer(0, Some(bar_hidden0_buffer), 0);
        encoder.set_buffer(1, Some(&cache.weight0_buffer), 0);
        encoder.set_buffer(2, Some(bar_edge_feats_buffer), 0);
        encoder.set_buffer(3, Some(edge_feat_vjp_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: edge_feat_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.edge_feats_vjp_add_pipeline),
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.density_vjp_edge_feats_add_pipeline);
        encoder.set_buffer(0, Some(&cache.edge_feats_buffer), 0);
        encoder.set_buffer(1, Some(&density_weight_buffer), 0);
        encoder.set_buffer(2, Some(&topology.receivers_buffer), 0);
        encoder.set_buffer(3, Some(bar_density_buffer), 0);
        encoder.set_buffer(4, Some(bar_edge_feats_buffer), 0);
        encoder.set_buffer(5, Some(&density_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: edge_feat_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.density_vjp_edge_feats_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();

        if wait_for_completion {
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                return Err(format!("Metal command buffer did not complete: {status:?}"));
            }
        }
        // Safety: the returned buffer is StorageModeShared and sized exactly
        // from `edge_feat_len`. When CPU readback follows immediately, callers
        // request completion here; otherwise subsequent Metal work remains
        // ordered behind this command buffer on the same queue.
        Ok(Mp0bF32BufferMetal {
            len: edge_feat_len,
            buffer: (*bar_edge_feats_buffer).clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn density_vjp_edge_feats_add_to_metal_buffer_f32(
        &self,
        num_edges: usize,
        num_bessel: usize,
        weight: &[f32],
        bar_density: &[f32],
        topology: &Mp0bLayer1ConvTpTopologyMetal,
        cache: &Mp0bConvTpWeightCacheMetal,
        bar_edge_feats: &Mp0bF32BufferMetal,
    ) -> Result<(), String> {
        let edge_feat_len = num_edges * num_bessel;
        if topology.num_edges != num_edges
            || cache.edge_feat_len != edge_feat_len
            || bar_edge_feats.len != edge_feat_len
            || weight.len() != num_bessel
        {
            return Err("density VJP Metal edge feature shape mismatch".to_owned());
        }
        if bar_density.len() != topology.num_nodes {
            return Err("density VJP Metal bar density shape mismatch".to_owned());
        }
        let params = Mp0bDensityVjpEdgeFeatParams {
            num_edges: u32::try_from(num_edges)
                .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?,
            num_bessel: u32::try_from(num_bessel)
                .map_err(|_| "bessel dimension exceeds Metal u32 range".to_owned())?,
        };
        let options = MTLResourceOptions::StorageModeShared;
        let weight_buffer = self.buffer_with_slice(weight, options);
        let bar_density_buffer = self.buffer_with_slice(bar_density, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.density_vjp_edge_feats_add_pipeline);
        encoder.set_buffer(0, Some(&cache.edge_feats_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&topology.receivers_buffer), 0);
        encoder.set_buffer(3, Some(&bar_density_buffer), 0);
        encoder.set_buffer(4, Some(&bar_edge_feats.buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: edge_feat_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.density_vjp_edge_feats_add_pipeline),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer did not complete: {status:?}"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_apply_derivative64(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bar_in: &metal::BufferRef,
        derivative: &metal::BufferRef,
        bar_out: &metal::BufferRef,
        params: &metal::BufferRef,
        total_threads: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.apply_derivative64_pipeline);
        encoder.set_buffer(0, Some(bar_in), 0);
        encoder.set_buffer(1, Some(derivative), 0);
        encoder.set_buffer(2, Some(bar_out), 0);
        encoder.set_buffer(3, Some(params), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.apply_derivative64_pipeline),
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dense64_vjp_derivative(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bar_out: &metal::BufferRef,
        weight: &metal::BufferRef,
        derivative: &metal::BufferRef,
        bar_in: &metal::BufferRef,
        params: &metal::BufferRef,
        total_threads: u64,
        threads_per_group: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.dense64_vjp_derivative_pipeline);
        encoder.set_buffer(0, Some(bar_out), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(derivative), 0);
        encoder.set_buffer(3, Some(bar_in), 0);
        encoder.set_buffer(4, Some(params), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
    }

    fn threads_per_group(&self, pipeline: &ComputePipelineState) -> u64 {
        let execution_width = pipeline.thread_execution_width().max(1);
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
        execution_width.min(max_threads).max(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dense_to_64_silu_derivative(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        weight: &metal::BufferRef,
        output: &metal::BufferRef,
        derivative: &metal::BufferRef,
        params: &metal::BufferRef,
        total_threads: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.dense_to_64_silu_derivative_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_buffer(3, Some(derivative), 0);
        encoder.set_buffer(4, Some(params), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.dense_to_64_silu_derivative_pipeline),
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dense64_silu_derivative(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        weight: &metal::BufferRef,
        output: &metal::BufferRef,
        derivative: &metal::BufferRef,
        params: &metal::BufferRef,
        total_threads: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.dense64_silu_derivative_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_buffer(3, Some(derivative), 0);
        encoder.set_buffer(4, Some(params), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads_per_group(&self.dense64_silu_derivative_pipeline),
                height: 1,
                depth: 1,
            },
        );
    }

    fn buffer_with_slice<T>(&self, values: &[T], options: MTLResourceOptions) -> Buffer {
        buffer_with_slice_on_device(&self.device, values, options)
    }

    fn layer1_hidden_weights_cache_f32(
        &self,
        weight: &[f32],
        output_mask: &[f32],
    ) -> Result<Mp0bLayer1HiddenWeightsMetal, String> {
        const OUTPUT_ROW_LEN: usize = 2048;
        const WEIGHT_LEN: usize = 10 * 128 * 128;
        if weight.len() != WEIGHT_LEN || output_mask.len() != OUTPUT_ROW_LEN {
            return Err("layer1 hidden static weight cache shape mismatch".to_owned());
        }
        let options = MTLResourceOptions::StorageModeShared;
        Ok(Mp0bLayer1HiddenWeightsMetal {
            weight_len: weight.len(),
            output_mask_len: output_mask.len(),
            weight_buffer: self.buffer_with_slice(weight, options),
            output_mask_buffer: self.buffer_with_slice(output_mask, options),
        })
    }

    fn copy_f32_buffer_to_slice(
        &self,
        source: &Mp0bF32BufferMetal,
        output: &mut [f32],
    ) -> Result<(), String> {
        if source.len != output.len() {
            return Err("Metal f32 buffer readback shape mismatch".to_owned());
        }
        // Safety: StorageModeShared exposes CPU-readable contents, and callers
        // use this only after the producing command buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.buffer.contents().cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }

    fn copy_slice_to_buffer<T>(&self, source: &[T], output: &Buffer) {
        if source.is_empty() {
            return;
        }
        // Safety: callers pass a StorageModeShared buffer sized for `source`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr().cast::<u8>(),
                output.contents().cast::<u8>(),
                std::mem::size_of_val(source),
            );
        }
    }

    fn layer1_conv_tp_topology_buffers(
        &self,
        num_nodes: usize,
        senders: &[usize],
        receivers: &[usize],
        options: MTLResourceOptions,
    ) -> Result<Mp0bLayer1ConvTpTopologyMetal, String> {
        let num_edges = senders.len();
        if receivers.len() != num_edges {
            return Err("layer1 conv-TP topology sender/receiver shape mismatch".to_owned());
        }

        let mut receiver_offsets = vec![0_u32; num_nodes + 1];
        let mut sender_offsets = vec![0_u32; num_nodes + 1];
        let mut senders_u32 = Vec::with_capacity(num_edges);
        let mut receivers_u32 = Vec::with_capacity(num_edges);
        for (&sender, &receiver) in senders.iter().zip(receivers) {
            if sender >= num_nodes {
                return Err("layer1 conv-TP topology sender out of range".to_owned());
            }
            if receiver >= num_nodes {
                return Err("layer1 conv-TP topology receiver out of range".to_owned());
            }
            sender_offsets[sender + 1] = sender_offsets[sender + 1]
                .checked_add(1)
                .ok_or_else(|| "layer1 conv-TP topology sender count overflow".to_owned())?;
            receiver_offsets[receiver + 1] = receiver_offsets[receiver + 1]
                .checked_add(1)
                .ok_or_else(|| "layer1 conv-TP topology receiver count overflow".to_owned())?;
            senders_u32.push(
                u32::try_from(sender)
                    .map_err(|_| "sender index exceeds Metal u32 range".to_owned())?,
            );
            receivers_u32.push(
                u32::try_from(receiver)
                    .map_err(|_| "receiver index exceeds Metal u32 range".to_owned())?,
            );
        }
        for node in 1..=num_nodes {
            sender_offsets[node] = sender_offsets[node]
                .checked_add(sender_offsets[node - 1])
                .ok_or_else(|| "layer1 conv-TP topology sender offset overflow".to_owned())?;
            receiver_offsets[node] = receiver_offsets[node]
                .checked_add(receiver_offsets[node - 1])
                .ok_or_else(|| "layer1 conv-TP topology receiver offset overflow".to_owned())?;
        }

        let mut next_sender_offsets = sender_offsets[..num_nodes].to_vec();
        let mut sender_edges = vec![0_u32; num_edges];
        let mut next_receiver_offsets = receiver_offsets[..num_nodes].to_vec();
        let mut receiver_edges = vec![0_u32; num_edges];
        for (edge, (&sender, &receiver)) in senders.iter().zip(receivers).enumerate() {
            let sender_target = next_sender_offsets[sender] as usize;
            sender_edges[sender_target] =
                u32::try_from(edge).map_err(|_| "edge count exceeds Metal u32 range".to_owned())?;
            next_sender_offsets[sender] += 1;

            let receiver_target = next_receiver_offsets[receiver] as usize;
            receiver_edges[receiver_target] =
                u32::try_from(edge).map_err(|_| "edge count exceeds Metal u32 range".to_owned())?;
            next_receiver_offsets[receiver] += 1;
        }

        let num_nodes_u32 = u32::try_from(num_nodes)
            .map_err(|_| "node count exceeds Metal u32 range".to_owned())?;
        let num_edges_u32 = u32::try_from(num_edges)
            .map_err(|_| "edge count exceeds Metal u32 range".to_owned())?;
        let forward_params = Mp0bLayer1HiddenVjpParams {
            num_nodes: num_nodes_u32,
        };
        let vjp_params = Mp0bLayer1ConvTpVjpParams {
            num_nodes: num_nodes_u32,
            num_edges: num_edges_u32,
            flags: 0,
        };

        Ok(Mp0bLayer1ConvTpTopologyMetal {
            num_nodes,
            num_edges,
            receiver_offsets_buffer: self.buffer_with_slice(&receiver_offsets, options),
            receiver_edges_buffer: self.buffer_with_slice(&receiver_edges, options),
            sender_offsets_buffer: self.buffer_with_slice(&sender_offsets, options),
            sender_edges_buffer: self.buffer_with_slice(&sender_edges, options),
            senders_buffer: self.buffer_with_slice(&senders_u32, options),
            receivers_buffer: self.buffer_with_slice(&receivers_u32, options),
            forward_params_buffer: self
                .buffer_with_slice(std::slice::from_ref(&forward_params), options),
            vjp_params_buffer: self.buffer_with_slice(std::slice::from_ref(&vjp_params), options),
        })
    }

    fn layer1_conv_tp_topology_f32(
        &self,
        num_nodes: usize,
        senders: &[usize],
        receivers: &[usize],
    ) -> Result<Mp0bLayer1ConvTpTopologyMetal, String> {
        self.layer1_conv_tp_topology_buffers(
            num_nodes,
            senders,
            receivers,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

fn buffer_with_slice_on_device<T>(
    device: &Device,
    values: &[T],
    options: MTLResourceOptions,
) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(values) as u64,
        options,
    )
}

fn pack_layer1_tp_metadata() -> Result<Layer1TpMetadataMetal, String> {
    let templates = super::MP0B_LAYER1_TP_TEMPLATES
        .iter()
        .map(|template| {
            Ok(Mp0bLayer1TpTemplateMetal {
                x_base: u32::try_from(template.x_base)
                    .map_err(|_| "template x_base exceeds Metal u32 range".to_owned())?,
                x_dim: u32::try_from(template.x_dim)
                    .map_err(|_| "template x_dim exceeds Metal u32 range".to_owned())?,
                y_base: u32::try_from(template.y_base)
                    .map_err(|_| "template y_base exceeds Metal u32 range".to_owned())?,
                out_base: u32::try_from(template.out_base)
                    .map_err(|_| "template out_base exceeds Metal u32 range".to_owned())?,
                out_dim: u32::try_from(template.out_dim)
                    .map_err(|_| "template out_dim exceeds Metal u32 range".to_owned())?,
                weight_base: u32::try_from(template.weight_base)
                    .map_err(|_| "template weight_base exceeds Metal u32 range".to_owned())?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut cg_offsets = Vec::with_capacity(super::MP0B_LAYER1_TP_CG_TEMPLATES_F32.len() + 1);
    let mut cgs = Vec::new();
    cg_offsets.push(0_u32);
    for cg_entries in super::MP0B_LAYER1_TP_CG_TEMPLATES_F32 {
        for cg in cg_entries {
            cgs.push(Mp0bLayer1CgMetal {
                i_m: u32::try_from(cg.i_m)
                    .map_err(|_| "CG i_m exceeds Metal u32 range".to_owned())?,
                j_m: u32::try_from(cg.j_m)
                    .map_err(|_| "CG j_m exceeds Metal u32 range".to_owned())?,
                k_m: u32::try_from(cg.k_m)
                    .map_err(|_| "CG k_m exceeds Metal u32 range".to_owned())?,
                scale: cg.scale,
            });
        }
        cg_offsets.push(
            u32::try_from(cgs.len()).map_err(|_| "CG offset exceeds Metal u32 range".to_owned())?,
        );
    }
    let mut forward_cg_offsets = Vec::with_capacity(templates.len() * 8 + 1);
    let mut forward_cgs = Vec::new();
    forward_cg_offsets.push(0_u32);
    for cg_entries in super::MP0B_LAYER1_TP_CG_TEMPLATES_F32 {
        for output_component in 0..8 {
            for cg in cg_entries {
                if cg.k_m != output_component {
                    continue;
                }
                forward_cgs.push(Mp0bLayer1CgMetal {
                    i_m: u32::try_from(cg.i_m)
                        .map_err(|_| "CG i_m exceeds Metal u32 range".to_owned())?,
                    j_m: u32::try_from(cg.j_m)
                        .map_err(|_| "CG j_m exceeds Metal u32 range".to_owned())?,
                    k_m: u32::try_from(cg.k_m)
                        .map_err(|_| "CG k_m exceeds Metal u32 range".to_owned())?,
                    scale: cg.scale,
                });
            }
            forward_cg_offsets.push(
                u32::try_from(forward_cgs.len())
                    .map_err(|_| "forward CG grouped offset exceeds Metal u32 range".to_owned())?,
            );
        }
    }
    let mut vjp_i_cg_offsets = Vec::with_capacity(templates.len() * 8 + 1);
    let mut vjp_i_cgs = Vec::new();
    vjp_i_cg_offsets.push(0_u32);
    for cg_entries in super::MP0B_LAYER1_TP_CG_TEMPLATES_F32 {
        for input_component in 0..8 {
            for cg in cg_entries {
                if cg.i_m != input_component {
                    continue;
                }
                vjp_i_cgs.push(Mp0bLayer1CgMetal {
                    i_m: u32::try_from(cg.i_m)
                        .map_err(|_| "CG i_m exceeds Metal u32 range".to_owned())?,
                    j_m: u32::try_from(cg.j_m)
                        .map_err(|_| "CG j_m exceeds Metal u32 range".to_owned())?,
                    k_m: u32::try_from(cg.k_m)
                        .map_err(|_| "CG k_m exceeds Metal u32 range".to_owned())?,
                    scale: cg.scale,
                });
            }
            vjp_i_cg_offsets.push(
                u32::try_from(vjp_i_cgs.len()).map_err(|_| {
                    "VJP input CG grouped offset exceeds Metal u32 range".to_owned()
                })?,
            );
        }
    }
    let mut weight_lanes = vec![
        Mp0bLayer1WeightLaneMetal {
            template_index: 0,
            feature: 0,
        };
        super::MP0B_LAYER1_TP_TEMPLATES.len() * super::MP0B_NUM_FEATURES
    ];
    let mut input_entries_by_lane = vec![Vec::new(); super::MP0B_PRODUCT0_ROW_LEN];
    for (template_index, template) in super::MP0B_LAYER1_TP_TEMPLATES.iter().enumerate() {
        for feature in 0..super::MP0B_NUM_FEATURES {
            let lane = template.weight_base + feature;
            let entry = weight_lanes
                .get_mut(lane)
                .ok_or_else(|| "layer1 weight lane index out of range".to_owned())?;
            *entry = Mp0bLayer1WeightLaneMetal {
                template_index: u32::try_from(template_index)
                    .map_err(|_| "template index exceeds Metal u32 range".to_owned())?,
                feature: u32::try_from(feature)
                    .map_err(|_| "feature index exceeds Metal u32 range".to_owned())?,
            };
            for input_component in 0..template.x_dim {
                if input_component >= 8 {
                    return Err("layer1 input component exceeds grouped CG range".to_owned());
                }
                let input_lane = template.x_base + feature * template.x_dim + input_component;
                let entries = input_entries_by_lane
                    .get_mut(input_lane)
                    .ok_or_else(|| "layer1 input lane index out of range".to_owned())?;
                entries.push(Mp0bLayer1InputLaneMetal {
                    template_index: u32::try_from(template_index)
                        .map_err(|_| "template index exceeds Metal u32 range".to_owned())?,
                    feature: u32::try_from(feature)
                        .map_err(|_| "feature index exceeds Metal u32 range".to_owned())?,
                    input_component: u32::try_from(input_component)
                        .map_err(|_| "input component exceeds Metal u32 range".to_owned())?,
                });
            }
        }
    }
    let mut input_lane_offsets = Vec::with_capacity(input_entries_by_lane.len() + 1);
    let mut input_lane_entries = Vec::new();
    input_lane_offsets.push(0_u32);
    for entries in input_entries_by_lane {
        input_lane_entries.extend(entries);
        input_lane_offsets.push(
            u32::try_from(input_lane_entries.len())
                .map_err(|_| "layer1 input lane offset exceeds Metal u32 range".to_owned())?,
        );
    }
    Ok((
        templates,
        cg_offsets,
        cgs,
        forward_cg_offsets,
        forward_cgs,
        vjp_i_cg_offsets,
        vjp_i_cgs,
        input_lane_offsets,
        input_lane_entries,
        weight_lanes,
    ))
}

pub(super) struct Mp0bConvTpWeightsForwardMetal {
    pub(super) output: Vec<f32>,
    pub(super) hidden0_derivative: Vec<f32>,
    pub(super) hidden1_derivative: Vec<f32>,
    pub(super) hidden2_derivative: Vec<f32>,
    pub(super) metal_cache: Mp0bConvTpWeightCacheMetal,
}

#[derive(Clone)]
pub(super) struct Mp0bConvTpWeightCacheMetal {
    hidden_len: usize,
    output_len: usize,
    edge_feat_len: usize,
    edge_feats_buffer: Buffer,
    output_buffer: Buffer,
    derivative0_buffer: Buffer,
    derivative1_buffer: Buffer,
    derivative2_buffer: Buffer,
    weight0_buffer: Buffer,
    weight1_buffer: Buffer,
    weight2_buffer: Buffer,
    weight3_buffer: Buffer,
    dense_layer3_vjp_params_buffer: Buffer,
    dense64_vjp_params_buffer: Buffer,
    edge_feat_vjp_params_buffer: Buffer,
    edge_feat_vjp_assign_params_buffer: Buffer,
    bar_hidden2_scratch_buffer: Buffer,
    bar_hidden1_scratch_buffer: Buffer,
    bar_hidden0_scratch_buffer: Buffer,
    bar_edge_feats_scratch_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bConvTpWeightCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bConvTpWeightCacheMetal")
            .field("hidden_len", &self.hidden_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bProduct0Layer0VjpMetal {
    pub(super) bar_input: Mp0bF32BufferMetal,
    pub(super) bar_density: Mp0bF32BufferMetal,
}

#[derive(Clone)]
pub(super) struct Mp0bProduct0ScalarVjpCacheMetal {
    num_nodes: usize,
    type_ids_buffer: Buffer,
    weights_u1_buffer: Buffer,
    weights_u2_buffer: Buffer,
    weights_u3_buffer: Buffer,
    u1_entries_buffer: Buffer,
    u2_entries_buffer: Buffer,
    u3_entries_buffer: Buffer,
    params_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bProduct0ScalarVjpCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bProduct0ScalarVjpCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct MaceScalarVjpCacheMetal {
    num_nodes: usize,
    num_features: usize,
    num_ell: usize,
    type_ids_buffer: Buffer,
    weights_u1_buffer: Buffer,
    weights_u2_buffer: Buffer,
    weights_u3_buffer: Buffer,
    u1_entries_buffer: Buffer,
    u2_entries_buffer: Buffer,
    u3_entries_buffer: Buffer,
    params_buffer: Buffer,
}

impl std::fmt::Debug for MaceScalarVjpCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaceScalarVjpCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .field("num_features", &self.num_features)
            .field("num_ell", &self.num_ell)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bProduct0ForwardCacheMetal {
    num_nodes: usize,
    type_ids_buffer: Buffer,
    scalar_weights_u1_buffer: Buffer,
    scalar_weights_u2_buffer: Buffer,
    scalar_weights_u3_buffer: Buffer,
    scalar_u1_entries_buffer: Buffer,
    scalar_u2_entries_buffer: Buffer,
    scalar_u3_entries_buffer: Buffer,
    vector_weights_u1_buffer: Buffer,
    vector_weights_u2_buffer: Buffer,
    vector_weights_u3_buffer: Buffer,
    vector_u1_offsets_buffer: Buffer,
    vector_u2_offsets_buffer: Buffer,
    vector_u3_offsets_buffer: Buffer,
    vector_u1_entries_buffer: Buffer,
    vector_u2_entries_buffer: Buffer,
    vector_u3_entries_buffer: Buffer,
    scalar_params_buffer: Buffer,
    vector_params_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bProduct0ForwardCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bProduct0ForwardCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bProduct0Layer0VjpCacheMetal {
    num_nodes: usize,
    type_ids_buffer: Buffer,
    product_linear_weight_buffer: Buffer,
    product_linear_output_mask_buffer: Buffer,
    scalar_weights_u1_buffer: Buffer,
    scalar_weights_u2_buffer: Buffer,
    scalar_weights_u3_buffer: Buffer,
    scalar_u1_entries_buffer: Buffer,
    scalar_u2_entries_buffer: Buffer,
    scalar_u3_entries_buffer: Buffer,
    vector_weights_u1_buffer: Buffer,
    vector_weights_u2_buffer: Buffer,
    vector_weights_u3_buffer: Buffer,
    vector_u1_offsets_buffer: Buffer,
    vector_u2_offsets_buffer: Buffer,
    vector_u3_offsets_buffer: Buffer,
    vector_u1_entries_buffer: Buffer,
    vector_u2_entries_buffer: Buffer,
    vector_u3_entries_buffer: Buffer,
    selector_weight_buffer: Buffer,
    selector_output_mask_buffer: Buffer,
    hidden_weight_buffer: Buffer,
    hidden_output_mask_buffer: Buffer,
    product_params_buffer: Buffer,
    hidden_params_buffer: Buffer,
    bar_interaction_out_buffer: Buffer,
    bar_linear_message_buffer: Buffer,
    bar_input_buffer: Buffer,
    bar_density_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bProduct0Layer0VjpCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bProduct0Layer0VjpCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bLayer1ConvTpVjpFixedCacheMetal {
    num_nodes: usize,
    num_edges: usize,
    edge_attrs_buffer: Buffer,
    layer3_weight_buffer: Buffer,
    linear_up_weight_buffer: Buffer,
    linear_up_output_mask_buffer: Buffer,
    bar_linear_up_buffer: Buffer,
    bar_edge_attrs_buffer: Buffer,
    bar_hidden2_buffer: Buffer,
    linear_up_vjp_params_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bLayer1ConvTpVjpFixedCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bLayer1ConvTpVjpFixedCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .field("num_edges", &self.num_edges)
            .finish_non_exhaustive()
    }
}

pub(super) struct Mp0bLayer1ConvTpVjpHidden2Metal {
    pub(super) bar_linear_up: Vec<f32>,
    pub(super) bar_edge_attrs: Vec<f32>,
    pub(super) bar_edge_attrs_metal: Option<Mp0bF32BufferMetal>,
    pub(super) bar_hidden2: Vec<f32>,
    pub(super) bar_hidden2_metal: Option<Mp0bF32BufferMetal>,
    pub(super) bar_layer0_product_out: Option<Vec<f32>>,
    pub(super) bar_layer0_product_out_metal: Option<Mp0bF32BufferMetal>,
}

#[derive(Clone)]
pub(super) struct Mp0bLayer0ConvTpVjpFixedCacheMetal {
    num_nodes: usize,
    num_edges: usize,
    edge_attrs_buffer: Buffer,
    layer3_weight_buffer: Buffer,
    bar_edge_attrs_buffer: Buffer,
    bar_hidden2_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bLayer0ConvTpVjpFixedCacheMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bLayer0ConvTpVjpFixedCacheMetal")
            .field("num_nodes", &self.num_nodes)
            .field("num_edges", &self.num_edges)
            .finish_non_exhaustive()
    }
}

pub(super) struct Mp0bLayer0ConvTpVjpHidden2Metal {
    pub(super) bar_edge_attrs: Vec<f32>,
    pub(super) bar_edge_attrs_metal: Option<Mp0bF32BufferMetal>,
    pub(super) bar_hidden2_metal: Mp0bF32BufferMetal,
}

pub(super) struct Mp0bLayer1ConvTpForwardMetal {
    pub(super) topology: Mp0bLayer1ConvTpTopologyMetal,
    pub(super) scatter_message_metal: Option<Mp0bF32BufferMetal>,
}

#[derive(Clone)]
pub(super) struct Mp0bLayer1ConvTpTopologyMetal {
    num_nodes: usize,
    num_edges: usize,
    receiver_offsets_buffer: Buffer,
    receiver_edges_buffer: Buffer,
    sender_offsets_buffer: Buffer,
    sender_edges_buffer: Buffer,
    senders_buffer: Buffer,
    receivers_buffer: Buffer,
    forward_params_buffer: Buffer,
    vjp_params_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bLayer1ConvTpTopologyMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bLayer1ConvTpTopologyMetal")
            .field("num_nodes", &self.num_nodes)
            .field("num_edges", &self.num_edges)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bF32BufferMetal {
    len: usize,
    buffer: Buffer,
}

impl std::fmt::Debug for Mp0bF32BufferMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bF32BufferMetal")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct Mp0bLayer1HiddenVjpMetal {
    pub(super) bar_input: Mp0bF32BufferMetal,
    pub(super) bar_density: Mp0bF32BufferMetal,
}

#[derive(Clone, Debug)]
pub(super) struct Mp0bLayer0HiddenForwardMetal {
    pub(super) linear_message: Mp0bF32BufferMetal,
    pub(super) density: Mp0bF32BufferMetal,
}

#[derive(Clone, Debug)]
pub(super) struct Mp0bLayer1HiddenForwardMetal {
    pub(super) linear_message: Mp0bF32BufferMetal,
    pub(super) density: Mp0bF32BufferMetal,
}

#[derive(Clone)]
pub(super) struct Mp0bLayer1HiddenWeightsMetal {
    weight_len: usize,
    output_mask_len: usize,
    weight_buffer: Buffer,
    output_mask_buffer: Buffer,
}

impl std::fmt::Debug for Mp0bLayer1HiddenWeightsMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp0bLayer1HiddenWeightsMetal")
            .field("weight_len", &self.weight_len)
            .field("output_mask_len", &self.output_mask_len)
            .finish_non_exhaustive()
    }
}

fn pack_vector_u1_entries(
    entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
) -> Result<(Vec<u32>, Vec<Mp0bVectorU1EntryMetal>), String> {
    let mut offsets = Vec::with_capacity(entries_by_equiv.len() + 1);
    let mut entries = Vec::new();
    offsets.push(0_u32);
    for equiv_entries in entries_by_equiv {
        for entry in equiv_entries {
            entries.push(Mp0bVectorU1EntryMetal {
                ell: u32::try_from(entry.ell)
                    .map_err(|_| "vector U1 ell exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "vector U1 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            });
        }
        offsets.push(
            u32::try_from(entries.len())
                .map_err(|_| "vector U1 offset exceeds Metal u32 range".to_owned())?,
        );
    }
    Ok((offsets, entries))
}

fn pack_vector_u2_entries(
    entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
) -> Result<(Vec<u32>, Vec<Mp0bVectorU2EntryMetal>), String> {
    let mut offsets = Vec::with_capacity(entries_by_equiv.len() + 1);
    let mut entries = Vec::new();
    offsets.push(0_u32);
    for equiv_entries in entries_by_equiv {
        for entry in equiv_entries {
            entries.push(Mp0bVectorU2EntryMetal {
                left: u32::try_from(entry.left)
                    .map_err(|_| "vector U2 left exceeds Metal u32 range".to_owned())?,
                right: u32::try_from(entry.right)
                    .map_err(|_| "vector U2 right exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "vector U2 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            });
        }
        offsets.push(
            u32::try_from(entries.len())
                .map_err(|_| "vector U2 offset exceeds Metal u32 range".to_owned())?,
        );
    }
    Ok((offsets, entries))
}

fn pack_vector_u3_entries(
    entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
) -> Result<(Vec<u32>, Vec<Mp0bVectorU3EntryMetal>), String> {
    let mut offsets = Vec::with_capacity(entries_by_equiv.len() + 1);
    let mut entries = Vec::new();
    offsets.push(0_u32);
    for equiv_entries in entries_by_equiv {
        for entry in equiv_entries {
            entries.push(Mp0bVectorU3EntryMetal {
                left: u32::try_from(entry.left)
                    .map_err(|_| "vector U3 left exceeds Metal u32 range".to_owned())?,
                right: u32::try_from(entry.right)
                    .map_err(|_| "vector U3 right exceeds Metal u32 range".to_owned())?,
                inner: u32::try_from(entry.inner)
                    .map_err(|_| "vector U3 inner exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "vector U3 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            });
        }
        offsets.push(
            u32::try_from(entries.len())
                .map_err(|_| "vector U3 offset exceeds Metal u32 range".to_owned())?,
        );
    }
    Ok((offsets, entries))
}

fn pack_scalar_u1_entries(
    entries: &[super::ScalarU1EntryF32],
) -> Result<Vec<Mp0bVectorU1EntryMetal>, String> {
    entries
        .iter()
        .map(|entry| {
            Ok(Mp0bVectorU1EntryMetal {
                ell: u32::try_from(entry.ell)
                    .map_err(|_| "scalar U1 ell exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "scalar U1 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            })
        })
        .collect()
}

fn pack_scalar_u2_entries(
    entries: &[super::ScalarU2EntryF32],
) -> Result<Vec<Mp0bVectorU2EntryMetal>, String> {
    entries
        .iter()
        .map(|entry| {
            Ok(Mp0bVectorU2EntryMetal {
                left: u32::try_from(entry.left)
                    .map_err(|_| "scalar U2 left exceeds Metal u32 range".to_owned())?,
                right: u32::try_from(entry.right)
                    .map_err(|_| "scalar U2 right exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "scalar U2 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            })
        })
        .collect()
}

fn pack_scalar_u3_entries(
    entries: &[super::ScalarU3EntryF32],
) -> Result<Vec<Mp0bVectorU3EntryMetal>, String> {
    entries
        .iter()
        .map(|entry| {
            Ok(Mp0bVectorU3EntryMetal {
                left: u32::try_from(entry.left)
                    .map_err(|_| "scalar U3 left exceeds Metal u32 range".to_owned())?,
                right: u32::try_from(entry.right)
                    .map_err(|_| "scalar U3 right exceeds Metal u32 range".to_owned())?,
                inner: u32::try_from(entry.inner)
                    .map_err(|_| "scalar U3 inner exceeds Metal u32 range".to_owned())?,
                param: u32::try_from(entry.param)
                    .map_err(|_| "scalar U3 param exceeds Metal u32 range".to_owned())?,
                value: entry.value,
            })
        })
        .collect()
}

fn backend() -> Result<&'static Mp0bMetalBackend, String> {
    static BACKEND: OnceLock<Result<Mp0bMetalBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(Mp0bMetalBackend::new)
        .as_ref()
        .map_err(Clone::clone)
}

/// Pure predicate for whether a parsed MACE Metal environment value enables the
/// Metal backend.
///
/// Per the design doc the Metal path is enabled *only* when the variable is
/// exactly `"1"`. Unset (`None`), `"0"`, `"false"`, `"off"`, and any other value
/// must fall back to the CPU path. Kept as a standalone function so it can be
/// unit-tested without mutating the process environment.
fn mace_metal_value_enables(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Returns whether the MACE Metal backend was explicitly requested.
///
/// `LLMLIP_MACE_METAL=1` is the generic gate for cross-model MACE Metal work.
fn mace_metal_requested() -> bool {
    mace_metal_value_enables(std::env::var("LLMLIP_MACE_METAL").ok().as_deref())
}

/// Returns whether the existing MP0b-specific Metal backend was explicitly
/// requested via `LLMLIP_MACE_MP0B_METAL=1`.
fn mp0b_metal_requested() -> bool {
    mace_metal_value_enables(std::env::var("LLMLIP_MACE_MP0B_METAL").ok().as_deref())
}

pub(super) fn mace_metal_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(mace_metal_requested)
}

pub(super) fn mp0b_metal_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(mp0b_metal_requested)
}

fn mp0b_trace_vjp_kernel_timings_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LLMLIP_MACE_MP0B_METAL_TRACE_VJP_KERNELS").is_some())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_node_embedding_f32(
    type_ids: &[usize],
    in_features: usize,
    out_features: usize,
    weight: &[f32],
    output_mask: &[f32],
    path_weight: f32,
    out: &mut [f32],
) -> Result<(), String> {
    if weight.len() != in_features * out_features {
        return Err("node embedding weight shape mismatch".to_owned());
    }
    let mut type_ids_u32 = Vec::with_capacity(type_ids.len());
    for &type_id in type_ids {
        if type_id >= in_features {
            return Err("node embedding type id out of range".to_owned());
        }
        type_ids_u32.push(
            u32::try_from(type_id).map_err(|_| "type id exceeds Metal u32 range".to_owned())?,
        );
    }
    backend()?.node_embedding_f32(
        &type_ids_u32,
        weight,
        output_mask,
        out_features,
        path_weight,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_node_embedding_f32(
    type_ids: &[usize],
    in_features: usize,
    out_features: usize,
    weight: &[f32],
    output_mask: &[f32],
    path_weight: f32,
    out: &mut [f32],
) -> Result<(), String> {
    mace_node_embedding_f32(
        type_ids,
        in_features,
        out_features,
        weight,
        output_mask,
        path_weight,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_vjp_edge_feats_f32(
    num_edges: usize,
    num_bessel: usize,
    output_dim: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    weight3: &[f32],
    hidden0_derivative: &[f32],
    hidden1_derivative: &[f32],
    hidden2_derivative: &[f32],
    bar_output: &[f32],
    bar_edge_feats: &mut [f32],
) -> Result<(), String> {
    backend()?.conv_tp_weights_vjp_edge_feats_f32(
        num_edges,
        num_bessel,
        output_dim,
        weight0,
        weight1,
        weight2,
        weight3,
        hidden0_derivative,
        hidden1_derivative,
        hidden2_derivative,
        bar_output,
        bar_edge_feats,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_vjp_edge_feats_with_metal_cache_f32(
    num_edges: usize,
    num_bessel: usize,
    output_dim: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    weight3: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    bar_output: &[f32],
    bar_edge_feats: &mut [f32],
) -> Result<(), String> {
    backend()?.conv_tp_weights_vjp_edge_feats_with_metal_cache_f32(
        num_edges,
        num_bessel,
        output_dim,
        weight0,
        weight1,
        weight2,
        weight3,
        cache,
        bar_output,
        bar_edge_feats,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_vjp_edge_feats_from_hidden2_f32(
    num_edges: usize,
    num_bessel: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    hidden0_derivative: &[f32],
    hidden1_derivative: &[f32],
    hidden2_derivative: &[f32],
    bar_hidden2: &[f32],
    bar_edge_feats: &mut [f32],
) -> Result<(), String> {
    backend()?.conv_tp_weights_vjp_edge_feats_from_hidden2_f32(
        num_edges,
        num_bessel,
        weight0,
        weight1,
        weight2,
        hidden0_derivative,
        hidden1_derivative,
        hidden2_derivative,
        bar_hidden2,
        bar_edge_feats,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_vjp_edge_feats_from_hidden2_with_metal_cache_f32(
    num_edges: usize,
    num_bessel: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    bar_hidden2: &[f32],
    bar_edge_feats: &mut [f32],
) -> Result<(), String> {
    backend()?.conv_tp_weights_vjp_edge_feats_from_hidden2_with_metal_cache_f32(
        num_edges,
        num_bessel,
        weight0,
        weight1,
        weight2,
        cache,
        bar_hidden2,
        bar_edge_feats,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_vjp_edge_feats_from_hidden2_metal_buffer_with_metal_cache_f32(
    num_edges: usize,
    num_bessel: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    bar_hidden2: &Mp0bF32BufferMetal,
    bar_edge_feats: &mut [f32],
    assign_bar_edge_feats: bool,
) -> Result<(), String> {
    backend()?.conv_tp_weights_vjp_edge_feats_from_hidden2_metal_buffer_with_metal_cache_f32(
        num_edges,
        num_bessel,
        weight0,
        weight1,
        weight2,
        cache,
        bar_hidden2,
        bar_edge_feats,
        assign_bar_edge_feats,
    )
}

pub(super) fn mp0b_density_vjp_edge_feats_add_to_metal_buffer_f32(
    num_edges: usize,
    num_bessel: usize,
    weight: &[f32],
    bar_density: &[f32],
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    cache: &Mp0bConvTpWeightCacheMetal,
    bar_edge_feats: &Mp0bF32BufferMetal,
) -> Result<(), String> {
    backend()?.density_vjp_edge_feats_add_to_metal_buffer_f32(
        num_edges,
        num_bessel,
        weight,
        bar_density,
        topology,
        cache,
        bar_edge_feats,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_and_density_vjp_edge_feats_from_hidden2_metal_buffer_to_metal_with_metal_cache_f32(
    num_edges: usize,
    num_bessel: usize,
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    density_weight: &[f32],
    bar_density: &[f32],
    bar_density_metal: Option<&Mp0bF32BufferMetal>,
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    cache: &Mp0bConvTpWeightCacheMetal,
    bar_hidden2: &Mp0bF32BufferMetal,
    output_bar_edge_feats: Option<&Mp0bF32BufferMetal>,
    assign_bar_edge_feats: bool,
    wait_for_completion: bool,
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?
        .conv_tp_weights_and_density_vjp_edge_feats_from_hidden2_metal_buffer_to_metal_with_metal_cache_f32(
            num_edges,
            num_bessel,
            weight0,
            weight1,
            weight2,
            density_weight,
            bar_density,
            bar_density_metal,
            topology,
            cache,
            bar_hidden2,
            output_bar_edge_feats,
            assign_bar_edge_feats,
            wait_for_completion,
        )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_forward_f32(
    num_edges: usize,
    num_bessel: usize,
    output_dim: usize,
    silu_scale: f32,
    edge_feats: &[f32],
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    weight3: &[f32],
) -> Result<Mp0bConvTpWeightsForwardMetal, String> {
    backend()?.conv_tp_weights_forward_f32(
        num_edges, num_bessel, output_dim, silu_scale, edge_feats, weight0, weight1, weight2,
        weight3, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_conv_tp_weights_forward_skip_output_readback_f32(
    num_edges: usize,
    num_bessel: usize,
    output_dim: usize,
    silu_scale: f32,
    edge_feats: &[f32],
    weight0: &[f32],
    weight1: &[f32],
    weight2: &[f32],
    weight3: &[f32],
) -> Result<Mp0bConvTpWeightsForwardMetal, String> {
    backend()?.conv_tp_weights_forward_f32(
        num_edges, num_bessel, output_dim, silu_scale, edge_feats, weight0, weight1, weight2,
        weight3, false,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_hidden_linear_vjp_input_density_f32(
    density: &[f32],
    linear_out: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<(), String> {
    backend()?
        .layer1_hidden_linear_vjp_input_density_f32(
            density,
            linear_out,
            None,
            weight,
            output_mask,
            None,
            bar_out,
            bar_input,
            bar_density,
            true,
            true,
            false,
        )
        .map(|_| ())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_hidden_linear_vjp_input_density_skip_input_readback_f32(
    density: &[f32],
    linear_out: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?
        .layer1_hidden_linear_vjp_input_density_f32(
            density,
            linear_out,
            None,
            weight,
            output_mask,
            None,
            bar_out,
            bar_input,
            bar_density,
            false,
            true,
            false,
        )?
        .map(|vjp| vjp.bar_input)
        .ok_or_else(|| "layer1 hidden VJP did not return Metal bar input".to_owned())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_hidden_linear_vjp_input_density_feature_major_f32(
    density: &[f32],
    linear_out: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    bar_out_feature_major: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<(), String> {
    backend()?
        .layer1_hidden_linear_vjp_input_density_f32(
            density,
            linear_out,
            None,
            weight,
            output_mask,
            None,
            bar_out_feature_major,
            bar_input,
            bar_density,
            true,
            true,
            true,
        )
        .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_hidden_linear_vjp_input_density_feature_major_skip_input_density_readback_f32(
    density: &[f32],
    linear_out: &[f32],
    linear_out_metal_cache: Option<&Mp0bF32BufferMetal>,
    weight: &[f32],
    output_mask: &[f32],
    static_cache: Option<&Mp0bLayer1HiddenWeightsMetal>,
    bar_out_feature_major: &[f32],
    bar_input: &mut [f32],
) -> Result<Mp0bLayer1HiddenVjpMetal, String> {
    backend()?
        .layer1_hidden_linear_vjp_input_density_f32(
            density,
            linear_out,
            linear_out_metal_cache,
            weight,
            output_mask,
            static_cache,
            bar_out_feature_major,
            bar_input,
            &mut [],
            false,
            false,
            true,
        )?
        .ok_or_else(|| "layer1 hidden VJP did not return Metal buffers".to_owned())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer0_hidden_linear_vjp_input_density_f32(
    density: &[f32],
    linear_out: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<(), String> {
    backend()?.layer0_hidden_linear_vjp_input_density_f32(
        density,
        linear_out,
        weight,
        output_mask,
        bar_out,
        bar_input,
        bar_density,
    )
}

pub(super) fn mp0b_hidden_message_selector_vjp_input_from_feature_major_f32(
    type_ids: &[u32],
    weight: &[f32],
    output_mask: &[f32],
    bar_out_feature_major: &[f32],
    bar_input: &mut [f32],
) -> Result<(), String> {
    backend()?.hidden_message_selector_vjp_input_from_feature_major_f32(
        type_ids,
        weight,
        output_mask,
        bar_out_feature_major,
        bar_input,
    )
}

pub(super) fn mp0b_hidden_message_selector_to_feature_major_with_input_cache_f32(
    type_ids: &[u32],
    input: &[f32],
    input_metal_cache: Option<&Mp0bF32BufferMetal>,
    weight: &[f32],
    output_mask: &[f32],
    output_feature_major: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.hidden_message_selector_to_feature_major_f32(
        type_ids,
        input,
        input_metal_cache,
        weight,
        output_mask,
        output_feature_major,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer0_selector_hidden_vjp_input_density_f32(
    type_ids: &[u32],
    selector_weight: &[f32],
    selector_output_mask: &[f32],
    bar_interaction_out: &[f32],
    density: &[f32],
    linear_out: &[f32],
    hidden_weight: &[f32],
    hidden_output_mask: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<(), String> {
    backend()?.layer0_selector_hidden_vjp_input_density_f32(
        type_ids,
        selector_weight,
        selector_output_mask,
        bar_interaction_out,
        density,
        linear_out,
        hidden_weight,
        hidden_output_mask,
        bar_input,
        bar_density,
    )
}

pub(super) fn mp0b_scalar_vector_product_linear_forward_f32(
    input: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    backend()?.scalar_vector_product_linear_forward_f32(input, weight, output_mask, output)
}

pub(super) fn mp0b_scalar_vector_product_linear_forward_with_input_cache_f32(
    input: &[f32],
    input_metal_cache: Option<&Mp0bF32BufferMetal>,
    weight: &[f32],
    output_mask: &[f32],
    output: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.scalar_vector_product_linear_forward_with_input_cache_f32(
        input,
        input_metal_cache,
        weight,
        output_mask,
        output,
    )
}

pub(super) fn mp0b_scalar_vector_product_linear_vjp_input_f32(
    weight: &[f32],
    output_mask: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
) -> Result<(), String> {
    backend()?.scalar_vector_product_linear_vjp_input_f32(weight, output_mask, bar_out, bar_input)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn mp0b_layer1_conv_tp_forward_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    senders: &[usize],
    receivers: &[usize],
    scatter_message: &mut [f32],
) -> Result<(), String> {
    backend()?
        .layer1_conv_tp_forward_f32(
            num_nodes,
            linear_up,
            edge_attrs,
            tp_weights,
            senders,
            receivers,
            Some(scatter_message),
            None,
            None,
            true,
            false,
        )
        .map(|forward| forward.topology)
        .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_forward_with_topology_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    senders: &[usize],
    receivers: &[usize],
    scatter_message: &mut [f32],
    topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
) -> Result<Mp0bLayer1ConvTpTopologyMetal, String> {
    backend()?
        .layer1_conv_tp_forward_f32(
            num_nodes,
            linear_up,
            edge_attrs,
            tp_weights,
            senders,
            receivers,
            Some(scatter_message),
            None,
            topology_cache,
            true,
            false,
        )
        .map(|forward| forward.topology)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn oam_layer1_conv_tp_forward_with_topology_skip_scatter_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    senders: &[usize],
    receivers: &[usize],
    topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
) -> Result<Mp0bLayer1ConvTpForwardMetal, String> {
    backend()?.layer1_conv_tp_forward_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        senders,
        receivers,
        None,
        None,
        topology_cache,
        false,
        true,
    )
}

pub(super) fn mp0b_layer1_conv_tp_topology_f32(
    num_nodes: usize,
    senders: &[usize],
    receivers: &[usize],
) -> Result<Mp0bLayer1ConvTpTopologyMetal, String> {
    backend()?.layer1_conv_tp_topology_f32(num_nodes, senders, receivers)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_forward_with_metal_cache_and_topology_skip_scatter_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    senders: &[usize],
    receivers: &[usize],
    topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
) -> Result<Mp0bLayer1ConvTpForwardMetal, String> {
    backend()?.layer1_conv_tp_forward_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        senders,
        receivers,
        None,
        Some(cache),
        topology_cache,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn oam_layer1_conv_tp_forward_with_metal_cache_and_topology_skip_scatter_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    senders: &[usize],
    receivers: &[usize],
    topology_cache: Option<&Mp0bLayer1ConvTpTopologyMetal>,
) -> Result<Mp0bLayer1ConvTpForwardMetal, String> {
    backend()?.layer1_conv_tp_forward_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        senders,
        receivers,
        None,
        Some(cache),
        topology_cache,
        false,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_vjp_hidden2_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    layer3_weight: &[f32],
    bar_scatter_message: &[f32],
    senders: &[usize],
    receivers: &[usize],
    linear_up_vjp: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Mp0bLayer1ConvTpVjpHidden2Metal, String> {
    backend()?.layer1_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        layer3_weight,
        bar_scatter_message,
        None,
        senders,
        receivers,
        linear_up_vjp,
        None,
        None,
        None,
        true,
        true,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_vjp_hidden2_with_metal_cache_skip_hidden2_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    layer3_weight: &[f32],
    bar_scatter_message: &[f32],
    bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
    senders: &[usize],
    receivers: &[usize],
    linear_up_vjp: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Mp0bLayer1ConvTpVjpHidden2Metal, String> {
    backend()?.layer1_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        layer3_weight,
        bar_scatter_message,
        bar_scatter_message_metal,
        senders,
        receivers,
        linear_up_vjp,
        Some(cache),
        None,
        None,
        true,
        false,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_vjp_hidden2_with_metal_cache_and_topology_skip_hidden2_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    layer3_weight: &[f32],
    bar_scatter_message: &[f32],
    bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
    senders: &[usize],
    receivers: &[usize],
    linear_up_vjp: Option<(&[f32], &[f32], &[f32])>,
) -> Result<Mp0bLayer1ConvTpVjpHidden2Metal, String> {
    backend()?.layer1_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        layer3_weight,
        bar_scatter_message,
        bar_scatter_message_metal,
        senders,
        receivers,
        linear_up_vjp,
        Some(cache),
        Some(topology),
        None,
        true,
        false,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_vjp_fixed_cache_f32(
    num_nodes: usize,
    edge_attrs: &[f32],
    layer3_weight: &[f32],
    linear_up_weight: &[f32],
    linear_up_output_mask: &[f32],
) -> Result<Mp0bLayer1ConvTpVjpFixedCacheMetal, String> {
    backend()?.layer1_conv_tp_vjp_fixed_cache_f32(
        num_nodes,
        edge_attrs,
        layer3_weight,
        linear_up_weight,
        linear_up_output_mask,
    )
}

pub(super) fn mp0b_layer0_conv_tp_vjp_fixed_cache_f32(
    num_nodes: usize,
    edge_attrs: &[f32],
    layer3_weight: &[f32],
) -> Result<Mp0bLayer0ConvTpVjpFixedCacheMetal, String> {
    backend()?.layer0_conv_tp_vjp_fixed_cache_f32(num_nodes, edge_attrs, layer3_weight)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer0_conv_tp_vjp_hidden2_with_fixed_cache_and_topology_f32(
    num_nodes: usize,
    linear_up: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    fixed_cache: &Mp0bLayer0ConvTpVjpFixedCacheMetal,
    bar_scatter_message: &[f32],
    bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
) -> Result<Mp0bLayer0ConvTpVjpHidden2Metal, String> {
    backend()?.layer0_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        tp_weights,
        cache,
        topology,
        fixed_cache,
        bar_scatter_message,
        bar_scatter_message_metal,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer0_conv_tp_vjp_hidden2_with_fixed_cache_and_topology_accumulate_edge_attrs_metal_skip_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    fixed_cache: &Mp0bLayer0ConvTpVjpFixedCacheMetal,
    bar_scatter_message: &[f32],
    bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
    accumulate_bar_edge_attrs_metal: &Mp0bF32BufferMetal,
) -> Result<Mp0bLayer0ConvTpVjpHidden2Metal, String> {
    backend()?.layer0_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        tp_weights,
        cache,
        topology,
        fixed_cache,
        bar_scatter_message,
        bar_scatter_message_metal,
        Some(accumulate_bar_edge_attrs_metal),
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_layer1_conv_tp_vjp_hidden2_with_fixed_cache_and_topology_skip_hidden2_readback_f32(
    num_nodes: usize,
    linear_up: &[f32],
    edge_attrs: &[f32],
    tp_weights: &[f32],
    cache: &Mp0bConvTpWeightCacheMetal,
    topology: &Mp0bLayer1ConvTpTopologyMetal,
    fixed_cache: &Mp0bLayer1ConvTpVjpFixedCacheMetal,
    layer3_weight: &[f32],
    bar_scatter_message: &[f32],
    bar_scatter_message_metal: Option<&Mp0bF32BufferMetal>,
    senders: &[usize],
    receivers: &[usize],
    linear_up_vjp: Option<(&[f32], &[f32], &[f32])>,
    read_bar_edge_attrs: bool,
    read_bar_layer0_product_out: bool,
) -> Result<Mp0bLayer1ConvTpVjpHidden2Metal, String> {
    backend()?.layer1_conv_tp_vjp_hidden2_f32(
        num_nodes,
        linear_up,
        edge_attrs,
        tp_weights,
        layer3_weight,
        bar_scatter_message,
        bar_scatter_message_metal,
        senders,
        receivers,
        linear_up_vjp,
        Some(cache),
        Some(topology),
        Some(fixed_cache),
        read_bar_edge_attrs,
        false,
        read_bar_layer0_product_out,
    )
}

pub(super) fn mp0b_copy_f32_buffer_to_slice(
    source: &Mp0bF32BufferMetal,
    output: &mut [f32],
) -> Result<(), String> {
    backend()?.copy_f32_buffer_to_slice(source, output)
}

pub(super) fn mp0b_layer1_hidden_forward_f32(
    input: &[f32],
    density: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    output: &mut [f32],
) -> Result<Mp0bLayer1HiddenForwardMetal, String> {
    backend()?.layer1_hidden_forward_f32(input, None, density, weight, output_mask, None, output)
}

pub(super) fn mp0b_layer1_hidden_forward_with_input_metal_cache_f32(
    input: &[f32],
    input_metal_cache: &Mp0bF32BufferMetal,
    density: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    output: &mut [f32],
) -> Result<Mp0bLayer1HiddenForwardMetal, String> {
    backend()?.layer1_hidden_forward_f32(
        input,
        Some(input_metal_cache),
        density,
        weight,
        output_mask,
        None,
        output,
    )
}

pub(super) fn mp0b_layer1_hidden_forward_with_input_and_static_cache_f32(
    input: &[f32],
    input_metal_cache: &Mp0bF32BufferMetal,
    density: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    static_cache: &Mp0bLayer1HiddenWeightsMetal,
    output: &mut [f32],
) -> Result<Mp0bLayer1HiddenForwardMetal, String> {
    backend()?.layer1_hidden_forward_f32(
        input,
        Some(input_metal_cache),
        density,
        weight,
        output_mask,
        Some(static_cache),
        output,
    )
}

pub(super) fn mp0b_layer1_hidden_weights_cache_f32(
    weight: &[f32],
    output_mask: &[f32],
) -> Result<Mp0bLayer1HiddenWeightsMetal, String> {
    backend()?.layer1_hidden_weights_cache_f32(weight, output_mask)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_vector_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    grad_out: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_vector_contraction_vjp_f32(
        type_ids,
        x,
        grad_out,
        weights_u1,
        weights_u2,
        weights_u3,
        u1_entries_by_equiv,
        u2_entries_by_equiv,
        u3_entries_by_equiv,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_vector_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    grad_out: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    num_elements: usize,
    num_features: usize,
    num_equivariance: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.vector_contraction_vjp_f32(
        type_ids,
        x,
        grad_out,
        weights_u1,
        weights_u2,
        weights_u3,
        u1_entries_by_equiv,
        u2_entries_by_equiv,
        u3_entries_by_equiv,
        num_elements,
        num_features,
        num_equivariance,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_vector_contraction_forward_f32(
    type_ids: &[u32],
    x: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    num_elements: usize,
    num_features: usize,
    num_equivariance: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    output_stride: usize,
    output_offset: usize,
    out: &mut [f32],
) -> Result<(), String> {
    backend()?.vector_contraction_forward_f32(
        type_ids,
        x,
        weights_u1,
        weights_u2,
        weights_u3,
        u1_entries_by_equiv,
        u2_entries_by_equiv,
        u3_entries_by_equiv,
        num_elements,
        num_features,
        num_equivariance,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        output_stride,
        output_offset,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_scalar_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    grad_out: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_scalar_contraction_vjp_f32(
        type_ids, x, grad_out, weights_u1, weights_u2, weights_u3, u1_entries, u2_entries,
        u3_entries, grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) fn mp0b_product0_scalar_contraction_vjp_cache_f32(
    type_ids: &[u32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
) -> Result<Mp0bProduct0ScalarVjpCacheMetal, String> {
    backend()?.product0_scalar_contraction_vjp_cache_f32(
        type_ids, weights_u1, weights_u2, weights_u3, u1_entries, u2_entries, u3_entries,
    )
}

#[allow(dead_code)]
pub(super) fn mp0b_product0_scalar_contraction_vjp_with_cache_f32(
    cache: &Mp0bProduct0ScalarVjpCacheMetal,
    x: &[f32],
    grad_out: &[f32],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_scalar_contraction_vjp_with_cache_f32(cache, x, grad_out, grad_x)
}

pub(super) fn mp0b_product0_scalar_contraction_vjp_with_input_cache_f32(
    cache: &Mp0bProduct0ScalarVjpCacheMetal,
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    grad_out: &[f32],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_scalar_contraction_vjp_with_input_cache_f32(
        cache,
        x,
        x_metal_cache,
        grad_out,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_scalar_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    grad_out: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    num_elements: usize,
    num_features: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.scalar_contraction_vjp_f32(
        type_ids,
        x,
        grad_out,
        weights_u1,
        weights_u2,
        weights_u3,
        num_elements,
        num_features,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        u1_entries,
        u2_entries,
        u3_entries,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_scalar_contraction_vjp_cache_f32(
    type_ids: &[u32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    num_elements: usize,
    num_features: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
) -> Result<MaceScalarVjpCacheMetal, String> {
    backend()?.scalar_contraction_vjp_cache_f32(
        type_ids,
        weights_u1,
        weights_u2,
        weights_u3,
        num_elements,
        num_features,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        u1_entries,
        u2_entries,
        u3_entries,
    )
}

pub(super) fn mace_scalar_contraction_vjp_with_input_cache_f32(
    cache: &MaceScalarVjpCacheMetal,
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    grad_out: &[f32],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.scalar_contraction_vjp_with_input_cache_f32(
        cache,
        x,
        x_metal_cache,
        grad_out,
        grad_x,
    )
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn mace_scalar_contraction_forward_f32(
    type_ids: &[u32],
    x: &[f32],
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    num_elements: usize,
    num_features: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
    output_stride: usize,
    output_offset: usize,
    out: &mut [f32],
) -> Result<(), String> {
    backend()?.scalar_contraction_forward_f32(
        type_ids,
        x,
        weights_u1,
        weights_u2,
        weights_u3,
        num_elements,
        num_features,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        u1_entries,
        u2_entries,
        u3_entries,
        output_stride,
        output_offset,
        out,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mace_scalar_contraction_forward_with_input_cache_f32(
    type_ids: &[u32],
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    weights_u1: &[f32],
    weights_u2: &[f32],
    weights_u3: &[f32],
    num_elements: usize,
    num_features: usize,
    num_ell: usize,
    u1_params: usize,
    u2_params: usize,
    u3_params: usize,
    u1_entries: &[super::ScalarU1EntryF32],
    u2_entries: &[super::ScalarU2EntryF32],
    u3_entries: &[super::ScalarU3EntryF32],
    output_stride: usize,
    output_offset: usize,
    out: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.scalar_contraction_forward_with_input_cache_f32(
        type_ids,
        x,
        x_metal_cache,
        weights_u1,
        weights_u2,
        weights_u3,
        num_elements,
        num_features,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        u1_entries,
        u2_entries,
        u3_entries,
        output_stride,
        output_offset,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_scalar_vector_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    scalar_grad_out: &[f32],
    vector_grad_out: &[f32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_scalar_vector_contraction_vjp_f32(
        type_ids,
        x,
        scalar_grad_out,
        vector_grad_out,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_linear_scalar_vector_contraction_vjp_f32(
    type_ids: &[u32],
    x: &[f32],
    bar_product_out: &[f32],
    product_linear_weight: &[f32],
    product_linear_output_mask: &[f32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    grad_x: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_linear_scalar_vector_contraction_vjp_f32(
        type_ids,
        x,
        bar_product_out,
        product_linear_weight,
        product_linear_output_mask,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        grad_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_linear_scalar_vector_contraction_layer0_selector_hidden_vjp_input_density_f32(
    type_ids: &[u32],
    x: &[f32],
    bar_product_out: &[f32],
    product_linear_weight: &[f32],
    product_linear_output_mask: &[f32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    selector_weight: &[f32],
    selector_output_mask: &[f32],
    density: &[f32],
    linear_out: &[f32],
    hidden_weight: &[f32],
    hidden_output_mask: &[f32],
    bar_input: &mut [f32],
    bar_density: &mut [f32],
) -> Result<(), String> {
    backend()?
        .product0_linear_scalar_vector_contraction_layer0_selector_hidden_vjp_input_density_f32(
            type_ids,
            x,
            bar_product_out,
            product_linear_weight,
            product_linear_output_mask,
            scalar_weights_u1,
            scalar_weights_u2,
            scalar_weights_u3,
            scalar_u1_entries,
            scalar_u2_entries,
            scalar_u3_entries,
            vector_weights_u1,
            vector_weights_u2,
            vector_weights_u3,
            vector_u1_entries_by_equiv,
            vector_u2_entries_by_equiv,
            vector_u3_entries_by_equiv,
            selector_weight,
            selector_output_mask,
            density,
            linear_out,
            hidden_weight,
            hidden_output_mask,
            bar_input,
            bar_density,
        )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_layer0_vjp_cache_f32(
    type_ids: &[u32],
    product_linear_weight: &[f32],
    product_linear_output_mask: &[f32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    selector_weight: &[f32],
    selector_output_mask: &[f32],
    hidden_weight: &[f32],
    hidden_output_mask: &[f32],
) -> Result<Mp0bProduct0Layer0VjpCacheMetal, String> {
    backend()?.product0_layer0_vjp_cache_f32(
        type_ids,
        product_linear_weight,
        product_linear_output_mask,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        selector_weight,
        selector_output_mask,
        hidden_weight,
        hidden_output_mask,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mp0b_product0_layer0_vjp_with_cache_skip_input_readback_f32(
    cache: &Mp0bProduct0Layer0VjpCacheMetal,
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    bar_product_out: &[f32],
    bar_product_out_metal: Option<&Mp0bF32BufferMetal>,
    density: &[f32],
    density_metal_cache: Option<&Mp0bF32BufferMetal>,
    linear_out: &[f32],
    linear_out_metal_cache: Option<&Mp0bF32BufferMetal>,
    bar_density: &mut [f32],
    read_bar_density: bool,
) -> Result<Mp0bProduct0Layer0VjpMetal, String> {
    backend()?.product0_layer0_vjp_with_cache_f32(
        cache,
        x,
        x_metal_cache,
        bar_product_out,
        bar_product_out_metal,
        density,
        density_metal_cache,
        linear_out,
        linear_out_metal_cache,
        &mut [],
        bar_density,
        false,
        read_bar_density,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_product0_contraction_forward_f32(
    type_ids: &[u32],
    x: &[f32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    product_symmetric: &mut [f32],
) -> Result<(), String> {
    backend()?.product0_contraction_forward_f32(
        type_ids,
        x,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        product_symmetric,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_product0_contraction_forward_with_input_cache_skip_readback_f32(
    type_ids: &[u32],
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    product_symmetric: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.product0_contraction_forward_with_input_cache_f32(
        type_ids,
        x,
        x_metal_cache,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        None,
        product_symmetric,
        false,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_product0_contraction_forward_with_input_cache_f32(
    type_ids: &[u32],
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
    product_symmetric: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.product0_contraction_forward_with_input_cache_f32(
        type_ids,
        x,
        x_metal_cache,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
        None,
        product_symmetric,
        true,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn mp0b_product0_forward_cache_f32(
    type_ids: &[u32],
    scalar_weights_u1: &[f32],
    scalar_weights_u2: &[f32],
    scalar_weights_u3: &[f32],
    scalar_u1_entries: &[super::ScalarU1EntryF32],
    scalar_u2_entries: &[super::ScalarU2EntryF32],
    scalar_u3_entries: &[super::ScalarU3EntryF32],
    vector_weights_u1: &[f32],
    vector_weights_u2: &[f32],
    vector_weights_u3: &[f32],
    vector_u1_entries_by_equiv: &[Vec<super::VectorU1EntryF32>],
    vector_u2_entries_by_equiv: &[Vec<super::VectorU2EntryF32>],
    vector_u3_entries_by_equiv: &[Vec<super::VectorU3EntryF32>],
) -> Result<Mp0bProduct0ForwardCacheMetal, String> {
    backend()?.product0_forward_cache_f32(
        type_ids,
        scalar_weights_u1,
        scalar_weights_u2,
        scalar_weights_u3,
        scalar_u1_entries,
        scalar_u2_entries,
        scalar_u3_entries,
        vector_weights_u1,
        vector_weights_u2,
        vector_weights_u3,
        vector_u1_entries_by_equiv,
        vector_u2_entries_by_equiv,
        vector_u3_entries_by_equiv,
    )
}

pub(super) fn mp0b_product0_contraction_forward_with_static_cache_skip_readback_f32(
    x: &[f32],
    x_metal_cache: Option<&Mp0bF32BufferMetal>,
    static_cache: &Mp0bProduct0ForwardCacheMetal,
    product_symmetric: &mut [f32],
) -> Result<Mp0bF32BufferMetal, String> {
    backend()?.product0_contraction_forward_with_input_cache_f32(
        &[],
        x,
        x_metal_cache,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        Some(static_cache),
        product_symmetric,
        false,
    )
}

pub(super) fn mp0b_layer0_hidden_forward_f32(
    input: &[f32],
    density: &[f32],
    weight: &[f32],
    output_mask: &[f32],
    output: &mut [f32],
) -> Result<Mp0bLayer0HiddenForwardMetal, String> {
    backend()?.layer0_hidden_forward_f32(input, density, weight, output_mask, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mace_metal_gate_enables_only_on_exact_one() {
        // Unset and falsy / off-like values must keep the CPU path.
        assert!(!mace_metal_value_enables(None));
        assert!(!mace_metal_value_enables(Some("0")));
        assert!(!mace_metal_value_enables(Some("false")));
        assert!(!mace_metal_value_enables(Some("off")));
        assert!(!mace_metal_value_enables(Some("")));
        // Only the exact "1" enables the Metal backend, matching the design doc.
        assert!(mace_metal_value_enables(Some("1")));
    }

    #[test]
    fn mp0b_node_embedding_kernel_matches_cpu_formula() -> Result<(), String> {
        let type_ids = [0usize, 2, 1];
        let in_features = 3usize;
        let out_features = 4usize;
        let path_weight = 0.25_f32;
        let weight = (0..in_features * out_features)
            .map(|index| index as f32 * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let output_mask = [1.0_f32, 0.5, -1.0, 2.0];
        let mut actual = vec![0.0_f32; type_ids.len() * out_features];

        match mp0b_node_embedding_f32(
            &type_ids,
            in_features,
            out_features,
            &weight,
            &output_mask,
            path_weight,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let mut expected = vec![0.0_f32; actual.len()];
        for (node, &type_id) in type_ids.iter().enumerate() {
            for feature in 0..out_features {
                expected[node * out_features + feature] =
                    weight[type_id * out_features + feature] * path_weight * output_mask[feature];
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "node embedding mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_conv_tp_weights_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let num_edges = 2usize;
        let num_bessel = 3usize;
        let output_dim = 5usize;
        let weight0 = (0..num_bessel * HIDDEN_DIM)
            .map(|index| (index as f32 % 17.0) * 0.01 - 0.07)
            .collect::<Vec<_>>();
        let weight1 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 23.0) * 0.005 - 0.04)
            .collect::<Vec<_>>();
        let weight2 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 29.0) * 0.004 - 0.05)
            .collect::<Vec<_>>();
        let weight3 = (0..HIDDEN_DIM * output_dim)
            .map(|index| (index as f32 % 31.0) * 0.006 - 0.06)
            .collect::<Vec<_>>();
        let hidden0_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.5 + (index % 7) as f32 * 0.03)
            .collect::<Vec<_>>();
        let hidden1_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.7 + (index % 11) as f32 * 0.02)
            .collect::<Vec<_>>();
        let hidden2_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.9 + (index % 13) as f32 * 0.01)
            .collect::<Vec<_>>();
        let bar_output = (0..num_edges * output_dim)
            .map(|index| (index as f32 % 19.0) * 0.02 - 0.1)
            .collect::<Vec<_>>();
        let mut actual = (0..num_edges * num_bessel)
            .map(|index| (index as f32 % 5.0) * 0.03)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        match mp0b_conv_tp_weights_vjp_edge_feats_f32(
            num_edges,
            num_bessel,
            output_dim,
            &weight0,
            &weight1,
            &weight2,
            &weight3,
            &hidden0_derivative,
            &hidden1_derivative,
            &hidden2_derivative,
            &bar_output,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let inv_hidden = 1.0 / (HIDDEN_DIM as f32).sqrt();
        let inv_bessel = 1.0 / (num_bessel as f32).sqrt();
        let mut bar_hidden2 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..output_dim {
                    acc += bar_output[edge * output_dim + out] * weight3[hidden * output_dim + out];
                }
                bar_hidden2[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * hidden2_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        let mut bar_hidden1 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden2[edge * HIDDEN_DIM + out] * weight2[hidden * HIDDEN_DIM + out];
                }
                bar_hidden1[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * hidden1_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        let mut bar_hidden0 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden1[edge * HIDDEN_DIM + out] * weight1[hidden * HIDDEN_DIM + out];
                }
                bar_hidden0[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * hidden0_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        for edge in 0..num_edges {
            for basis in 0..num_bessel {
                let mut acc = 0.0_f32;
                for hidden in 0..HIDDEN_DIM {
                    acc += bar_hidden0[edge * HIDDEN_DIM + hidden]
                        * weight0[basis * HIDDEN_DIM + hidden];
                }
                expected[edge * num_bessel + basis] += acc * inv_bessel;
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "conv-TP weight VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_conv_tp_weights_vjp_from_hidden2_kernel_matches_cpu_formula() -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let num_edges = 2usize;
        let num_bessel = 3usize;
        let weight0 = (0..num_bessel * HIDDEN_DIM)
            .map(|index| (index as f32 % 17.0) * 0.01 - 0.07)
            .collect::<Vec<_>>();
        let weight1 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 23.0) * 0.005 - 0.04)
            .collect::<Vec<_>>();
        let weight2 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 29.0) * 0.004 - 0.05)
            .collect::<Vec<_>>();
        let hidden0_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.5 + (index % 7) as f32 * 0.03)
            .collect::<Vec<_>>();
        let hidden1_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.7 + (index % 11) as f32 * 0.02)
            .collect::<Vec<_>>();
        let hidden2_derivative = (0..num_edges * HIDDEN_DIM)
            .map(|index| 0.9 + (index % 13) as f32 * 0.01)
            .collect::<Vec<_>>();
        let bar_hidden2_input = (0..num_edges * HIDDEN_DIM)
            .map(|index| (index as f32 % 19.0) * 0.02 - 0.1)
            .collect::<Vec<_>>();
        let mut actual = (0..num_edges * num_bessel)
            .map(|index| (index as f32 % 5.0) * 0.03)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        match mp0b_conv_tp_weights_vjp_edge_feats_from_hidden2_f32(
            num_edges,
            num_bessel,
            &weight0,
            &weight1,
            &weight2,
            &hidden0_derivative,
            &hidden1_derivative,
            &hidden2_derivative,
            &bar_hidden2_input,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let inv_hidden = 1.0 / (HIDDEN_DIM as f32).sqrt();
        let inv_bessel = 1.0 / (num_bessel as f32).sqrt();
        let mut bar_hidden2 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for index in 0..bar_hidden2.len() {
            bar_hidden2[index] = bar_hidden2_input[index] * hidden2_derivative[index];
        }
        let mut bar_hidden1 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden2[edge * HIDDEN_DIM + out] * weight2[hidden * HIDDEN_DIM + out];
                }
                bar_hidden1[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * hidden1_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        let mut bar_hidden0 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden1[edge * HIDDEN_DIM + out] * weight1[hidden * HIDDEN_DIM + out];
                }
                bar_hidden0[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * hidden0_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        for edge in 0..num_edges {
            for basis in 0..num_bessel {
                let mut acc = 0.0_f32;
                for hidden in 0..HIDDEN_DIM {
                    acc += bar_hidden0[edge * HIDDEN_DIM + hidden]
                        * weight0[basis * HIDDEN_DIM + hidden];
                }
                expected[edge * num_bessel + basis] += acc * inv_bessel;
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "conv-TP weight hidden2 VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_conv_tp_weights_and_density_vjp_from_hidden2_to_metal_matches_cpu_formula(
    ) -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let num_nodes = 3usize;
        let senders = vec![0usize, 1, 2];
        let receivers = vec![1usize, 2, 0];
        let num_edges = senders.len();
        let num_bessel = 3usize;
        let output_dim = 5usize;
        let silu_scale = 1.679_176_8_f32;
        let edge_feats = (0..num_edges * num_bessel)
            .map(|index| (index as f32 % 7.0) * 0.04 - 0.1)
            .collect::<Vec<_>>();
        let weight0 = (0..num_bessel * HIDDEN_DIM)
            .map(|index| (index as f32 % 17.0) * 0.01 - 0.07)
            .collect::<Vec<_>>();
        let weight1 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 23.0) * 0.005 - 0.04)
            .collect::<Vec<_>>();
        let weight2 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 29.0) * 0.004 - 0.05)
            .collect::<Vec<_>>();
        let weight3 = (0..HIDDEN_DIM * output_dim)
            .map(|index| (index as f32 % 31.0) * 0.006 - 0.06)
            .collect::<Vec<_>>();
        let forward = match mp0b_conv_tp_weights_forward_f32(
            num_edges,
            num_bessel,
            output_dim,
            silu_scale,
            &edge_feats,
            &weight0,
            &weight1,
            &weight2,
            &weight3,
        ) {
            Ok(forward) => forward,
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        };
        let topology = mp0b_layer1_conv_tp_topology_f32(num_nodes, &senders, &receivers)?;
        let bar_hidden2_input = (0..num_edges * HIDDEN_DIM)
            .map(|index| (index as f32 % 19.0) * 0.02 - 0.1)
            .collect::<Vec<_>>();
        let density_weight = (0..num_bessel)
            .map(|index| (index as f32 + 1.0) * 0.07 - 0.09)
            .collect::<Vec<_>>();
        let bar_density = (0..num_nodes)
            .map(|index| (index as f32 % 5.0) * 0.11 - 0.2)
            .collect::<Vec<_>>();
        let bar_hidden2_buffer =
            backend()?.buffer_with_slice(&bar_hidden2_input, MTLResourceOptions::StorageModeShared);
        let bar_hidden2_metal = Mp0bF32BufferMetal {
            len: bar_hidden2_input.len(),
            buffer: bar_hidden2_buffer,
        };
        let actual_metal =
            mp0b_conv_tp_weights_and_density_vjp_edge_feats_from_hidden2_metal_buffer_to_metal_with_metal_cache_f32(
                num_edges,
                num_bessel,
                &weight0,
                &weight1,
                &weight2,
                &density_weight,
                &bar_density,
                None,
                &topology,
                &forward.metal_cache,
                &bar_hidden2_metal,
                None,
                true,
                true,
            )?;
        let mut actual = vec![0.0_f32; num_edges * num_bessel];
        mp0b_copy_f32_buffer_to_slice(&actual_metal, &mut actual)?;

        let inv_hidden = 1.0 / (HIDDEN_DIM as f32).sqrt();
        let inv_bessel = 1.0 / (num_bessel as f32).sqrt();
        let mut expected = vec![0.0_f32; num_edges * num_bessel];
        let mut bar_hidden2 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for index in 0..bar_hidden2.len() {
            bar_hidden2[index] = bar_hidden2_input[index] * forward.hidden2_derivative[index];
        }
        let mut bar_hidden1 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden2[edge * HIDDEN_DIM + out] * weight2[hidden * HIDDEN_DIM + out];
                }
                bar_hidden1[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * forward.hidden1_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        let mut bar_hidden0 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut acc = 0.0_f32;
                for out in 0..HIDDEN_DIM {
                    acc +=
                        bar_hidden1[edge * HIDDEN_DIM + out] * weight1[hidden * HIDDEN_DIM + out];
                }
                bar_hidden0[edge * HIDDEN_DIM + hidden] =
                    acc * inv_hidden * forward.hidden0_derivative[edge * HIDDEN_DIM + hidden];
            }
        }
        for edge in 0..num_edges {
            for basis in 0..num_bessel {
                let mut acc = 0.0_f32;
                for hidden in 0..HIDDEN_DIM {
                    acc += bar_hidden0[edge * HIDDEN_DIM + hidden]
                        * weight0[basis * HIDDEN_DIM + hidden];
                }
                expected[edge * num_bessel + basis] += acc * inv_bessel;
            }
            let raw = (0..num_bessel)
                .map(|basis| edge_feats[edge * num_bessel + basis] * density_weight[basis])
                .sum::<f32>();
            let edge_density = (raw * raw).tanh();
            let bar_linear =
                bar_density[receivers[edge]] * (1.0 - edge_density * edge_density) * 2.0 * raw;
            for basis in 0..num_bessel {
                expected[edge * num_bessel + basis] += bar_linear * density_weight[basis];
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 2.5e-5 {
                return Err(format!(
                    "combined conv-TP/density hidden2 VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_conv_tp_weights_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const HIDDEN_DIM: usize = 64;
        let num_edges = 2usize;
        let num_bessel = 3usize;
        let output_dim = 5usize;
        let silu_scale = 1.679_176_8_f32;
        let edge_feats = (0..num_edges * num_bessel)
            .map(|index| (index as f32 % 7.0) * 0.04 - 0.1)
            .collect::<Vec<_>>();
        let weight0 = (0..num_bessel * HIDDEN_DIM)
            .map(|index| (index as f32 % 17.0) * 0.01 - 0.07)
            .collect::<Vec<_>>();
        let weight1 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 23.0) * 0.005 - 0.04)
            .collect::<Vec<_>>();
        let weight2 = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|index| (index as f32 % 29.0) * 0.004 - 0.05)
            .collect::<Vec<_>>();
        let weight3 = (0..HIDDEN_DIM * output_dim)
            .map(|index| (index as f32 % 31.0) * 0.006 - 0.06)
            .collect::<Vec<_>>();

        let actual = match mp0b_conv_tp_weights_forward_f32(
            num_edges,
            num_bessel,
            output_dim,
            silu_scale,
            &edge_feats,
            &weight0,
            &weight1,
            &weight2,
            &weight3,
        ) {
            Ok(output) => output,
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        };

        let silu = |raw: f32| {
            let sigmoid = 1.0 / (1.0 + (-raw).exp());
            (
                silu_scale * raw * sigmoid,
                silu_scale * sigmoid * (1.0 + raw * (1.0 - sigmoid)),
            )
        };
        let inv_bessel = 1.0 / (num_bessel as f32).sqrt();
        let inv_hidden = 1.0 / (HIDDEN_DIM as f32).sqrt();
        let mut hidden0 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        let mut hidden0_derivative = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut raw = 0.0_f32;
                for basis in 0..num_bessel {
                    raw += edge_feats[edge * num_bessel + basis]
                        * weight0[basis * HIDDEN_DIM + hidden];
                }
                let (value, derivative) = silu(raw * inv_bessel);
                hidden0[edge * HIDDEN_DIM + hidden] = value;
                hidden0_derivative[edge * HIDDEN_DIM + hidden] = derivative;
            }
        }
        let mut hidden1 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        let mut hidden1_derivative = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut raw = 0.0_f32;
                for input in 0..HIDDEN_DIM {
                    raw +=
                        hidden0[edge * HIDDEN_DIM + input] * weight1[input * HIDDEN_DIM + hidden];
                }
                let (value, derivative) = silu(raw * inv_hidden);
                hidden1[edge * HIDDEN_DIM + hidden] = value;
                hidden1_derivative[edge * HIDDEN_DIM + hidden] = derivative;
            }
        }
        let mut hidden2 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        let mut hidden2_derivative = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut raw = 0.0_f32;
                for input in 0..HIDDEN_DIM {
                    raw +=
                        hidden1[edge * HIDDEN_DIM + input] * weight2[input * HIDDEN_DIM + hidden];
                }
                let (value, derivative) = silu(raw * inv_hidden);
                hidden2[edge * HIDDEN_DIM + hidden] = value;
                hidden2_derivative[edge * HIDDEN_DIM + hidden] = derivative;
            }
        }
        let mut expected_output = vec![0.0_f32; num_edges * output_dim];
        for edge in 0..num_edges {
            for out in 0..output_dim {
                let mut value = 0.0_f32;
                for hidden in 0..HIDDEN_DIM {
                    value +=
                        hidden2[edge * HIDDEN_DIM + hidden] * weight3[hidden * output_dim + out];
                }
                expected_output[edge * output_dim + out] = value * inv_hidden;
            }
        }

        for (label, actual, expected) in [
            (
                "output",
                actual.output.as_slice(),
                expected_output.as_slice(),
            ),
            (
                "hidden0_derivative",
                actual.hidden0_derivative.as_slice(),
                hidden0_derivative.as_slice(),
            ),
            (
                "hidden1_derivative",
                actual.hidden1_derivative.as_slice(),
                hidden1_derivative.as_slice(),
            ),
            (
                "hidden2_derivative",
                actual.hidden2_derivative.as_slice(),
                hidden2_derivative.as_slice(),
            ),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                if (actual - expected).abs() > 2.5e-5 {
                    return Err(format!(
                        "{label} mismatch at {index}: actual={actual} expected={expected}"
                    ));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer1_conv_tp_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const INPUT_ROW_LEN: usize = 512;
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 1280;
        const OUTPUT_ROW_LEN: usize = 5120;
        let num_nodes = 3usize;
        let senders = vec![0usize, 1, 2, 0, 1];
        let receivers = vec![1usize, 2, 0, 1, 0];
        let num_edges = senders.len();
        let linear_up = (0..num_nodes * INPUT_ROW_LEN)
            .map(|index| (index as f32 % 37.0) * 0.002 - 0.03)
            .collect::<Vec<_>>();
        let edge_attrs = (0..num_edges * EDGE_ATTR_ROW_LEN)
            .map(|index| (index as f32 % 19.0) * 0.003 - 0.02)
            .collect::<Vec<_>>();
        let tp_weights = (0..num_edges * TP_WEIGHT_ROW_LEN)
            .map(|index| (index as f32 % 29.0) * 0.001 - 0.01)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0_f32; num_nodes * OUTPUT_ROW_LEN];

        match mp0b_layer1_conv_tp_forward_f32(
            num_nodes,
            &linear_up,
            &edge_attrs,
            &tp_weights,
            &senders,
            &receivers,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let mut expected = vec![0.0_f32; actual.len()];
        for edge in 0..num_edges {
            let sender = senders[edge];
            let receiver = receivers[edge];
            let x = &linear_up[sender * INPUT_ROW_LEN..(sender + 1) * INPUT_ROW_LEN];
            let y = &edge_attrs[edge * EDGE_ATTR_ROW_LEN..(edge + 1) * EDGE_ATTR_ROW_LEN];
            let w = &tp_weights[edge * TP_WEIGHT_ROW_LEN..(edge + 1) * TP_WEIGHT_ROW_LEN];
            let scatter = &mut expected[receiver * OUTPUT_ROW_LEN..(receiver + 1) * OUTPUT_ROW_LEN];
            for (template_index, template) in
                super::super::MP0B_LAYER1_TP_TEMPLATES.iter().enumerate()
            {
                let cg_entries = super::super::MP0B_LAYER1_TP_CG_TEMPLATES_F32[template_index];
                for feature in 0..128 {
                    let weight = w[template.weight_base + feature];
                    let x_base = template.x_base + feature * template.x_dim;
                    let out_base = template.out_base + feature * template.out_dim;
                    for cg in cg_entries {
                        scatter[out_base + cg.k_m] +=
                            weight * cg.scale * x[x_base + cg.i_m] * y[template.y_base + cg.j_m];
                    }
                }
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer1 conv TP forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer1_conv_tp_vjp_hidden2_kernel_matches_cpu_formula() -> Result<(), String> {
        const INPUT_ROW_LEN: usize = 512;
        const EDGE_ATTR_ROW_LEN: usize = 16;
        const TP_WEIGHT_ROW_LEN: usize = 1280;
        const OUTPUT_ROW_LEN: usize = 5120;
        const HIDDEN_DIM: usize = 64;
        let num_nodes = 3usize;
        let senders = vec![0usize, 1, 2, 0, 1];
        let receivers = vec![1usize, 2, 0, 1, 0];
        let num_edges = senders.len();
        let linear_up = (0..num_nodes * INPUT_ROW_LEN)
            .map(|index| (index as f32 % 37.0) * 0.002 - 0.03)
            .collect::<Vec<_>>();
        let edge_attrs = (0..num_edges * EDGE_ATTR_ROW_LEN)
            .map(|index| (index as f32 % 19.0) * 0.003 - 0.02)
            .collect::<Vec<_>>();
        let tp_weights = (0..num_edges * TP_WEIGHT_ROW_LEN)
            .map(|index| (index as f32 % 29.0) * 0.001 - 0.01)
            .collect::<Vec<_>>();
        let layer3_weight = (0..HIDDEN_DIM * TP_WEIGHT_ROW_LEN)
            .map(|index| (index as f32 % 31.0) * 0.0009 - 0.012)
            .collect::<Vec<_>>();
        let bar_scatter_message = (0..num_nodes * OUTPUT_ROW_LEN)
            .map(|index| (index as f32 % 23.0) * 0.0025 - 0.025)
            .collect::<Vec<_>>();

        let actual = match mp0b_layer1_conv_tp_vjp_hidden2_f32(
            num_nodes,
            &linear_up,
            &edge_attrs,
            &tp_weights,
            &layer3_weight,
            &bar_scatter_message,
            &senders,
            &receivers,
            None,
        ) {
            Ok(actual) => actual,
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        };

        let mut expected_linear_up = vec![0.0_f32; num_nodes * INPUT_ROW_LEN];
        let mut expected_edge_attrs = vec![0.0_f32; num_edges * EDGE_ATTR_ROW_LEN];
        let mut expected_tp_weights = vec![0.0_f32; num_edges * TP_WEIGHT_ROW_LEN];
        for edge in 0..num_edges {
            let sender = senders[edge];
            let receiver = receivers[edge];
            let x = &linear_up[sender * INPUT_ROW_LEN..(sender + 1) * INPUT_ROW_LEN];
            let y = &edge_attrs[edge * EDGE_ATTR_ROW_LEN..(edge + 1) * EDGE_ATTR_ROW_LEN];
            let w = &tp_weights[edge * TP_WEIGHT_ROW_LEN..(edge + 1) * TP_WEIGHT_ROW_LEN];
            let bar_scatter =
                &bar_scatter_message[receiver * OUTPUT_ROW_LEN..(receiver + 1) * OUTPUT_ROW_LEN];
            for (template_index, template) in
                super::super::MP0B_LAYER1_TP_TEMPLATES.iter().enumerate()
            {
                let cg_entries = super::super::MP0B_LAYER1_TP_CG_TEMPLATES_F32[template_index];
                for feature in 0..128 {
                    let weight_index = template.weight_base + feature;
                    let weight = w[weight_index];
                    let x_base = template.x_base + feature * template.x_dim;
                    let out_base = template.out_base + feature * template.out_dim;
                    let mut bar_weight = 0.0_f32;
                    for cg in cg_entries {
                        let bar_out = bar_scatter[out_base + cg.k_m];
                        let scale_bar = cg.scale * bar_out;
                        expected_linear_up[sender * INPUT_ROW_LEN + x_base + cg.i_m] +=
                            scale_bar * weight * y[template.y_base + cg.j_m];
                        expected_edge_attrs[edge * EDGE_ATTR_ROW_LEN + template.y_base + cg.j_m] +=
                            scale_bar * weight * x[x_base + cg.i_m];
                        bar_weight += scale_bar * x[x_base + cg.i_m] * y[template.y_base + cg.j_m];
                    }
                    expected_tp_weights[edge * TP_WEIGHT_ROW_LEN + weight_index] = bar_weight;
                }
            }
        }
        let mut expected_hidden2 = vec![0.0_f32; num_edges * HIDDEN_DIM];
        for edge in 0..num_edges {
            for hidden in 0..HIDDEN_DIM {
                let mut value = 0.0_f32;
                for out in 0..TP_WEIGHT_ROW_LEN {
                    value += expected_tp_weights[edge * TP_WEIGHT_ROW_LEN + out]
                        * layer3_weight[hidden * TP_WEIGHT_ROW_LEN + out];
                }
                expected_hidden2[edge * HIDDEN_DIM + hidden] = value * 0.125;
            }
        }

        for (label, actual, expected) in [
            (
                "bar_linear_up",
                actual.bar_linear_up.as_slice(),
                expected_linear_up.as_slice(),
            ),
            (
                "bar_edge_attrs",
                actual.bar_edge_attrs.as_slice(),
                expected_edge_attrs.as_slice(),
            ),
            (
                "bar_hidden2",
                actual.bar_hidden2.as_slice(),
                expected_hidden2.as_slice(),
            ),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                if (actual - expected).abs() > 5.0e-5 {
                    return Err(format!(
                        "{label} mismatch at {index}: actual={actual} expected={expected}"
                    ));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mace_scalar_contraction_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const NUM_NODES: usize = 3;
        const NUM_ELEMENTS: usize = 97;
        const NUM_FEATURES: usize = 7;
        const NUM_ELL: usize = 16;
        const U1_PARAMS: usize = 2;
        const U2_PARAMS: usize = 4;
        const U3_PARAMS: usize = 5;
        const OUT_STRIDE: usize = NUM_FEATURES + 2;
        const OUT_OFFSET: usize = 1;
        let type_ids = [2_u32, 88_u32, 96_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let weights_u1 = (0..NUM_ELEMENTS * U1_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let weights_u2 = (0..NUM_ELEMENTS * U2_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let weights_u3 = (0..NUM_ELEMENTS * U3_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let u1_entries = vec![
            super::super::ScalarU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            },
            super::super::ScalarU1EntryF32 {
                ell: 4,
                param: 1,
                value: -0.5,
            },
        ];
        let u2_entries = vec![
            super::super::ScalarU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            },
            super::super::ScalarU2EntryF32 {
                left: 2,
                right: 5,
                param: 3,
                value: -0.3,
            },
        ];
        let u3_entries = vec![
            super::super::ScalarU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            },
            super::super::ScalarU3EntryF32 {
                left: 1,
                right: 1,
                inner: 4,
                param: 1,
                value: -0.13,
            },
            super::super::ScalarU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 4,
                value: 0.17,
            },
            super::super::ScalarU3EntryF32 {
                left: 3,
                right: 7,
                inner: 9,
                param: 2,
                value: -0.19,
            },
        ];
        let mut actual = vec![-123.0_f32; NUM_NODES * OUT_STRIDE];
        match mace_scalar_contraction_forward_f32(
            &type_ids,
            &x,
            &weights_u1,
            &weights_u2,
            &weights_u3,
            NUM_ELEMENTS,
            NUM_FEATURES,
            NUM_ELL,
            U1_PARAMS,
            U2_PARAMS,
            U3_PARAMS,
            &u1_entries,
            &u2_entries,
            &u3_entries,
            OUT_STRIDE,
            OUT_OFFSET,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let mut expected = vec![-123.0_f32; NUM_NODES * OUT_STRIDE];
        for node in 0..NUM_NODES {
            let element = type_ids[node] as usize;
            for feature in 0..NUM_FEATURES {
                let x_base = (node * NUM_FEATURES + feature) * NUM_ELL;
                let mut acc = 0.0_f32;
                for entry in &u1_entries {
                    let weight =
                        weights_u1[(element * U1_PARAMS + entry.param) * NUM_FEATURES + feature];
                    acc += weight * entry.value * x[x_base + entry.ell];
                }
                for entry in &u2_entries {
                    let weight =
                        weights_u2[(element * U2_PARAMS + entry.param) * NUM_FEATURES + feature];
                    acc += weight * entry.value * x[x_base + entry.left] * x[x_base + entry.right];
                }
                for entry in &u3_entries {
                    let weight =
                        weights_u3[(element * U3_PARAMS + entry.param) * NUM_FEATURES + feature];
                    acc += weight
                        * entry.value
                        * x[x_base + entry.left]
                        * x[x_base + entry.right]
                        * x[x_base + entry.inner];
                }
                expected[node * OUT_STRIDE + OUT_OFFSET + feature] = acc;
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "scalar forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mace_scalar_contraction_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const NUM_NODES: usize = 3;
        const NUM_ELEMENTS: usize = 97;
        const NUM_FEATURES: usize = 7;
        const NUM_ELL: usize = 16;
        const U1_PARAMS: usize = 2;
        const U2_PARAMS: usize = 4;
        const U3_PARAMS: usize = 5;
        let type_ids = [2_u32, 88_u32, 96_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let grad_out = (0..NUM_NODES * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.002 - 0.025)
            .collect::<Vec<_>>();
        let weights_u1 = (0..NUM_ELEMENTS * U1_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let weights_u2 = (0..NUM_ELEMENTS * U2_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let weights_u3 = (0..NUM_ELEMENTS * U3_PARAMS * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let u1_entries = vec![
            super::super::ScalarU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            },
            super::super::ScalarU1EntryF32 {
                ell: 4,
                param: 1,
                value: -0.5,
            },
        ];
        let u2_entries = vec![
            super::super::ScalarU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            },
            super::super::ScalarU2EntryF32 {
                left: 2,
                right: 5,
                param: 3,
                value: -0.3,
            },
        ];
        let u3_entries = vec![
            super::super::ScalarU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            },
            super::super::ScalarU3EntryF32 {
                left: 1,
                right: 1,
                inner: 4,
                param: 1,
                value: -0.13,
            },
            super::super::ScalarU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 4,
                value: 0.17,
            },
            super::super::ScalarU3EntryF32 {
                left: 3,
                right: 7,
                inner: 9,
                param: 2,
                value: -0.19,
            },
        ];
        let mut actual = vec![0.0_f32; x.len()];
        match mace_scalar_contraction_vjp_f32(
            &type_ids,
            &x,
            &grad_out,
            &weights_u1,
            &weights_u2,
            &weights_u3,
            NUM_ELEMENTS,
            NUM_FEATURES,
            NUM_ELL,
            U1_PARAMS,
            U2_PARAMS,
            U3_PARAMS,
            &u1_entries,
            &u2_entries,
            &u3_entries,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let mut expected = vec![0.0_f32; x.len()];
        for node in 0..NUM_NODES {
            let element = type_ids[node] as usize;
            for feature in 0..NUM_FEATURES {
                let bar = grad_out[node * NUM_FEATURES + feature];
                let x_base = (node * NUM_FEATURES + feature) * NUM_ELL;
                for entry in &u1_entries {
                    let weight =
                        weights_u1[(element * U1_PARAMS + entry.param) * NUM_FEATURES + feature];
                    expected[x_base + entry.ell] += bar * entry.value * weight;
                }
                for entry in &u2_entries {
                    let weight =
                        weights_u2[(element * U2_PARAMS + entry.param) * NUM_FEATURES + feature];
                    let derivative = bar * entry.value * weight;
                    expected[x_base + entry.left] += derivative * x[x_base + entry.right];
                    expected[x_base + entry.right] += derivative * x[x_base + entry.left];
                }
                for entry in &u3_entries {
                    let weight =
                        weights_u3[(element * U3_PARAMS + entry.param) * NUM_FEATURES + feature];
                    let derivative = bar * entry.value * weight;
                    let left = x[x_base + entry.left];
                    let right = x[x_base + entry.right];
                    let inner = x[x_base + entry.inner];
                    if entry.left == entry.right {
                        if entry.right == entry.inner {
                            expected[x_base + entry.left] += 3.0 * derivative * left * left;
                        } else {
                            expected[x_base + entry.left] += 2.0 * derivative * left * inner;
                            expected[x_base + entry.inner] += derivative * left * left;
                        }
                    } else if entry.right == entry.inner {
                        expected[x_base + entry.left] += derivative * right * right;
                        expected[x_base + entry.right] += 2.0 * derivative * left * right;
                    } else {
                        expected[x_base + entry.left] += derivative * right * inner;
                        expected[x_base + entry.right] += derivative * left * inner;
                        expected[x_base + entry.inner] += derivative * left * right;
                    }
                }
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "scalar contraction VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_product0_vector_contraction_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const NUM_NODES: usize = 2;
        const NUM_ELEMENTS: usize = 89;
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        let type_ids = [2_u32, 7_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let grad_out = (0..NUM_NODES * NUM_FEATURES * NUM_EQUIV)
            .map(|index| (index as f32 % 29.0) * 0.002 - 0.025)
            .collect::<Vec<_>>();
        let weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let weights_u2 = (0..NUM_ELEMENTS * 6 * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let weights_u3 = (0..NUM_ELEMENTS * 51 * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let u1_entries = vec![
            vec![super::super::VectorU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 4,
                param: 0,
                value: -0.5,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 9,
                param: 0,
                value: 0.25,
            }],
        ];
        let u2_entries = vec![
            vec![
                super::super::VectorU2EntryF32 {
                    left: 1,
                    right: 1,
                    param: 0,
                    value: 0.7,
                },
                super::super::VectorU2EntryF32 {
                    left: 2,
                    right: 5,
                    param: 1,
                    value: -0.3,
                },
            ],
            vec![super::super::VectorU2EntryF32 {
                left: 3,
                right: 8,
                param: 2,
                value: 0.4,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 6,
                right: 6,
                param: 3,
                value: -0.2,
            }],
        ];
        let u3_entries = vec![
            vec![
                super::super::VectorU3EntryF32 {
                    left: 0,
                    right: 0,
                    inner: 0,
                    param: 0,
                    value: 0.11,
                },
                super::super::VectorU3EntryF32 {
                    left: 1,
                    right: 1,
                    inner: 4,
                    param: 1,
                    value: -0.13,
                },
            ],
            vec![super::super::VectorU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 2,
                value: 0.17,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            }],
        ];
        let mut actual = (0..x.len())
            .map(|index| (index as f32 % 13.0) * 0.0003)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        match mp0b_product0_vector_contraction_vjp_f32(
            &type_ids,
            &x,
            &grad_out,
            &weights_u1,
            &weights_u2,
            &weights_u3,
            &u1_entries,
            &u2_entries,
            &u3_entries,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        for (node, &type_id) in type_ids.iter().enumerate().take(NUM_NODES) {
            let element = type_id as usize;
            for feature in 0..NUM_FEATURES {
                let x_base = (node * NUM_FEATURES + feature) * NUM_ELL;
                let grad_out_base = (node * NUM_FEATURES + feature) * NUM_EQUIV;
                let mut grad_values = [0.0_f32; NUM_ELL];
                for equiv in 0..NUM_EQUIV {
                    let bar = grad_out[grad_out_base + equiv];
                    for entry in &u1_entries[equiv] {
                        let weight = weights_u1[(element + entry.param) * NUM_FEATURES + feature];
                        grad_values[entry.ell] += bar * weight * entry.value;
                    }
                    for entry in &u2_entries[equiv] {
                        let weight =
                            weights_u2[(element * 6 + entry.param) * NUM_FEATURES + feature];
                        let derivative = bar * weight * entry.value;
                        let left = x[x_base + entry.left];
                        let right = x[x_base + entry.right];
                        grad_values[entry.left] += derivative * right;
                        grad_values[entry.right] += derivative * left;
                    }
                    for entry in &u3_entries[equiv] {
                        let weight =
                            weights_u3[(element * 51 + entry.param) * NUM_FEATURES + feature];
                        let derivative = bar * weight * entry.value;
                        let left = x[x_base + entry.left];
                        let right = x[x_base + entry.right];
                        let inner = x[x_base + entry.inner];
                        if entry.left == entry.right {
                            if entry.right == entry.inner {
                                grad_values[entry.left] += 3.0 * derivative * left * left;
                            } else {
                                grad_values[entry.left] += 2.0 * derivative * left * inner;
                                grad_values[entry.inner] += derivative * left * left;
                            }
                        } else if entry.right == entry.inner {
                            grad_values[entry.left] += derivative * right * right;
                            grad_values[entry.right] += 2.0 * derivative * left * right;
                        } else {
                            grad_values[entry.left] += derivative * right * inner;
                            grad_values[entry.right] += derivative * left * inner;
                            grad_values[entry.inner] += derivative * left * right;
                        }
                    }
                }
                for ell in 0..NUM_ELL {
                    expected[x_base + ell] += grad_values[ell];
                }
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "product0 vector VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_product0_scalar_contraction_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const NUM_NODES: usize = 2;
        const NUM_ELEMENTS: usize = 89;
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        let type_ids = [3_u32, 11_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let grad_out = (0..NUM_NODES * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.002 - 0.025)
            .collect::<Vec<_>>();
        let weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let weights_u2 = (0..NUM_ELEMENTS * 4 * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let weights_u3 = (0..NUM_ELEMENTS * 23 * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let u1_entries = vec![
            super::super::ScalarU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            },
            super::super::ScalarU1EntryF32 {
                ell: 5,
                param: 0,
                value: -0.25,
            },
        ];
        let u2_entries = vec![
            super::super::ScalarU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            },
            super::super::ScalarU2EntryF32 {
                left: 2,
                right: 5,
                param: 1,
                value: -0.3,
            },
        ];
        let u3_entries = vec![
            super::super::ScalarU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            },
            super::super::ScalarU3EntryF32 {
                left: 1,
                right: 1,
                inner: 4,
                param: 1,
                value: -0.13,
            },
            super::super::ScalarU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 2,
                value: 0.17,
            },
            super::super::ScalarU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            },
        ];
        let mut actual = (0..x.len())
            .map(|index| (index as f32 % 13.0) * 0.0003)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();

        match mp0b_product0_scalar_contraction_vjp_f32(
            &type_ids,
            &x,
            &grad_out,
            &weights_u1,
            &weights_u2,
            &weights_u3,
            &u1_entries,
            &u2_entries,
            &u3_entries,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        for (node, &type_id) in type_ids.iter().enumerate().take(NUM_NODES) {
            let element = type_id as usize;
            for feature in 0..NUM_FEATURES {
                let x_base = (node * NUM_FEATURES + feature) * NUM_ELL;
                let bar = grad_out[node * NUM_FEATURES + feature];
                let mut grad_values = [0.0_f32; NUM_ELL];
                for entry in &u1_entries {
                    let weight = weights_u1[(element + entry.param) * NUM_FEATURES + feature];
                    grad_values[entry.ell] += bar * weight * entry.value;
                }
                for entry in &u2_entries {
                    let weight = weights_u2[(element * 4 + entry.param) * NUM_FEATURES + feature];
                    let derivative = bar * weight * entry.value;
                    let left = x[x_base + entry.left];
                    let right = x[x_base + entry.right];
                    grad_values[entry.left] += derivative * right;
                    grad_values[entry.right] += derivative * left;
                }
                for entry in &u3_entries {
                    let weight = weights_u3[(element * 23 + entry.param) * NUM_FEATURES + feature];
                    let derivative = bar * weight * entry.value;
                    let left = x[x_base + entry.left];
                    let right = x[x_base + entry.right];
                    let inner = x[x_base + entry.inner];
                    if entry.left == entry.right {
                        if entry.right == entry.inner {
                            grad_values[entry.left] += 3.0 * derivative * left * left;
                        } else {
                            grad_values[entry.left] += 2.0 * derivative * left * inner;
                            grad_values[entry.inner] += derivative * left * left;
                        }
                    } else if entry.right == entry.inner {
                        grad_values[entry.left] += derivative * right * right;
                        grad_values[entry.right] += 2.0 * derivative * left * right;
                    } else {
                        grad_values[entry.left] += derivative * right * inner;
                        grad_values[entry.right] += derivative * left * inner;
                        grad_values[entry.inner] += derivative * left * right;
                    }
                }
                for ell in 0..NUM_ELL {
                    expected[x_base + ell] += grad_values[ell];
                }
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "product0 scalar VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }

        let cache = mp0b_product0_scalar_contraction_vjp_cache_f32(
            &type_ids,
            &weights_u1,
            &weights_u2,
            &weights_u3,
            &u1_entries,
            &u2_entries,
            &u3_entries,
        )?;
        let mut cached_actual = (0..x.len())
            .map(|index| (index as f32 % 13.0) * 0.0003)
            .collect::<Vec<_>>();
        mp0b_product0_scalar_contraction_vjp_with_cache_f32(
            &cache,
            &x,
            &grad_out,
            &mut cached_actual,
        )?;
        for (index, (&actual, &expected)) in cached_actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "cached product0 scalar VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }

        Ok(())
    }

    #[test]
    fn mp0b_product0_scalar_vector_contraction_vjp_matches_separate_kernels() -> Result<(), String>
    {
        const NUM_NODES: usize = 2;
        const NUM_ELEMENTS: usize = 89;
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        let type_ids = [5_u32, 13_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let scalar_grad_out = (0..NUM_NODES * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.002 - 0.025)
            .collect::<Vec<_>>();
        let vector_grad_out = (0..NUM_NODES * NUM_FEATURES * NUM_EQUIV)
            .map(|index| (index as f32 % 23.0) * 0.0017 - 0.018)
            .collect::<Vec<_>>();
        let scalar_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let scalar_weights_u2 = (0..NUM_ELEMENTS * 4 * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let scalar_weights_u3 = (0..NUM_ELEMENTS * 23 * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let vector_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 19.0) * 0.0012 - 0.009)
            .collect::<Vec<_>>();
        let vector_weights_u2 = (0..NUM_ELEMENTS * 6 * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.0009 - 0.011)
            .collect::<Vec<_>>();
        let vector_weights_u3 = (0..NUM_ELEMENTS * 51 * NUM_FEATURES)
            .map(|index| (index as f32 % 41.0) * 0.0006 - 0.014)
            .collect::<Vec<_>>();
        let scalar_u1_entries = vec![super::super::ScalarU1EntryF32 {
            ell: 0,
            param: 0,
            value: 1.0,
        }];
        let scalar_u2_entries = vec![super::super::ScalarU2EntryF32 {
            left: 2,
            right: 5,
            param: 1,
            value: -0.3,
        }];
        let scalar_u3_entries = vec![super::super::ScalarU3EntryF32 {
            left: 3,
            right: 7,
            inner: 11,
            param: 3,
            value: -0.19,
        }];
        let vector_u1_entries = vec![
            vec![super::super::VectorU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 4,
                param: 0,
                value: -0.5,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 9,
                param: 0,
                value: 0.25,
            }],
        ];
        let vector_u2_entries = vec![
            vec![super::super::VectorU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 2,
                right: 5,
                param: 1,
                value: -0.3,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 6,
                right: 6,
                param: 3,
                value: -0.2,
            }],
        ];
        let vector_u3_entries = vec![
            vec![super::super::VectorU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 2,
                value: 0.17,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            }],
        ];
        let initial = (0..x.len())
            .map(|index| (index as f32 % 13.0) * 0.0003)
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        match mp0b_product0_scalar_contraction_vjp_f32(
            &type_ids,
            &x,
            &scalar_grad_out,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &mut expected,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }
        mp0b_product0_vector_contraction_vjp_f32(
            &type_ids,
            &x,
            &vector_grad_out,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
            &mut expected,
        )?;

        let mut actual = initial;
        mp0b_product0_scalar_vector_contraction_vjp_f32(
            &type_ids,
            &x,
            &scalar_grad_out,
            &vector_grad_out,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
            &mut actual,
        )?;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "product0 scalar/vector VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_product0_linear_scalar_vector_contraction_vjp_matches_staged_path() -> Result<(), String>
    {
        const NUM_NODES: usize = 2;
        const NUM_ELEMENTS: usize = 89;
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        const ROW_LEN: usize = 512;
        let type_ids = [5_u32, 13_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let bar_product_out = (0..NUM_NODES * ROW_LEN)
            .map(|index| (index as f32 % 23.0) * 0.0017 - 0.018)
            .collect::<Vec<_>>();
        let product_linear_weight = (0..2 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 31.0) * 0.0007 - 0.01)
            .collect::<Vec<_>>();
        let product_linear_output_mask = (0..ROW_LEN)
            .map(|index| if index % 19 == 0 { 0.25 } else { 1.0 })
            .collect::<Vec<_>>();
        let scalar_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let scalar_weights_u2 = (0..NUM_ELEMENTS * 4 * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let scalar_weights_u3 = (0..NUM_ELEMENTS * 23 * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let vector_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 19.0) * 0.0012 - 0.009)
            .collect::<Vec<_>>();
        let vector_weights_u2 = (0..NUM_ELEMENTS * 6 * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.0009 - 0.011)
            .collect::<Vec<_>>();
        let vector_weights_u3 = (0..NUM_ELEMENTS * 51 * NUM_FEATURES)
            .map(|index| (index as f32 % 41.0) * 0.0006 - 0.014)
            .collect::<Vec<_>>();
        let scalar_u1_entries = vec![super::super::ScalarU1EntryF32 {
            ell: 0,
            param: 0,
            value: 1.0,
        }];
        let scalar_u2_entries = vec![super::super::ScalarU2EntryF32 {
            left: 2,
            right: 5,
            param: 1,
            value: -0.3,
        }];
        let scalar_u3_entries = vec![super::super::ScalarU3EntryF32 {
            left: 3,
            right: 7,
            inner: 11,
            param: 3,
            value: -0.19,
        }];
        let vector_u1_entries = vec![
            vec![super::super::VectorU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 4,
                param: 0,
                value: -0.5,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 9,
                param: 0,
                value: 0.25,
            }],
        ];
        let vector_u2_entries = vec![
            vec![super::super::VectorU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 2,
                right: 5,
                param: 1,
                value: -0.3,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 6,
                right: 6,
                param: 3,
                value: -0.2,
            }],
        ];
        let vector_u3_entries = vec![
            vec![super::super::VectorU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 2,
                value: 0.17,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            }],
        ];

        let mut product_grad = vec![0.0_f32; NUM_NODES * ROW_LEN];
        match mp0b_scalar_vector_product_linear_vjp_input_f32(
            &product_linear_weight,
            &product_linear_output_mask,
            &bar_product_out,
            &mut product_grad,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }
        let mut scalar_grad = vec![0.0_f32; NUM_NODES * NUM_FEATURES];
        let mut vector_grad = vec![0.0_f32; NUM_NODES * NUM_FEATURES * NUM_EQUIV];
        for node in 0..NUM_NODES {
            let product_base = node * ROW_LEN;
            let scalar_base = node * NUM_FEATURES;
            let vector_base = node * NUM_FEATURES * NUM_EQUIV;
            scalar_grad[scalar_base..scalar_base + NUM_FEATURES]
                .copy_from_slice(&product_grad[product_base..product_base + NUM_FEATURES]);
            vector_grad[vector_base..vector_base + NUM_FEATURES * NUM_EQUIV].copy_from_slice(
                &product_grad[product_base + NUM_FEATURES..product_base + ROW_LEN],
            );
        }

        let initial = (0..x.len())
            .map(|index| (index as f32 % 13.0) * 0.0003)
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        mp0b_product0_scalar_vector_contraction_vjp_f32(
            &type_ids,
            &x,
            &scalar_grad,
            &vector_grad,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
            &mut expected,
        )?;

        let mut actual = initial;
        mp0b_product0_linear_scalar_vector_contraction_vjp_f32(
            &type_ids,
            &x,
            &bar_product_out,
            &product_linear_weight,
            &product_linear_output_mask,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
            &mut actual,
        )?;

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 2.0e-5 {
                return Err(format!(
                    "product0 linear scalar/vector VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_product0_contraction_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const NUM_NODES: usize = 2;
        const NUM_ELEMENTS: usize = 89;
        const NUM_FEATURES: usize = 128;
        const NUM_ELL: usize = 16;
        const NUM_EQUIV: usize = 3;
        let type_ids = [5_u32, 13_u32];
        let x = (0..NUM_NODES * NUM_FEATURES * NUM_ELL)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let scalar_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0015 - 0.01)
            .collect::<Vec<_>>();
        let scalar_weights_u2 = (0..NUM_ELEMENTS * 4 * NUM_FEATURES)
            .map(|index| (index as f32 % 23.0) * 0.001 - 0.012)
            .collect::<Vec<_>>();
        let scalar_weights_u3 = (0..NUM_ELEMENTS * 23 * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0007 - 0.015)
            .collect::<Vec<_>>();
        let vector_weights_u1 = (0..NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 19.0) * 0.0012 - 0.009)
            .collect::<Vec<_>>();
        let vector_weights_u2 = (0..NUM_ELEMENTS * 6 * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.0009 - 0.011)
            .collect::<Vec<_>>();
        let vector_weights_u3 = (0..NUM_ELEMENTS * 51 * NUM_FEATURES)
            .map(|index| (index as f32 % 41.0) * 0.0006 - 0.014)
            .collect::<Vec<_>>();
        let scalar_u1_entries = vec![super::super::ScalarU1EntryF32 {
            ell: 0,
            param: 0,
            value: 1.0,
        }];
        let scalar_u2_entries = vec![
            super::super::ScalarU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            },
            super::super::ScalarU2EntryF32 {
                left: 2,
                right: 5,
                param: 1,
                value: -0.3,
            },
        ];
        let scalar_u3_entries = vec![
            super::super::ScalarU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            },
            super::super::ScalarU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            },
        ];
        let vector_u1_entries = vec![
            vec![super::super::VectorU1EntryF32 {
                ell: 0,
                param: 0,
                value: 1.0,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 4,
                param: 0,
                value: -0.5,
            }],
            vec![super::super::VectorU1EntryF32 {
                ell: 9,
                param: 0,
                value: 0.25,
            }],
        ];
        let vector_u2_entries = vec![
            vec![super::super::VectorU2EntryF32 {
                left: 1,
                right: 1,
                param: 0,
                value: 0.7,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 2,
                right: 5,
                param: 1,
                value: -0.3,
            }],
            vec![super::super::VectorU2EntryF32 {
                left: 6,
                right: 6,
                param: 3,
                value: -0.2,
            }],
        ];
        let vector_u3_entries = vec![
            vec![super::super::VectorU3EntryF32 {
                left: 0,
                right: 0,
                inner: 0,
                param: 0,
                value: 0.11,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 2,
                right: 5,
                inner: 5,
                param: 2,
                value: 0.17,
            }],
            vec![super::super::VectorU3EntryF32 {
                left: 3,
                right: 7,
                inner: 11,
                param: 3,
                value: -0.19,
            }],
        ];
        let mut actual = vec![0.0_f32; NUM_NODES * 512];
        match mp0b_product0_contraction_forward_f32(
            &type_ids,
            &x,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let cache = mp0b_product0_forward_cache_f32(
            &type_ids,
            &scalar_weights_u1,
            &scalar_weights_u2,
            &scalar_weights_u3,
            &scalar_u1_entries,
            &scalar_u2_entries,
            &scalar_u3_entries,
            &vector_weights_u1,
            &vector_weights_u2,
            &vector_weights_u3,
            &vector_u1_entries,
            &vector_u2_entries,
            &vector_u3_entries,
        )?;
        let mut cached_actual = vec![0.0_f32; actual.len()];
        let cached_buffer = mp0b_product0_contraction_forward_with_static_cache_skip_readback_f32(
            &x,
            None,
            &cache,
            &mut cached_actual,
        )?;
        mp0b_copy_f32_buffer_to_slice(&cached_buffer, &mut cached_actual)?;
        for (index, (&cached, &direct)) in cached_actual.iter().zip(&actual).enumerate() {
            if (cached - direct).abs() > 1.0e-6 {
                return Err(format!(
                    "product0 cached contraction forward mismatch at {index}: cached={cached} direct={direct}"
                ));
            }
        }

        let mut expected = vec![0.0_f32; actual.len()];
        for (node, &type_id) in type_ids.iter().enumerate().take(NUM_NODES) {
            let element = type_id as usize;
            for feature in 0..NUM_FEATURES {
                let x_base = (node * NUM_FEATURES + feature) * NUM_ELL;
                let product_base = node * 512;
                let mut scalar = 0.0_f32;
                for entry in &scalar_u1_entries {
                    scalar += scalar_weights_u1[(element + entry.param) * NUM_FEATURES + feature]
                        * entry.value
                        * x[x_base + entry.ell];
                }
                for entry in &scalar_u2_entries {
                    scalar += scalar_weights_u2
                        [(element * 4 + entry.param) * NUM_FEATURES + feature]
                        * entry.value
                        * x[x_base + entry.left]
                        * x[x_base + entry.right];
                }
                for entry in &scalar_u3_entries {
                    scalar += scalar_weights_u3
                        [(element * 23 + entry.param) * NUM_FEATURES + feature]
                        * entry.value
                        * x[x_base + entry.left]
                        * x[x_base + entry.right]
                        * x[x_base + entry.inner];
                }
                expected[product_base + feature] = scalar;

                for equiv in 0..NUM_EQUIV {
                    let mut vector = 0.0_f32;
                    for entry in &vector_u1_entries[equiv] {
                        vector += vector_weights_u1
                            [(element + entry.param) * NUM_FEATURES + feature]
                            * entry.value
                            * x[x_base + entry.ell];
                    }
                    for entry in &vector_u2_entries[equiv] {
                        vector += vector_weights_u2
                            [(element * 6 + entry.param) * NUM_FEATURES + feature]
                            * entry.value
                            * x[x_base + entry.left]
                            * x[x_base + entry.right];
                    }
                    for entry in &vector_u3_entries[equiv] {
                        vector += vector_weights_u3
                            [(element * 51 + entry.param) * NUM_FEATURES + feature]
                            * entry.value
                            * x[x_base + entry.left]
                            * x[x_base + entry.right]
                            * x[x_base + entry.inner];
                    }
                    expected[product_base + 128 + feature * NUM_EQUIV + equiv] = vector;
                }
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-6 {
                return Err(format!(
                    "product0 contraction forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer1_hidden_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const INPUT_ROW_LEN: usize = 5120;
        const OUTPUT_ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        let num_nodes = 2usize;
        let input = (0..num_nodes * INPUT_ROW_LEN)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let density = (0..num_nodes)
            .map(|index| 0.2 + index as f32 * 0.1)
            .collect::<Vec<_>>();
        let weight = (0..10 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0008 - 0.012)
            .collect::<Vec<_>>();
        let output_mask = (0..OUTPUT_ROW_LEN)
            .map(|index| if index % 19 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let mut actual = vec![0.0_f32; num_nodes * OUTPUT_ROW_LEN];

        match mp0b_layer1_hidden_forward_f32(&input, &density, &weight, &output_mask, &mut actual) {
            Ok(_) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let input_block_dims = [1usize, 1, 3, 3, 3, 5, 5, 5, 7, 7];
        let output_block_dims = [1usize, 3, 5, 7];
        let output_blocks = [0usize, 0, 1, 1, 1, 2, 2, 2, 3, 3];
        let mut input_offsets = [0usize; 10];
        for block in 1..input_block_dims.len() {
            input_offsets[block] =
                input_offsets[block - 1] + NUM_FEATURES * input_block_dims[block - 1];
        }
        let mut output_offsets = [0usize; 4];
        for block in 1..output_block_dims.len() {
            output_offsets[block] =
                output_offsets[block - 1] + NUM_FEATURES * output_block_dims[block - 1];
        }
        let mut expected = vec![0.0_f32; actual.len()];
        for node in 0..num_nodes {
            let input_row = &input[node * INPUT_ROW_LEN..(node + 1) * INPUT_ROW_LEN];
            let expected_row = &mut expected[node * OUTPUT_ROW_LEN..(node + 1) * OUTPUT_ROW_LEN];
            for (block, &component_dim) in input_block_dims.iter().enumerate() {
                let input_offset = input_offsets[block];
                let output_offset = output_offsets[output_blocks[block]];
                let weight_offset = block * NUM_FEATURES * NUM_FEATURES;
                for component in 0..component_dim {
                    for out_feature in 0..NUM_FEATURES {
                        let mut value = 0.0_f32;
                        for in_feature in 0..NUM_FEATURES {
                            value += input_row
                                [input_offset + in_feature * component_dim + component]
                                * weight[weight_offset + in_feature * NUM_FEATURES + out_feature];
                        }
                        expected_row[output_offset + out_feature * component_dim + component] +=
                            value;
                    }
                }
            }
            let density_norm = density[node] + 1.0;
            for (value, &mask) in expected_row.iter_mut().zip(&output_mask) {
                *value = *value * mask / density_norm;
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer1 hidden forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer0_hidden_forward_kernel_matches_cpu_formula() -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        let num_nodes = 2usize;
        let input = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let density = (0..num_nodes)
            .map(|index| 0.15 + index as f32 * 0.12)
            .collect::<Vec<_>>();
        let weight = (0..4 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.0008 - 0.012)
            .collect::<Vec<_>>();
        let output_mask = (0..ROW_LEN)
            .map(|index| if index % 19 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let mut actual = vec![0.0_f32; num_nodes * ROW_LEN];

        match mp0b_layer0_hidden_forward_f32(&input, &density, &weight, &output_mask, &mut actual) {
            Ok(_metal) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let block_dims = [1usize, 3, 5, 7];
        let mut block_offsets = [0usize; 4];
        for block in 1..block_dims.len() {
            block_offsets[block] = block_offsets[block - 1] + NUM_FEATURES * block_dims[block - 1];
        }
        let path_weight = 1.0 / (NUM_FEATURES as f32).sqrt();
        let mut expected = vec![0.0_f32; actual.len()];
        for node in 0..num_nodes {
            let input_row = &input[node * ROW_LEN..(node + 1) * ROW_LEN];
            let expected_row = &mut expected[node * ROW_LEN..(node + 1) * ROW_LEN];
            for (block, &component_dim) in block_dims.iter().enumerate() {
                let offset = block_offsets[block];
                let weight_offset = block * NUM_FEATURES * NUM_FEATURES;
                for component in 0..component_dim {
                    for out_feature in 0..NUM_FEATURES {
                        let mut value = 0.0_f32;
                        for in_feature in 0..NUM_FEATURES {
                            value += input_row[offset + in_feature * component_dim + component]
                                * path_weight
                                * weight[weight_offset + in_feature * NUM_FEATURES + out_feature];
                        }
                        let out_index = offset + out_feature * component_dim + component;
                        expected_row[out_index] =
                            value * output_mask[out_index] / (density[node] + 1.0);
                    }
                }
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer0 hidden forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_hidden_message_selector_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        const NUM_ELEMENTS: usize = 89;
        let type_ids = [2_u32, 7_u32];
        let num_nodes = type_ids.len();
        let weight = (0..4 * NUM_FEATURES * NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0007 - 0.006)
            .collect::<Vec<_>>();
        let output_mask = (0..ROW_LEN)
            .map(|index| if index % 17 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let bar_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 23.0) * 0.002 - 0.02)
            .collect::<Vec<_>>();
        let mut actual_input = vec![0.0_f32; num_nodes * ROW_LEN];

        match mp0b_hidden_message_selector_vjp_input_from_feature_major_f32(
            &type_ids,
            &weight,
            &output_mask,
            &bar_out,
            &mut actual_input,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let block_dims = [1usize, 3, 5, 7];
        let path_weights = [
            1.0 / ((NUM_FEATURES * NUM_ELEMENTS) as f32).sqrt(),
            0.016_227_837_f32,
            0.020_950_048_f32,
            0.024_788_43_f32,
        ];
        let mut block_offsets = [0usize; 4];
        let mut component_offsets = [0usize; 4];
        for block in 1..block_dims.len() {
            block_offsets[block] = block_offsets[block - 1] + NUM_FEATURES * block_dims[block - 1];
            component_offsets[block] = component_offsets[block - 1] + block_dims[block - 1];
        }
        let block_weights = NUM_FEATURES * NUM_ELEMENTS * NUM_FEATURES;
        let mut expected_input = vec![0.0_f32; actual_input.len()];
        for (node, &type_id) in type_ids.iter().enumerate() {
            let type_id = type_id as usize;
            let bar_out_row = &bar_out[node * ROW_LEN..(node + 1) * ROW_LEN];
            let expected_row = &mut expected_input[node * ROW_LEN..(node + 1) * ROW_LEN];
            for (block, &component_dim) in block_dims.iter().enumerate() {
                let block_offset = block_offsets[block];
                let component_offset = component_offsets[block];
                let weight_offset = block * block_weights;
                let block_scale = path_weights[block] / (component_dim as f32).sqrt();
                for component in 0..component_dim {
                    for in_feature in 0..NUM_FEATURES {
                        let weight_row_offset =
                            weight_offset + (in_feature * NUM_ELEMENTS + type_id) * NUM_FEATURES;
                        let mut value = 0.0_f32;
                        for out_feature in 0..NUM_FEATURES {
                            let hidden_index =
                                block_offset + out_feature * component_dim + component;
                            let feature_major_index =
                                out_feature * 16 + component_offset + component;
                            value += bar_out_row[feature_major_index]
                                * block_scale
                                * output_mask[hidden_index]
                                * weight[weight_row_offset + out_feature];
                        }
                        expected_row[block_offset + in_feature * component_dim + component] = value;
                    }
                }
            }
        }
        for (index, (&actual, &expected)) in actual_input.iter().zip(&expected_input).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "hidden message selector VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_scalar_vector_product_linear_forward_kernel_matches_cpu_formula() -> Result<(), String>
    {
        const ROW_LEN: usize = 512;
        const NUM_FEATURES: usize = 128;
        const VECTOR_EQUIVARIANCE: usize = 3;
        let num_nodes = 3usize;
        let input = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 37.0) * 0.0015 - 0.025)
            .collect::<Vec<_>>();
        let weight = (0..2 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 29.0) * 0.0008 - 0.011)
            .collect::<Vec<_>>();
        let output_mask = (0..ROW_LEN)
            .map(|index| if index % 23 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let mut actual = vec![0.0_f32; input.len()];

        match mp0b_scalar_vector_product_linear_forward_f32(
            &input,
            &weight,
            &output_mask,
            &mut actual,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let block_weights = NUM_FEATURES * NUM_FEATURES;
        let mut expected = vec![0.0_f32; actual.len()];
        for node in 0..num_nodes {
            let input_row = &input[node * ROW_LEN..(node + 1) * ROW_LEN];
            let expected_row = &mut expected[node * ROW_LEN..(node + 1) * ROW_LEN];
            for out_feature in 0..NUM_FEATURES {
                let mut value = 0.0_f32;
                for in_feature in 0..NUM_FEATURES {
                    value +=
                        input_row[in_feature] * weight[in_feature * NUM_FEATURES + out_feature];
                }
                expected_row[out_feature] = value * output_mask[out_feature];
            }
            for out_feature in 0..NUM_FEATURES {
                for component in 0..VECTOR_EQUIVARIANCE {
                    let out_index = NUM_FEATURES + out_feature * VECTOR_EQUIVARIANCE + component;
                    let mut value = 0.0_f32;
                    for in_feature in 0..NUM_FEATURES {
                        let in_index = NUM_FEATURES + in_feature * VECTOR_EQUIVARIANCE + component;
                        let weight_index = block_weights + in_feature * NUM_FEATURES + out_feature;
                        value += input_row[in_index] * weight[weight_index];
                    }
                    expected_row[out_index] = value * output_mask[out_index];
                }
            }
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "scalar-vector product linear forward mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_scalar_vector_product_linear_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const ROW_LEN: usize = 512;
        const NUM_FEATURES: usize = 128;
        const VECTOR_EQUIVARIANCE: usize = 3;
        let num_nodes = 3usize;
        let weight = (0..2 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 31.0) * 0.0007 - 0.01)
            .collect::<Vec<_>>();
        let output_mask = (0..ROW_LEN)
            .map(|index| if index % 19 == 0 { 0.25 } else { 1.0 })
            .collect::<Vec<_>>();
        let bar_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 23.0) * 0.002 - 0.021)
            .collect::<Vec<_>>();
        let initial_input = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 11.0) * 0.0009)
            .collect::<Vec<_>>();
        let mut actual_input = initial_input.clone();

        match mp0b_scalar_vector_product_linear_vjp_input_f32(
            &weight,
            &output_mask,
            &bar_out,
            &mut actual_input,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let block_weights = NUM_FEATURES * NUM_FEATURES;
        let mut expected_input = initial_input;
        for node in 0..num_nodes {
            let bar_out_row = &bar_out[node * ROW_LEN..(node + 1) * ROW_LEN];
            let expected_row = &mut expected_input[node * ROW_LEN..(node + 1) * ROW_LEN];
            for in_feature in 0..NUM_FEATURES {
                let mut value = 0.0_f32;
                for out_feature in 0..NUM_FEATURES {
                    value += bar_out_row[out_feature]
                        * output_mask[out_feature]
                        * weight[in_feature * NUM_FEATURES + out_feature];
                }
                expected_row[in_feature] += value;
            }
            for in_feature in 0..NUM_FEATURES {
                for component in 0..VECTOR_EQUIVARIANCE {
                    let in_index = NUM_FEATURES + in_feature * VECTOR_EQUIVARIANCE + component;
                    let mut value = 0.0_f32;
                    for out_feature in 0..NUM_FEATURES {
                        let out_index =
                            NUM_FEATURES + out_feature * VECTOR_EQUIVARIANCE + component;
                        let weight_index = block_weights + in_feature * NUM_FEATURES + out_feature;
                        value +=
                            bar_out_row[out_index] * output_mask[out_index] * weight[weight_index];
                    }
                    expected_row[in_index] += value;
                }
            }
        }
        for (index, (&actual, &expected)) in actual_input.iter().zip(&expected_input).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "scalar-vector product linear VJP mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer0_hidden_linear_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        let num_nodes = 2usize;
        let density = (0..num_nodes)
            .map(|index| 0.2 + index as f32 * 0.15)
            .collect::<Vec<_>>();
        let linear_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 43.0) * 0.0015 - 0.025)
            .collect::<Vec<_>>();
        let weight = (0..4 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let output_mask = (0..ROW_LEN)
            .map(|index| if index % 13 == 0 { 0.25 } else { 1.0 })
            .collect::<Vec<_>>();
        let bar_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 29.0) * 0.002 - 0.035)
            .collect::<Vec<_>>();
        let mut actual_input = vec![0.0_f32; num_nodes * ROW_LEN];
        let mut actual_density = vec![0.0_f32; num_nodes];

        match mp0b_layer0_hidden_linear_vjp_input_density_f32(
            &density,
            &linear_out,
            &weight,
            &output_mask,
            &bar_out,
            &mut actual_input,
            &mut actual_density,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let block_dims = [1usize, 3, 5, 7];
        let mut block_offsets = [0usize; 4];
        for block in 1..block_dims.len() {
            block_offsets[block] = block_offsets[block - 1] + NUM_FEATURES * block_dims[block - 1];
        }
        let path_weight = 1.0 / (NUM_FEATURES as f32).sqrt();
        let mut expected_input = vec![0.0_f32; actual_input.len()];
        let mut expected_density = vec![0.0_f32; actual_density.len()];
        for node in 0..num_nodes {
            let density_norm = density[node] + 1.0;
            let linear_out_row = &linear_out[node * ROW_LEN..(node + 1) * ROW_LEN];
            let bar_out_row = &bar_out[node * ROW_LEN..(node + 1) * ROW_LEN];
            for index in 0..ROW_LEN {
                expected_density[node] -= bar_out_row[index] * linear_out_row[index] / density_norm;
            }
            for (block, &component_dim) in block_dims.iter().enumerate() {
                let offset = block_offsets[block];
                let weight_offset = block * NUM_FEATURES * NUM_FEATURES;
                for component in 0..component_dim {
                    for in_feature in 0..NUM_FEATURES {
                        let mut value = 0.0_f32;
                        for out_feature in 0..NUM_FEATURES {
                            let out_index = offset + out_feature * component_dim + component;
                            value += bar_out_row[out_index]
                                * output_mask[out_index]
                                * weight[weight_offset + in_feature * NUM_FEATURES + out_feature]
                                * path_weight
                                / density_norm;
                        }
                        expected_input
                            [node * ROW_LEN + offset + in_feature * component_dim + component] =
                            value;
                    }
                }
            }
        }
        for (index, (&actual, &expected)) in actual_input.iter().zip(&expected_input).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer0 hidden VJP input mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        for (index, (&actual, &expected)) in
            actual_density.iter().zip(&expected_density).enumerate()
        {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer0 hidden VJP density mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer0_selector_hidden_vjp_matches_separate_kernels() -> Result<(), String> {
        const ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        const NUM_ELEMENTS: usize = 89;
        let type_ids = [2_u32, 7_u32];
        let num_nodes = type_ids.len();
        let selector_weight = (0..4 * NUM_FEATURES * NUM_ELEMENTS * NUM_FEATURES)
            .map(|index| (index as f32 % 17.0) * 0.0007 - 0.006)
            .collect::<Vec<_>>();
        let selector_output_mask = (0..ROW_LEN)
            .map(|index| if index % 17 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let bar_interaction_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 23.0) * 0.002 - 0.02)
            .collect::<Vec<_>>();
        let density = (0..num_nodes)
            .map(|index| 0.2 + index as f32 * 0.15)
            .collect::<Vec<_>>();
        let linear_out = (0..num_nodes * ROW_LEN)
            .map(|index| (index as f32 % 43.0) * 0.0015 - 0.025)
            .collect::<Vec<_>>();
        let hidden_weight = (0..4 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 31.0) * 0.001 - 0.015)
            .collect::<Vec<_>>();
        let hidden_output_mask = (0..ROW_LEN)
            .map(|index| if index % 13 == 0 { 0.25 } else { 1.0 })
            .collect::<Vec<_>>();

        let mut bar_linear_message = vec![0.0_f32; num_nodes * ROW_LEN];
        match mp0b_hidden_message_selector_vjp_input_from_feature_major_f32(
            &type_ids,
            &selector_weight,
            &selector_output_mask,
            &bar_interaction_out,
            &mut bar_linear_message,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }
        let mut expected_input = vec![0.0_f32; num_nodes * ROW_LEN];
        let mut expected_density = vec![0.0_f32; num_nodes];
        mp0b_layer0_hidden_linear_vjp_input_density_f32(
            &density,
            &linear_out,
            &hidden_weight,
            &hidden_output_mask,
            &bar_linear_message,
            &mut expected_input,
            &mut expected_density,
        )?;

        let mut actual_input = vec![0.0_f32; num_nodes * ROW_LEN];
        let mut actual_density = vec![0.0_f32; num_nodes];
        mp0b_layer0_selector_hidden_vjp_input_density_f32(
            &type_ids,
            &selector_weight,
            &selector_output_mask,
            &bar_interaction_out,
            &density,
            &linear_out,
            &hidden_weight,
            &hidden_output_mask,
            &mut actual_input,
            &mut actual_density,
        )?;

        for (index, (&actual, &expected)) in actual_input.iter().zip(&expected_input).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer0 selector/hidden input mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        for (index, (&actual, &expected)) in
            actual_density.iter().zip(&expected_density).enumerate()
        {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer0 selector/hidden density mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mp0b_layer1_hidden_linear_vjp_kernel_matches_cpu_formula() -> Result<(), String> {
        const INPUT_ROW_LEN: usize = 5120;
        const OUTPUT_ROW_LEN: usize = 2048;
        const NUM_FEATURES: usize = 128;
        let num_nodes = 2usize;
        let density = (0..num_nodes)
            .map(|index| 0.25 + index as f32 * 0.1)
            .collect::<Vec<_>>();
        let linear_out = (0..num_nodes * OUTPUT_ROW_LEN)
            .map(|index| (index as f32 % 41.0) * 0.002 - 0.03)
            .collect::<Vec<_>>();
        let weight = (0..10 * NUM_FEATURES * NUM_FEATURES)
            .map(|index| (index as f32 % 37.0) * 0.001 - 0.02)
            .collect::<Vec<_>>();
        let output_mask = (0..OUTPUT_ROW_LEN)
            .map(|index| if index % 11 == 0 { 0.5 } else { 1.0 })
            .collect::<Vec<_>>();
        let bar_out = (0..num_nodes * OUTPUT_ROW_LEN)
            .map(|index| (index as f32 % 29.0) * 0.003 - 0.04)
            .collect::<Vec<_>>();
        let mut actual_input = vec![0.0_f32; num_nodes * INPUT_ROW_LEN];
        let mut actual_density = vec![0.0_f32; num_nodes];

        match mp0b_layer1_hidden_linear_vjp_input_density_f32(
            &density,
            &linear_out,
            &weight,
            &output_mask,
            &bar_out,
            &mut actual_input,
            &mut actual_density,
        ) {
            Ok(()) => {}
            Err(err) if err.contains("no Metal device") => return Ok(()),
            Err(err) => return Err(err),
        }

        let input_block_dims = [1usize, 1, 3, 3, 3, 5, 5, 5, 7, 7];
        let output_blocks = [0usize, 0, 1, 1, 1, 2, 2, 2, 3, 3];
        let mut input_offsets = [0usize; 10];
        for block in 1..input_block_dims.len() {
            input_offsets[block] =
                input_offsets[block - 1] + NUM_FEATURES * input_block_dims[block - 1];
        }
        let output_block_dims = [1usize, 3, 5, 7];
        let mut output_offsets = [0usize; 4];
        for block in 1..output_block_dims.len() {
            output_offsets[block] =
                output_offsets[block - 1] + NUM_FEATURES * output_block_dims[block - 1];
        }
        let mut expected_input = vec![0.0_f32; actual_input.len()];
        let mut expected_density = vec![0.0_f32; actual_density.len()];
        for node in 0..num_nodes {
            let density_norm = density[node] + 1.0;
            let linear_out_row = &linear_out[node * OUTPUT_ROW_LEN..(node + 1) * OUTPUT_ROW_LEN];
            let bar_out_row = &bar_out[node * OUTPUT_ROW_LEN..(node + 1) * OUTPUT_ROW_LEN];
            for index in 0..OUTPUT_ROW_LEN {
                expected_density[node] -= bar_out_row[index] * linear_out_row[index] / density_norm;
            }
            for (block, &component_dim) in input_block_dims.iter().enumerate() {
                let input_offset = input_offsets[block];
                let output_offset = output_offsets[output_blocks[block]];
                let weight_offset = block * NUM_FEATURES * NUM_FEATURES;
                for component in 0..component_dim {
                    for in_feature in 0..NUM_FEATURES {
                        let mut value = 0.0_f32;
                        for out_feature in 0..NUM_FEATURES {
                            let out_index = output_offset + out_feature * component_dim + component;
                            value += bar_out_row[out_index]
                                * output_mask[out_index]
                                * weight[weight_offset + in_feature * NUM_FEATURES + out_feature]
                                / density_norm;
                        }
                        expected_input[node * INPUT_ROW_LEN
                            + input_offset
                            + in_feature * component_dim
                            + component] = value;
                    }
                }
            }
        }
        for (index, (&actual, &expected)) in actual_input.iter().zip(&expected_input).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer1 hidden VJP input mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        for (index, (&actual, &expected)) in
            actual_density.iter().zip(&expected_density).enumerate()
        {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(format!(
                    "layer1 hidden VJP density mismatch at {index}: actual={actual} expected={expected}"
                ));
            }
        }
        Ok(())
    }
}
