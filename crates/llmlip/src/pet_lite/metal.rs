//! Apple Metal kernels for pinned PET FP32 runtime paths.
//!
//! This module is intentionally narrow. The first PET target is the
//! compact-projected attention core used by PET-MAD-XS/S prepared f32 paths.
#![allow(dead_code, unsafe_code)]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

const PET_METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PetAttentionForwardParams {
    uint n_atoms;
    uint active_len;
    uint max_edges_per_atom;
    uint d_edge;
    uint num_heads;
    uint head_dim;
    uint qkv_token_stride;
    float scale;
    float attention_eps;
};

kernel void pet_attention_forward_compact_projected_f32(
    device const float* projected [[buffer(0)]],
    device const uint* active_tokens [[buffer(1)]],
    device const float* cutoff_factors [[buffer(2)]],
    device float* attention [[buffer(3)]],
    device float* probabilities [[buffer(4)]],
    constant PetAttentionForwardParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows_per_atom = params.num_heads * params.active_len;
    const uint atom = gid / rows_per_atom;
    if (atom >= params.n_atoms) {
        return;
    }
    const uint local = gid - atom * rows_per_atom;
    const uint head = local / params.active_len;
    const uint query = local - head * params.active_len;
    const uint active_len = params.active_len;
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint head_offset = head * head_dim;
    const uint projected_atom_base = atom * active_len * qkv_token_stride;
    const uint active_atom_base = atom * active_len;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;

    float scores[64];
    float max_score = -INFINITY;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        const uint key_token = active_tokens[active_atom_base + key_index];
        const uint k_base = projected_atom_base + key_index * qkv_token_stride + d_edge + head_offset;
        float dot = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            dot += projected[q_base + dim] * projected[k_base + dim];
        }
        float key_factor = 0.0f;
        if (key_token > 0) {
            const float cutoff = cutoff_factors[cutoff_atom_base + key_token - 1];
            key_factor = log(max(cutoff, params.attention_eps));
        }
        const float score = dot * params.scale + key_factor;
        scores[key_index] = score;
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        const float value = exp(scores[key_index] - max_score);
        scores[key_index] = value;
        denom += value;
    }
    const uint probability_base =
        atom * params.num_heads * active_len * active_len + head * active_len * active_len + query * active_len;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        scores[key_index] = scores[key_index] / denom;
        probabilities[probability_base + key_index] = scores[key_index];
    }

    const uint attention_base = atom * active_len * d_edge + query * d_edge + head_offset;
    for (uint dim = 0; dim < head_dim; ++dim) {
        float value = 0.0f;
        for (uint key_index = 0; key_index < active_len; ++key_index) {
            const uint v_base = projected_atom_base + key_index * qkv_token_stride + 2 * d_edge + head_offset;
            value += scores[key_index] * projected[v_base + dim];
        }
        attention[attention_base + dim] = value;
    }
}

kernel void pet_attention_forward_ragged_compact_projected_f32(
    device const float* projected [[buffer(0)]],
    device const uint* active_tokens [[buffer(1)]],
    device const uint* active_token_offsets [[buffer(2)]],
    device const uint* compact_probability_offsets [[buffer(3)]],
    device const float* cutoff_factors [[buffer(4)]],
    device float* attention [[buffer(5)]],
    device float* probabilities [[buffer(6)]],
    constant PetAttentionForwardParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows_per_atom = params.num_heads * params.active_len;
    const uint atom = gid / rows_per_atom;
    if (atom >= params.n_atoms) {
        return;
    }
    const uint local = gid - atom * rows_per_atom;
    const uint head = local / params.active_len;
    const uint query = local - head * params.active_len;
    const uint active_start = active_token_offsets[atom];
    const uint active_end = active_token_offsets[atom + 1];
    const uint active_len = active_end - active_start;
    if (query >= active_len) {
        return;
    }
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint head_offset = head * head_dim;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint q_base = (active_start + query) * qkv_token_stride + head_offset;

    float scores[64];
    float max_score = -INFINITY;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        const uint key_token = active_tokens[active_start + key_index];
        const uint k_base = (active_start + key_index) * qkv_token_stride + d_edge + head_offset;
        float dot = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            dot += projected[q_base + dim] * projected[k_base + dim];
        }
        float key_factor = 0.0f;
        if (key_token > 0) {
            const float cutoff = cutoff_factors[cutoff_atom_base + key_token - 1];
            key_factor = log(max(cutoff, params.attention_eps));
        }
        const float score = dot * params.scale + key_factor;
        scores[key_index] = score;
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        const float value = exp(scores[key_index] - max_score);
        scores[key_index] = value;
        denom += value;
    }
    const uint probability_base =
        compact_probability_offsets[atom] + head * active_len * active_len + query * active_len;
    for (uint key_index = 0; key_index < active_len; ++key_index) {
        scores[key_index] = scores[key_index] / denom;
        probabilities[probability_base + key_index] = scores[key_index];
    }

    const uint attention_base = (active_start + query) * d_edge + head_offset;
    for (uint dim = 0; dim < head_dim; ++dim) {
        float value = 0.0f;
        for (uint key_index = 0; key_index < active_len; ++key_index) {
            const uint v_base = (active_start + key_index) * qkv_token_stride + 2 * d_edge + head_offset;
            value += scores[key_index] * projected[v_base + dim];
        }
        attention[attention_base + dim] = value;
    }
}

kernel void pet_attention_forward_full_projected_f32(
    device const float* projected [[buffer(0)]],
    device const float* cutoff_factors [[buffer(1)]],
    device float* attention [[buffer(2)]],
    device float* probabilities [[buffer(3)]],
    constant PetAttentionForwardParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows_per_atom = params.num_heads * params.active_len;
    const uint atom = gid / rows_per_atom;
    if (atom >= params.n_atoms) {
        return;
    }
    const uint local = gid - atom * rows_per_atom;
    const uint head = local / params.active_len;
    const uint query = local - head * params.active_len;
    const uint seq_len = params.active_len;
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint head_offset = head * head_dim;
    const uint projected_atom_base = atom * seq_len * qkv_token_stride;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;

    float scores[64];
    float max_score = -INFINITY;
    for (uint key = 0; key < seq_len; ++key) {
        const uint k_base = projected_atom_base + key * qkv_token_stride + d_edge + head_offset;
        float dot = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            dot += projected[q_base + dim] * projected[k_base + dim];
        }
        float key_factor = 0.0f;
        if (key > 0) {
            const float cutoff = cutoff_factors[cutoff_atom_base + key - 1];
            key_factor = log(max(cutoff, params.attention_eps));
        }
        const float score = dot * params.scale + key_factor;
        scores[key] = score;
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    for (uint key = 0; key < seq_len; ++key) {
        const float value = exp(scores[key] - max_score);
        scores[key] = value;
        denom += value;
    }
    const uint probability_base =
        atom * params.num_heads * seq_len * seq_len + head * seq_len * seq_len + query * seq_len;
    for (uint key = 0; key < seq_len; ++key) {
        scores[key] = scores[key] / denom;
        probabilities[probability_base + key] = scores[key];
    }

    const uint attention_base = atom * seq_len * d_edge + query * d_edge + head_offset;
    for (uint dim = 0; dim < head_dim; ++dim) {
        float value = 0.0f;
        for (uint key = 0; key < seq_len; ++key) {
            const uint v_base = projected_atom_base + key * qkv_token_stride + 2 * d_edge + head_offset;
            value += scores[key] * projected[v_base + dim];
        }
        attention[attention_base + dim] = value;
    }
}

kernel void pet_attention_vjp_compact_projected_f32(
    device const float* projected [[buffer(0)]],
    device const uint* active_tokens [[buffer(1)]],
    device const float* cutoff_factors [[buffer(2)]],
    device const float* probabilities [[buffer(3)]],
    device const float* bar_attention [[buffer(4)]],
    device float* bar_projected [[buffer(5)]],
    device float* bar_cutoff [[buffer(6)]],
    constant PetAttentionForwardParams& params [[buffer(7)]],
    uint atom [[thread_position_in_grid]]
) {
    if (atom >= params.n_atoms) {
        return;
    }
    const uint active_len = params.active_len;
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint projected_atom_base = atom * active_len * qkv_token_stride;
    const uint active_atom_base = atom * active_len;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint probability_atom_base = atom * params.num_heads * active_len * active_len;
    const uint attention_atom_base = atom * active_len * d_edge;

    for (uint index = 0; index < active_len * qkv_token_stride; ++index) {
        bar_projected[projected_atom_base + index] = 0.0f;
    }
    for (uint edge = 0; edge < params.max_edges_per_atom; ++edge) {
        bar_cutoff[cutoff_atom_base + edge] = 0.0f;
    }

    float bar_probability[64];
    float bar_scores[64];
    for (uint head = 0; head < params.num_heads; ++head) {
        const uint head_offset = head * head_dim;
        const uint probability_head_base = probability_atom_base + head * active_len * active_len;
        for (uint query = 0; query < active_len; ++query) {
            const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;
            const uint probability_row_base = probability_head_base + query * active_len;
            const uint bar_value_base = attention_atom_base + query * d_edge + head_offset;

            for (uint key_index = 0; key_index < active_len; ++key_index) {
                const float probability = probabilities[probability_row_base + key_index];
                const uint v_base =
                    projected_atom_base + key_index * qkv_token_stride + 2 * d_edge + head_offset;
                float dot = 0.0f;
                for (uint dim = 0; dim < head_dim; ++dim) {
                    const float bar_value = bar_attention[bar_value_base + dim];
                    dot += bar_value * projected[v_base + dim];
                    bar_projected[v_base + dim] += bar_value * probability;
                }
                bar_probability[key_index] = dot;
            }

            float probability_bar_dot = 0.0f;
            for (uint key_index = 0; key_index < active_len; ++key_index) {
                probability_bar_dot +=
                    probabilities[probability_row_base + key_index] * bar_probability[key_index];
            }
            for (uint key_index = 0; key_index < active_len; ++key_index) {
                const float probability = probabilities[probability_row_base + key_index];
                bar_scores[key_index] =
                    probability * (bar_probability[key_index] - probability_bar_dot);
            }

            for (uint key_index = 0; key_index < active_len; ++key_index) {
                const float bar_score = bar_scores[key_index];
                const uint k_base =
                    projected_atom_base + key_index * qkv_token_stride + d_edge + head_offset;
                for (uint dim = 0; dim < head_dim; ++dim) {
                    bar_projected[q_base + dim] +=
                        bar_score * params.scale * projected[k_base + dim];
                    bar_projected[k_base + dim] +=
                        bar_score * params.scale * projected[q_base + dim];
                }
                const uint key_token = active_tokens[active_atom_base + key_index];
                if (key_token > 0) {
                    const uint edge = key_token - 1;
                    const float cutoff = cutoff_factors[cutoff_atom_base + edge];
                    if (cutoff > params.attention_eps) {
                        bar_cutoff[cutoff_atom_base + edge] += bar_score / cutoff;
                    }
                }
            }
        }
    }
}

	kernel void pet_attention_vjp_full_projected_f32(
	    device const float* projected [[buffer(0)]],
	    device const float* cutoff_factors [[buffer(1)]],
	    device const float* probabilities [[buffer(2)]],
    device const float* bar_attention [[buffer(3)]],
    device float* bar_projected [[buffer(4)]],
    device float* bar_cutoff [[buffer(5)]],
    constant PetAttentionForwardParams& params [[buffer(6)]],
    uint atom [[thread_position_in_grid]]
) {
    if (atom >= params.n_atoms) {
        return;
    }
    const uint seq_len = params.active_len;
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint projected_atom_base = atom * seq_len * qkv_token_stride;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint probability_atom_base = atom * params.num_heads * seq_len * seq_len;
    const uint attention_atom_base = atom * seq_len * d_edge;

    for (uint index = 0; index < seq_len * qkv_token_stride; ++index) {
        bar_projected[projected_atom_base + index] = 0.0f;
    }
    for (uint edge = 0; edge < params.max_edges_per_atom; ++edge) {
        bar_cutoff[cutoff_atom_base + edge] = 0.0f;
    }

    float bar_probability[64];
    float bar_scores[64];
    for (uint head = 0; head < params.num_heads; ++head) {
        const uint head_offset = head * head_dim;
        const uint probability_head_base = probability_atom_base + head * seq_len * seq_len;
        for (uint query = 0; query < seq_len; ++query) {
            const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;
            const uint probability_row_base = probability_head_base + query * seq_len;
            const uint bar_value_base = attention_atom_base + query * d_edge + head_offset;

            for (uint key = 0; key < seq_len; ++key) {
                const float probability = probabilities[probability_row_base + key];
                const uint v_base =
                    projected_atom_base + key * qkv_token_stride + 2 * d_edge + head_offset;
                float dot = 0.0f;
                for (uint dim = 0; dim < head_dim; ++dim) {
                    const float bar_value = bar_attention[bar_value_base + dim];
                    dot += bar_value * projected[v_base + dim];
                    bar_projected[v_base + dim] += bar_value * probability;
                }
                bar_probability[key] = dot;
            }

            float probability_bar_dot = 0.0f;
            for (uint key = 0; key < seq_len; ++key) {
                probability_bar_dot +=
                    probabilities[probability_row_base + key] * bar_probability[key];
            }
            for (uint key = 0; key < seq_len; ++key) {
                const float probability = probabilities[probability_row_base + key];
                bar_scores[key] = probability * (bar_probability[key] - probability_bar_dot);
            }

            for (uint key = 0; key < seq_len; ++key) {
                const float bar_score = bar_scores[key];
                const uint k_base =
                    projected_atom_base + key * qkv_token_stride + d_edge + head_offset;
                for (uint dim = 0; dim < head_dim; ++dim) {
                    bar_projected[q_base + dim] +=
                        bar_score * params.scale * projected[k_base + dim];
                    bar_projected[k_base + dim] +=
                        bar_score * params.scale * projected[q_base + dim];
                }
                if (key > 0) {
                    const uint edge = key - 1;
                    const float cutoff = cutoff_factors[cutoff_atom_base + edge];
                    if (cutoff > params.attention_eps) {
                        bar_cutoff[cutoff_atom_base + edge] += bar_score / cutoff;
                    }
                }
            }
        }
	    }
	}

	kernel void pet_attention_vjp_full_projected_split_bar_f32(
	    device const float* projected [[buffer(0)]],
	    device const float* cutoff_factors [[buffer(1)]],
	    device const float* probabilities [[buffer(2)]],
	    device const float* bar_center_attention [[buffer(3)]],
	    device const float* bar_edge_attention [[buffer(4)]],
	    device float* bar_projected [[buffer(5)]],
	    device float* bar_cutoff [[buffer(6)]],
	    constant PetAttentionForwardParams& params [[buffer(7)]],
	    uint atom [[thread_position_in_grid]]
	) {
	    if (atom >= params.n_atoms) {
	        return;
	    }
	    const uint seq_len = params.active_len;
	    const uint head_dim = params.head_dim;
	    const uint d_edge = params.d_edge;
	    const uint qkv_token_stride = params.qkv_token_stride;
	    const uint projected_atom_base = atom * seq_len * qkv_token_stride;
	    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
	    const uint probability_atom_base = atom * params.num_heads * seq_len * seq_len;
	    const uint center_bar_atom_base = atom * d_edge;
	    const uint edge_bar_atom_base = atom * params.max_edges_per_atom * d_edge;

	    for (uint index = 0; index < seq_len * qkv_token_stride; ++index) {
	        bar_projected[projected_atom_base + index] = 0.0f;
	    }
	    for (uint edge = 0; edge < params.max_edges_per_atom; ++edge) {
	        bar_cutoff[cutoff_atom_base + edge] = 0.0f;
	    }

	    float bar_probability[64];
	    float bar_scores[64];
	    for (uint head = 0; head < params.num_heads; ++head) {
	        const uint head_offset = head * head_dim;
	        const uint probability_head_base = probability_atom_base + head * seq_len * seq_len;
	        for (uint query = 0; query < seq_len; ++query) {
	            const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;
	            const uint probability_row_base = probability_head_base + query * seq_len;

	            for (uint key = 0; key < seq_len; ++key) {
	                const float probability = probabilities[probability_row_base + key];
	                const uint v_base =
	                    projected_atom_base + key * qkv_token_stride + 2 * d_edge + head_offset;
	                float dot = 0.0f;
	                for (uint dim = 0; dim < head_dim; ++dim) {
	                    const float bar_value = query == 0
	                        ? bar_center_attention[center_bar_atom_base + head_offset + dim]
	                        : bar_edge_attention[
	                              edge_bar_atom_base + (query - 1) * d_edge + head_offset + dim
	                          ];
	                    dot += bar_value * projected[v_base + dim];
	                    bar_projected[v_base + dim] += bar_value * probability;
	                }
	                bar_probability[key] = dot;
	            }

	            float probability_bar_dot = 0.0f;
	            for (uint key = 0; key < seq_len; ++key) {
	                probability_bar_dot +=
	                    probabilities[probability_row_base + key] * bar_probability[key];
	            }
	            for (uint key = 0; key < seq_len; ++key) {
	                const float probability = probabilities[probability_row_base + key];
	                bar_scores[key] = probability * (bar_probability[key] - probability_bar_dot);
	            }

	            for (uint key = 0; key < seq_len; ++key) {
	                const float bar_score = bar_scores[key];
	                const uint k_base =
	                    projected_atom_base + key * qkv_token_stride + d_edge + head_offset;
	                for (uint dim = 0; dim < head_dim; ++dim) {
	                    bar_projected[q_base + dim] +=
	                        bar_score * params.scale * projected[k_base + dim];
	                    bar_projected[k_base + dim] +=
	                        bar_score * params.scale * projected[q_base + dim];
	                }
	                if (key > 0) {
	                    const uint edge = key - 1;
	                    const float cutoff = cutoff_factors[cutoff_atom_base + edge];
	                    if (cutoff > params.attention_eps) {
	                        bar_cutoff[cutoff_atom_base + edge] += bar_score / cutoff;
	                    }
	                }
	            }
	        }
	    }
	}

	kernel void pet_attention_vjp_full_projected_head_f32(
	    device const float* projected [[buffer(0)]],
	    device const float* cutoff_factors [[buffer(1)]],
    device const float* probabilities [[buffer(2)]],
    device const float* bar_attention [[buffer(3)]],
    device float* bar_projected [[buffer(4)]],
    device float* bar_cutoff_heads [[buffer(5)]],
    constant PetAttentionForwardParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint atom = gid / params.num_heads;
    if (atom >= params.n_atoms) {
        return;
    }
    const uint head = gid - atom * params.num_heads;
    const uint seq_len = params.active_len;
    const uint head_dim = params.head_dim;
    const uint d_edge = params.d_edge;
    const uint qkv_token_stride = params.qkv_token_stride;
    const uint projected_atom_base = atom * seq_len * qkv_token_stride;
    const uint cutoff_atom_base = atom * params.max_edges_per_atom;
    const uint probability_atom_base = atom * params.num_heads * seq_len * seq_len;
    const uint attention_atom_base = atom * seq_len * d_edge;
    const uint head_offset = head * head_dim;
    const uint cutoff_head_base =
        (atom * params.num_heads + head) * params.max_edges_per_atom;

    for (uint token = 0; token < seq_len; ++token) {
        const uint q_base = projected_atom_base + token * qkv_token_stride + head_offset;
        const uint k_base = projected_atom_base + token * qkv_token_stride + d_edge + head_offset;
        const uint v_base = projected_atom_base + token * qkv_token_stride + 2 * d_edge + head_offset;
        for (uint dim = 0; dim < head_dim; ++dim) {
            bar_projected[q_base + dim] = 0.0f;
            bar_projected[k_base + dim] = 0.0f;
            bar_projected[v_base + dim] = 0.0f;
        }
    }
    for (uint edge = 0; edge < params.max_edges_per_atom; ++edge) {
        bar_cutoff_heads[cutoff_head_base + edge] = 0.0f;
    }

    float bar_probability[64];
    float bar_scores[64];
    const uint probability_head_base = probability_atom_base + head * seq_len * seq_len;
    for (uint query = 0; query < seq_len; ++query) {
        const uint q_base = projected_atom_base + query * qkv_token_stride + head_offset;
        const uint probability_row_base = probability_head_base + query * seq_len;
        const uint bar_value_base = attention_atom_base + query * d_edge + head_offset;

        for (uint key = 0; key < seq_len; ++key) {
            const float probability = probabilities[probability_row_base + key];
            const uint v_base =
                projected_atom_base + key * qkv_token_stride + 2 * d_edge + head_offset;
            float dot = 0.0f;
            for (uint dim = 0; dim < head_dim; ++dim) {
                const float bar_value = bar_attention[bar_value_base + dim];
                dot += bar_value * projected[v_base + dim];
                bar_projected[v_base + dim] += bar_value * probability;
            }
            bar_probability[key] = dot;
        }

        float probability_bar_dot = 0.0f;
        for (uint key = 0; key < seq_len; ++key) {
            probability_bar_dot +=
                probabilities[probability_row_base + key] * bar_probability[key];
        }
        for (uint key = 0; key < seq_len; ++key) {
            const float probability = probabilities[probability_row_base + key];
            bar_scores[key] = probability * (bar_probability[key] - probability_bar_dot);
        }

        for (uint key = 0; key < seq_len; ++key) {
            const float bar_score = bar_scores[key];
            const uint k_base =
                projected_atom_base + key * qkv_token_stride + d_edge + head_offset;
            for (uint dim = 0; dim < head_dim; ++dim) {
                bar_projected[q_base + dim] +=
                    bar_score * params.scale * projected[k_base + dim];
                bar_projected[k_base + dim] +=
                    bar_score * params.scale * projected[q_base + dim];
            }
            if (key > 0) {
                const uint edge = key - 1;
                const float cutoff = cutoff_factors[cutoff_atom_base + edge];
                if (cutoff > params.attention_eps) {
                    bar_cutoff_heads[cutoff_head_base + edge] += bar_score / cutoff;
                }
            }
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct PetAttentionForwardParams {
    n_atoms: u32,
    active_len: u32,
    max_edges_per_atom: u32,
    d_edge: u32,
    num_heads: u32,
    head_dim: u32,
    qkv_token_stride: u32,
    scale: f32,
    attention_eps: f32,
}

struct PetMetalBackend {
    device: Device,
    queue: CommandQueue,
    attention_forward_pipeline: ComputePipelineState,
    attention_ragged_forward_pipeline: ComputePipelineState,
    attention_full_forward_pipeline: ComputePipelineState,
    attention_vjp_pipeline: ComputePipelineState,
    attention_full_vjp_pipeline: ComputePipelineState,
    attention_full_vjp_split_bar_pipeline: ComputePipelineState,
    attention_full_vjp_head_pipeline: ComputePipelineState,
}

impl PetMetalBackend {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "no Metal device found".to_owned())?;
        let library = device
            .new_library_with_source(PET_METAL_SOURCE, &CompileOptions::new())
            .map_err(|err| format!("compile PET Metal library: {err}"))?;
        let forward_function = library
            .get_function("pet_attention_forward_compact_projected_f32", None)
            .map_err(|err| format!("load pet_attention_forward_compact_projected_f32: {err}"))?;
        let attention_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_function)
            .map_err(|err| {
                format!("build pet_attention_forward_compact_projected_f32 pipeline: {err}")
            })?;
        let ragged_forward_function = library
            .get_function("pet_attention_forward_ragged_compact_projected_f32", None)
            .map_err(|err| {
                format!("load pet_attention_forward_ragged_compact_projected_f32: {err}")
            })?;
        let attention_ragged_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&ragged_forward_function)
            .map_err(|err| {
                format!("build pet_attention_forward_ragged_compact_projected_f32 pipeline: {err}")
            })?;
        let full_forward_function = library
            .get_function("pet_attention_forward_full_projected_f32", None)
            .map_err(|err| format!("load pet_attention_forward_full_projected_f32: {err}"))?;
        let attention_full_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&full_forward_function)
            .map_err(|err| {
                format!("build pet_attention_forward_full_projected_f32 pipeline: {err}")
            })?;
        let vjp_function = library
            .get_function("pet_attention_vjp_compact_projected_f32", None)
            .map_err(|err| format!("load pet_attention_vjp_compact_projected_f32: {err}"))?;
        let attention_vjp_pipeline = device
            .new_compute_pipeline_state_with_function(&vjp_function)
            .map_err(|err| {
                format!("build pet_attention_vjp_compact_projected_f32 pipeline: {err}")
            })?;
        let full_vjp_function = library
            .get_function("pet_attention_vjp_full_projected_f32", None)
            .map_err(|err| format!("load pet_attention_vjp_full_projected_f32: {err}"))?;
        let attention_full_vjp_pipeline = device
            .new_compute_pipeline_state_with_function(&full_vjp_function)
            .map_err(|err| format!("build pet_attention_vjp_full_projected_f32 pipeline: {err}"))?;
        let full_vjp_split_bar_function = library
            .get_function("pet_attention_vjp_full_projected_split_bar_f32", None)
            .map_err(|err| format!("load pet_attention_vjp_full_projected_split_bar_f32: {err}"))?;
        let attention_full_vjp_split_bar_pipeline = device
            .new_compute_pipeline_state_with_function(&full_vjp_split_bar_function)
            .map_err(|err| {
                format!("build pet_attention_vjp_full_projected_split_bar_f32 pipeline: {err}")
            })?;
        let full_vjp_head_function = library
            .get_function("pet_attention_vjp_full_projected_head_f32", None)
            .map_err(|err| format!("load pet_attention_vjp_full_projected_head_f32: {err}"))?;
        let attention_full_vjp_head_pipeline = device
            .new_compute_pipeline_state_with_function(&full_vjp_head_function)
            .map_err(|err| {
                format!("build pet_attention_vjp_full_projected_head_f32 pipeline: {err}")
            })?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            attention_forward_pipeline,
            attention_ragged_forward_pipeline,
            attention_full_forward_pipeline,
            attention_vjp_pipeline,
            attention_full_vjp_pipeline,
            attention_full_vjp_split_bar_pipeline,
            attention_full_vjp_head_pipeline,
        })
    }

    fn buffer_with_slice<T>(&self, values: &[T], options: MTLResourceOptions) -> Buffer {
        self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(values) as u64,
            options,
        )
    }

    fn input_buffer_with_slice<T>(&self, values: &[T], options: MTLResourceOptions) -> Buffer {
        if pet_metal_no_copy_input_enabled() {
            return self.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(values) as u64,
                options,
                None,
            );
        }
        self.buffer_with_slice(values, options)
    }

    fn output_buffer_with_mut_slice<T>(
        &self,
        values: &mut [T],
        options: MTLResourceOptions,
    ) -> (Buffer, bool) {
        if pet_metal_no_copy_input_enabled() {
            return (
                self.device.new_buffer_with_bytes_no_copy(
                    values.as_mut_ptr().cast::<c_void>(),
                    std::mem::size_of_val(values) as u64,
                    options,
                    None,
                ),
                true,
            );
        }
        (
            self.device
                .new_buffer(std::mem::size_of_val(values) as u64, options),
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward_compact_projected_f32(
        &self,
        projected: &[f32],
        active_tokens: &[u32],
        cutoff_factors: &[f32],
        n_atoms: usize,
        active_len: usize,
        max_edges_per_atom: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        attention: &mut [f32],
        probabilities: &mut [f32],
    ) -> Result<(), String> {
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(active_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        let active_token_len = n_atoms
            .checked_mul(active_len)
            .ok_or_else(|| "PET Metal active token length overflows".to_owned())?;
        if active_tokens.len() != active_token_len {
            return Err(format!(
                "PET Metal active token length must be {active_token_len}, got {}",
                active_tokens.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        let attention_len = n_atoms
            .checked_mul(active_len)
            .and_then(|rows| rows.checked_mul(d_edge))
            .ok_or_else(|| "PET Metal attention length overflows".to_owned())?;
        if attention.len() != attention_len {
            return Err(format!(
                "PET Metal attention length must be {attention_len}, got {}",
                attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(active_len))
            .and_then(|values| values.checked_mul(active_len))
            .ok_or_else(|| "PET Metal probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if active_len > 64 {
            return Err("PET Metal active length exceeds fixed score capacity".to_owned());
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal attention dimensions are inconsistent".to_owned());
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(active_len)
                .map_err(|_| "PET Metal active length exceeds u32 range".to_owned())?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom)
                .map_err(|_| "PET Metal max edge count exceeds u32 range".to_owned())?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| "PET Metal head dimension exceeds u32 range".to_owned())?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.buffer_with_slice(projected, options);
        let active_tokens_buffer = self.buffer_with_slice(active_tokens, options);
        let cutoff_buffer = self.buffer_with_slice(cutoff_factors, options);
        let (attention_buffer, attention_no_copy) =
            self.output_buffer_with_mut_slice(attention, options);
        let (probabilities_buffer, probabilities_no_copy) =
            self.output_buffer_with_mut_slice(probabilities, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_forward_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&active_tokens_buffer), 0);
        encoder.set_buffer(2, Some(&cutoff_buffer), 0);
        encoder.set_buffer(3, Some(&attention_buffer), 0);
        encoder.set_buffer(4, Some(&probabilities_buffer), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);

        let total_threads = (n_atoms * num_heads * active_len) as u64;
        let execution_width = self
            .attention_forward_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_forward_pipeline
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
            return Err(format!(
                "PET Metal command buffer did not complete: {status:?}"
            ));
        }

        if !attention_no_copy {
            let attention_ptr = attention_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    attention_ptr,
                    attention.as_mut_ptr(),
                    attention.len(),
                );
            }
        }
        if !probabilities_no_copy {
            let probabilities_ptr = probabilities_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    probabilities_ptr,
                    probabilities.as_mut_ptr(),
                    probabilities.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward_ragged_compact_projected_f32(
        &self,
        projected: &[f32],
        active_tokens: &[u32],
        active_token_offsets: &[u32],
        compact_probability_offsets: &[u32],
        cutoff_factors: &[f32],
        n_atoms: usize,
        max_active_len: usize,
        max_edges_per_atom: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        attention: &mut [f32],
        probabilities: &mut [f32],
    ) -> Result<(), String> {
        let qkv_token_stride = 3 * d_edge;
        if active_token_offsets.len() != n_atoms + 1 {
            return Err(format!(
                "PET Metal ragged active token offsets length must be {}, got {}",
                n_atoms + 1,
                active_token_offsets.len()
            ));
        }
        if compact_probability_offsets.len() != n_atoms + 1 {
            return Err(format!(
                "PET Metal ragged probability offsets length must be {}, got {}",
                n_atoms + 1,
                compact_probability_offsets.len()
            ));
        }
        let total_active_tokens = active_token_offsets
            .last()
            .copied()
            .ok_or_else(|| "PET Metal ragged active token offsets are empty".to_owned())?
            as usize;
        if active_tokens.len() != total_active_tokens {
            return Err(format!(
                "PET Metal ragged active token length must be {total_active_tokens}, got {}",
                active_tokens.len()
            ));
        }
        let projected_len = total_active_tokens
            .checked_mul(qkv_token_stride)
            .ok_or_else(|| "PET Metal ragged projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal ragged projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal ragged cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal ragged cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        let attention_len = total_active_tokens
            .checked_mul(d_edge)
            .ok_or_else(|| "PET Metal ragged attention length overflows".to_owned())?;
        if attention.len() != attention_len {
            return Err(format!(
                "PET Metal ragged attention length must be {attention_len}, got {}",
                attention.len()
            ));
        }
        let probability_len = compact_probability_offsets
            .last()
            .copied()
            .ok_or_else(|| "PET Metal ragged probability offsets are empty".to_owned())?
            as usize;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal ragged probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if max_active_len > 64 {
            return Err("PET Metal ragged active length exceeds fixed score capacity".to_owned());
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal ragged attention dimensions are inconsistent".to_owned());
        }
        for atom in 0..n_atoms {
            let start = active_token_offsets[atom] as usize;
            let end = active_token_offsets[atom + 1] as usize;
            if end < start {
                return Err(format!(
                    "PET Metal ragged active token offsets must be nondecreasing at atom {atom}"
                ));
            }
            let active_len = end - start;
            if active_len == 0 {
                return Err(format!(
                    "PET Metal ragged active token length for atom {atom} must be nonzero"
                ));
            }
            if active_len > max_active_len {
                return Err(format!(
                    "PET Metal ragged active token length for atom {atom} exceeds max_active_len {max_active_len}"
                ));
            }
            let probability_start = compact_probability_offsets[atom] as usize;
            let probability_end = compact_probability_offsets[atom + 1] as usize;
            let expected_probability_end = probability_start
                .checked_add(
                    num_heads
                        .checked_mul(active_len)
                        .and_then(|values| values.checked_mul(active_len))
                        .ok_or_else(|| {
                            "PET Metal ragged probability atom length overflows".to_owned()
                        })?,
                )
                .ok_or_else(|| "PET Metal ragged probability offset overflows".to_owned())?;
            if probability_end != expected_probability_end {
                return Err(format!(
                    "PET Metal ragged probability offsets for atom {atom} must end at {expected_probability_end}, got {probability_end}"
                ));
            }
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(max_active_len)
                .map_err(|_| "PET Metal max active length exceeds u32 range".to_owned())?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom)
                .map_err(|_| "PET Metal max edge count exceeds u32 range".to_owned())?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| "PET Metal head dimension exceeds u32 range".to_owned())?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.buffer_with_slice(projected, options);
        let active_tokens_buffer = self.buffer_with_slice(active_tokens, options);
        let active_token_offsets_buffer = self.buffer_with_slice(active_token_offsets, options);
        let compact_probability_offsets_buffer =
            self.buffer_with_slice(compact_probability_offsets, options);
        let cutoff_buffer = self.buffer_with_slice(cutoff_factors, options);
        let (attention_buffer, attention_no_copy) =
            self.output_buffer_with_mut_slice(attention, options);
        let (probabilities_buffer, probabilities_no_copy) =
            self.output_buffer_with_mut_slice(probabilities, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_ragged_forward_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&active_tokens_buffer), 0);
        encoder.set_buffer(2, Some(&active_token_offsets_buffer), 0);
        encoder.set_buffer(3, Some(&compact_probability_offsets_buffer), 0);
        encoder.set_buffer(4, Some(&cutoff_buffer), 0);
        encoder.set_buffer(5, Some(&attention_buffer), 0);
        encoder.set_buffer(6, Some(&probabilities_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);

        let total_threads = (n_atoms * num_heads * max_active_len) as u64;
        let execution_width = self
            .attention_ragged_forward_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_ragged_forward_pipeline
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
            return Err(format!(
                "PET Metal ragged command buffer did not complete: {status:?}"
            ));
        }

        if !attention_no_copy {
            let attention_ptr = attention_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    attention_ptr,
                    attention.as_mut_ptr(),
                    attention.len(),
                );
            }
        }
        if !probabilities_no_copy {
            let probabilities_ptr = probabilities_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    probabilities_ptr,
                    probabilities.as_mut_ptr(),
                    probabilities.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_vjp_compact_projected_f32(
        &self,
        projected: &[f32],
        active_tokens: &[u32],
        cutoff_factors: &[f32],
        probabilities: &[f32],
        bar_attention: &[f32],
        n_atoms: usize,
        active_len: usize,
        max_edges_per_atom: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        bar_projected: &mut [f32],
        bar_cutoff: &mut [f32],
    ) -> Result<(), String> {
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(active_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal VJP projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal VJP projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        if bar_projected.len() != projected_len {
            return Err(format!(
                "PET Metal VJP bar_projected length must be {projected_len}, got {}",
                bar_projected.len()
            ));
        }
        let active_token_len = n_atoms
            .checked_mul(active_len)
            .ok_or_else(|| "PET Metal VJP active token length overflows".to_owned())?;
        if active_tokens.len() != active_token_len {
            return Err(format!(
                "PET Metal VJP active token length must be {active_token_len}, got {}",
                active_tokens.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal VJP cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal VJP cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        if bar_cutoff.len() != cutoff_len {
            return Err(format!(
                "PET Metal VJP bar_cutoff length must be {cutoff_len}, got {}",
                bar_cutoff.len()
            ));
        }
        let attention_len = n_atoms
            .checked_mul(active_len)
            .and_then(|rows| rows.checked_mul(d_edge))
            .ok_or_else(|| "PET Metal VJP bar_attention length overflows".to_owned())?;
        if bar_attention.len() != attention_len {
            return Err(format!(
                "PET Metal VJP bar_attention length must be {attention_len}, got {}",
                bar_attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(active_len))
            .and_then(|values| values.checked_mul(active_len))
            .ok_or_else(|| "PET Metal VJP probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal VJP probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if active_len > 64 {
            return Err("PET Metal VJP active length exceeds fixed score capacity".to_owned());
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal VJP attention dimensions are inconsistent".to_owned());
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal VJP atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(active_len)
                .map_err(|_| "PET Metal VJP active length exceeds u32 range".to_owned())?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom)
                .map_err(|_| "PET Metal VJP max edge count exceeds u32 range".to_owned())?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal VJP d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal VJP head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| "PET Metal VJP head dimension exceeds u32 range".to_owned())?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal VJP qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.buffer_with_slice(projected, options);
        let active_tokens_buffer = self.buffer_with_slice(active_tokens, options);
        let cutoff_buffer = self.buffer_with_slice(cutoff_factors, options);
        let probabilities_buffer = self.buffer_with_slice(probabilities, options);
        let bar_attention_buffer = self.buffer_with_slice(bar_attention, options);
        let (bar_projected_buffer, bar_projected_no_copy) =
            self.output_buffer_with_mut_slice(bar_projected, options);
        let (bar_cutoff_buffer, bar_cutoff_no_copy) =
            self.output_buffer_with_mut_slice(bar_cutoff, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_vjp_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&active_tokens_buffer), 0);
        encoder.set_buffer(2, Some(&cutoff_buffer), 0);
        encoder.set_buffer(3, Some(&probabilities_buffer), 0);
        encoder.set_buffer(4, Some(&bar_attention_buffer), 0);
        encoder.set_buffer(5, Some(&bar_projected_buffer), 0);
        encoder.set_buffer(6, Some(&bar_cutoff_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);

        let total_threads = n_atoms as u64;
        let execution_width = self.attention_vjp_pipeline.thread_execution_width().max(1);
        let max_threads = self
            .attention_vjp_pipeline
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
            return Err(format!(
                "PET Metal VJP command buffer did not complete: {status:?}"
            ));
        }

        if !bar_projected_no_copy {
            let bar_projected_ptr = bar_projected_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_projected_ptr,
                    bar_projected.as_mut_ptr(),
                    bar_projected.len(),
                );
            }
        }
        if !bar_cutoff_no_copy {
            let bar_cutoff_ptr = bar_cutoff_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_cutoff_ptr,
                    bar_cutoff.as_mut_ptr(),
                    bar_cutoff.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward_full_projected_f32(
        &self,
        projected: &[f32],
        cutoff_factors: &[f32],
        n_atoms: usize,
        seq_len: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        attention: &mut [f32],
        probabilities: &mut [f32],
    ) -> Result<(), String> {
        let max_edges_per_atom = seq_len
            .checked_sub(1)
            .ok_or_else(|| "PET Metal full forward sequence length must be nonzero".to_owned())?;
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal full forward projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal full forward projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal full forward cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal full forward cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        let attention_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(d_edge))
            .ok_or_else(|| "PET Metal full forward attention length overflows".to_owned())?;
        if attention.len() != attention_len {
            return Err(format!(
                "PET Metal full forward attention length must be {attention_len}, got {}",
                attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(seq_len))
            .and_then(|values| values.checked_mul(seq_len))
            .ok_or_else(|| "PET Metal full forward probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal full forward probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if seq_len > 64 {
            return Err(
                "PET Metal full forward sequence length exceeds fixed score capacity".to_owned(),
            );
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal full forward attention dimensions are inconsistent".to_owned());
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal full forward atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(seq_len).map_err(|_| {
                "PET Metal full forward sequence length exceeds u32 range".to_owned()
            })?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom).map_err(|_| {
                "PET Metal full forward max edge count exceeds u32 range".to_owned()
            })?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal full forward d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal full forward head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                "PET Metal full forward head dimension exceeds u32 range".to_owned()
            })?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal full forward qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.input_buffer_with_slice(projected, options);
        let cutoff_buffer = self.input_buffer_with_slice(cutoff_factors, options);
        let (attention_buffer, attention_no_copy) =
            self.output_buffer_with_mut_slice(attention, options);
        let (probabilities_buffer, probabilities_no_copy) =
            self.output_buffer_with_mut_slice(probabilities, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_full_forward_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&cutoff_buffer), 0);
        encoder.set_buffer(2, Some(&attention_buffer), 0);
        encoder.set_buffer(3, Some(&probabilities_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);

        let total_threads = (n_atoms * num_heads * seq_len) as u64;
        let execution_width = self
            .attention_full_forward_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_full_forward_pipeline
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
            return Err(format!(
                "PET Metal full forward command buffer did not complete: {status:?}"
            ));
        }

        if !attention_no_copy {
            let attention_ptr = attention_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    attention_ptr,
                    attention.as_mut_ptr(),
                    attention.len(),
                );
            }
        }
        if !probabilities_no_copy {
            let probabilities_ptr = probabilities_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    probabilities_ptr,
                    probabilities.as_mut_ptr(),
                    probabilities.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_vjp_full_projected_f32(
        &self,
        projected: &[f32],
        cutoff_factors: &[f32],
        probabilities: &[f32],
        bar_attention: &[f32],
        n_atoms: usize,
        seq_len: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        bar_projected: &mut [f32],
        bar_cutoff: &mut [f32],
    ) -> Result<(), String> {
        let max_edges_per_atom = seq_len
            .checked_sub(1)
            .ok_or_else(|| "PET Metal full VJP sequence length must be nonzero".to_owned())?;
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal full VJP projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal full VJP projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        if bar_projected.len() != projected_len {
            return Err(format!(
                "PET Metal full VJP bar_projected length must be {projected_len}, got {}",
                bar_projected.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal full VJP cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal full VJP cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        if bar_cutoff.len() != cutoff_len {
            return Err(format!(
                "PET Metal full VJP bar_cutoff length must be {cutoff_len}, got {}",
                bar_cutoff.len()
            ));
        }
        let attention_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(d_edge))
            .ok_or_else(|| "PET Metal full VJP bar_attention length overflows".to_owned())?;
        if bar_attention.len() != attention_len {
            return Err(format!(
                "PET Metal full VJP bar_attention length must be {attention_len}, got {}",
                bar_attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(seq_len))
            .and_then(|values| values.checked_mul(seq_len))
            .ok_or_else(|| "PET Metal full VJP probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal full VJP probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if seq_len > 64 {
            return Err(
                "PET Metal full VJP sequence length exceeds fixed score capacity".to_owned(),
            );
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal full VJP attention dimensions are inconsistent".to_owned());
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal full VJP atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(seq_len)
                .map_err(|_| "PET Metal full VJP sequence length exceeds u32 range".to_owned())?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom)
                .map_err(|_| "PET Metal full VJP max edge count exceeds u32 range".to_owned())?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal full VJP d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal full VJP head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| "PET Metal full VJP head dimension exceeds u32 range".to_owned())?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal full VJP qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.input_buffer_with_slice(projected, options);
        let cutoff_buffer = self.input_buffer_with_slice(cutoff_factors, options);
        let probabilities_buffer = self.input_buffer_with_slice(probabilities, options);
        let bar_attention_buffer = self.input_buffer_with_slice(bar_attention, options);
        let (bar_projected_buffer, bar_projected_no_copy) =
            self.output_buffer_with_mut_slice(bar_projected, options);
        let (bar_cutoff_buffer, bar_cutoff_no_copy) =
            self.output_buffer_with_mut_slice(bar_cutoff, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_full_vjp_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&cutoff_buffer), 0);
        encoder.set_buffer(2, Some(&probabilities_buffer), 0);
        encoder.set_buffer(3, Some(&bar_attention_buffer), 0);
        encoder.set_buffer(4, Some(&bar_projected_buffer), 0);
        encoder.set_buffer(5, Some(&bar_cutoff_buffer), 0);
        encoder.set_buffer(6, Some(&params_buffer), 0);

        let total_threads = n_atoms as u64;
        let execution_width = self
            .attention_full_vjp_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_full_vjp_pipeline
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
            return Err(format!(
                "PET Metal full VJP command buffer did not complete: {status:?}"
            ));
        }

        if !bar_projected_no_copy {
            let bar_projected_ptr = bar_projected_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_projected_ptr,
                    bar_projected.as_mut_ptr(),
                    bar_projected.len(),
                );
            }
        }
        if !bar_cutoff_no_copy {
            let bar_cutoff_ptr = bar_cutoff_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_cutoff_ptr,
                    bar_cutoff.as_mut_ptr(),
                    bar_cutoff.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_vjp_full_projected_split_bar_f32(
        &self,
        projected: &[f32],
        cutoff_factors: &[f32],
        probabilities: &[f32],
        bar_center_attention: &[f32],
        bar_edge_attention: &[f32],
        n_atoms: usize,
        seq_len: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        bar_projected: &mut [f32],
        bar_cutoff: &mut [f32],
    ) -> Result<(), String> {
        let max_edges_per_atom = seq_len
            .checked_sub(1)
            .ok_or_else(|| "PET Metal full split VJP sequence length must be nonzero".to_owned())?;
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal full split VJP projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal full split VJP projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        if bar_projected.len() != projected_len {
            return Err(format!(
                "PET Metal full split VJP bar_projected length must be {projected_len}, got {}",
                bar_projected.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal full split VJP cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal full split VJP cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        if bar_cutoff.len() != cutoff_len {
            return Err(format!(
                "PET Metal full split VJP bar_cutoff length must be {cutoff_len}, got {}",
                bar_cutoff.len()
            ));
        }
        let center_bar_len = n_atoms
            .checked_mul(d_edge)
            .ok_or_else(|| "PET Metal full split VJP center bar length overflows".to_owned())?;
        if bar_center_attention.len() != center_bar_len {
            return Err(format!(
                "PET Metal full split VJP center bar length must be {center_bar_len}, got {}",
                bar_center_attention.len()
            ));
        }
        let edge_bar_len = cutoff_len
            .checked_mul(d_edge)
            .ok_or_else(|| "PET Metal full split VJP edge bar length overflows".to_owned())?;
        if bar_edge_attention.len() != edge_bar_len {
            return Err(format!(
                "PET Metal full split VJP edge bar length must be {edge_bar_len}, got {}",
                bar_edge_attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(seq_len))
            .and_then(|values| values.checked_mul(seq_len))
            .ok_or_else(|| "PET Metal full split VJP probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal full split VJP probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if seq_len > 64 {
            return Err(
                "PET Metal full split VJP sequence length exceeds fixed score capacity".to_owned(),
            );
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err(
                "PET Metal full split VJP attention dimensions are inconsistent".to_owned(),
            );
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal full split VJP atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(seq_len).map_err(|_| {
                "PET Metal full split VJP sequence length exceeds u32 range".to_owned()
            })?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom).map_err(|_| {
                "PET Metal full split VJP max edge count exceeds u32 range".to_owned()
            })?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal full split VJP d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal full split VJP head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                "PET Metal full split VJP head dimension exceeds u32 range".to_owned()
            })?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal full split VJP qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.input_buffer_with_slice(projected, options);
        let cutoff_buffer = self.input_buffer_with_slice(cutoff_factors, options);
        let probabilities_buffer = self.input_buffer_with_slice(probabilities, options);
        let bar_center_buffer = self.input_buffer_with_slice(bar_center_attention, options);
        let bar_edge_buffer = self.input_buffer_with_slice(bar_edge_attention, options);
        let (bar_projected_buffer, bar_projected_no_copy) =
            self.output_buffer_with_mut_slice(bar_projected, options);
        let (bar_cutoff_buffer, bar_cutoff_no_copy) =
            self.output_buffer_with_mut_slice(bar_cutoff, options);
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_full_vjp_split_bar_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&cutoff_buffer), 0);
        encoder.set_buffer(2, Some(&probabilities_buffer), 0);
        encoder.set_buffer(3, Some(&bar_center_buffer), 0);
        encoder.set_buffer(4, Some(&bar_edge_buffer), 0);
        encoder.set_buffer(5, Some(&bar_projected_buffer), 0);
        encoder.set_buffer(6, Some(&bar_cutoff_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);

        let total_threads = n_atoms as u64;
        let execution_width = self
            .attention_full_vjp_split_bar_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_full_vjp_split_bar_pipeline
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
            return Err(format!(
                "PET Metal full split VJP command buffer did not complete: {status:?}"
            ));
        }

        if !bar_projected_no_copy {
            let bar_projected_ptr = bar_projected_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_projected_ptr,
                    bar_projected.as_mut_ptr(),
                    bar_projected.len(),
                );
            }
        }
        if !bar_cutoff_no_copy {
            let bar_cutoff_ptr = bar_cutoff_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_cutoff_ptr,
                    bar_cutoff.as_mut_ptr(),
                    bar_cutoff.len(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_vjp_full_projected_head_parallel_f32(
        &self,
        projected: &[f32],
        cutoff_factors: &[f32],
        probabilities: &[f32],
        bar_attention: &[f32],
        n_atoms: usize,
        seq_len: usize,
        d_edge: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
        attention_eps: f32,
        bar_projected: &mut [f32],
        bar_cutoff: &mut [f32],
    ) -> Result<(), String> {
        let max_edges_per_atom = seq_len
            .checked_sub(1)
            .ok_or_else(|| "PET Metal full head VJP sequence length must be nonzero".to_owned())?;
        let qkv_token_stride = 3 * d_edge;
        let projected_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(qkv_token_stride))
            .ok_or_else(|| "PET Metal full head VJP projected length overflows".to_owned())?;
        if projected.len() != projected_len {
            return Err(format!(
                "PET Metal full head VJP projected length must be {projected_len}, got {}",
                projected.len()
            ));
        }
        if bar_projected.len() != projected_len {
            return Err(format!(
                "PET Metal full head VJP bar_projected length must be {projected_len}, got {}",
                bar_projected.len()
            ));
        }
        let cutoff_len = n_atoms
            .checked_mul(max_edges_per_atom)
            .ok_or_else(|| "PET Metal full head VJP cutoff length overflows".to_owned())?;
        if cutoff_factors.len() != cutoff_len {
            return Err(format!(
                "PET Metal full head VJP cutoff length must be {cutoff_len}, got {}",
                cutoff_factors.len()
            ));
        }
        if bar_cutoff.len() != cutoff_len {
            return Err(format!(
                "PET Metal full head VJP bar_cutoff length must be {cutoff_len}, got {}",
                bar_cutoff.len()
            ));
        }
        let attention_len = n_atoms
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(d_edge))
            .ok_or_else(|| "PET Metal full head VJP bar_attention length overflows".to_owned())?;
        if bar_attention.len() != attention_len {
            return Err(format!(
                "PET Metal full head VJP bar_attention length must be {attention_len}, got {}",
                bar_attention.len()
            ));
        }
        let probability_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(seq_len))
            .and_then(|values| values.checked_mul(seq_len))
            .ok_or_else(|| "PET Metal full head VJP probability length overflows".to_owned())?;
        if probabilities.len() != probability_len {
            return Err(format!(
                "PET Metal full head VJP probability length must be {probability_len}, got {}",
                probabilities.len()
            ));
        }
        if seq_len > 64 {
            return Err(
                "PET Metal full head VJP sequence length exceeds fixed score capacity".to_owned(),
            );
        }
        if d_edge != num_heads.saturating_mul(head_dim) {
            return Err("PET Metal full head VJP attention dimensions are inconsistent".to_owned());
        }

        let params = PetAttentionForwardParams {
            n_atoms: u32::try_from(n_atoms)
                .map_err(|_| "PET Metal full head VJP atom count exceeds u32 range".to_owned())?,
            active_len: u32::try_from(seq_len).map_err(|_| {
                "PET Metal full head VJP sequence length exceeds u32 range".to_owned()
            })?,
            max_edges_per_atom: u32::try_from(max_edges_per_atom).map_err(|_| {
                "PET Metal full head VJP max edge count exceeds u32 range".to_owned()
            })?,
            d_edge: u32::try_from(d_edge)
                .map_err(|_| "PET Metal full head VJP d_edge exceeds u32 range".to_owned())?,
            num_heads: u32::try_from(num_heads)
                .map_err(|_| "PET Metal full head VJP head count exceeds u32 range".to_owned())?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                "PET Metal full head VJP head dimension exceeds u32 range".to_owned()
            })?,
            qkv_token_stride: u32::try_from(qkv_token_stride)
                .map_err(|_| "PET Metal full head VJP qkv stride exceeds u32 range".to_owned())?,
            scale,
            attention_eps,
        };

        let options = MTLResourceOptions::StorageModeShared;
        let projected_buffer = self.input_buffer_with_slice(projected, options);
        let cutoff_buffer = self.input_buffer_with_slice(cutoff_factors, options);
        let probabilities_buffer = self.input_buffer_with_slice(probabilities, options);
        let bar_attention_buffer = self.input_buffer_with_slice(bar_attention, options);
        let (bar_projected_buffer, bar_projected_no_copy) =
            self.output_buffer_with_mut_slice(bar_projected, options);
        let cutoff_head_len = n_atoms
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(max_edges_per_atom))
            .ok_or_else(|| "PET Metal full head VJP cutoff-head length overflows".to_owned())?;
        let bar_cutoff_heads_buffer = self.device.new_buffer(
            (cutoff_head_len * std::mem::size_of::<f32>()) as u64,
            options,
        );
        let params_buffer = self.buffer_with_slice(std::slice::from_ref(&params), options);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_full_vjp_head_pipeline);
        encoder.set_buffer(0, Some(&projected_buffer), 0);
        encoder.set_buffer(1, Some(&cutoff_buffer), 0);
        encoder.set_buffer(2, Some(&probabilities_buffer), 0);
        encoder.set_buffer(3, Some(&bar_attention_buffer), 0);
        encoder.set_buffer(4, Some(&bar_projected_buffer), 0);
        encoder.set_buffer(5, Some(&bar_cutoff_heads_buffer), 0);
        encoder.set_buffer(6, Some(&params_buffer), 0);

        let total_threads = (n_atoms * num_heads) as u64;
        let execution_width = self
            .attention_full_vjp_head_pipeline
            .thread_execution_width()
            .max(1);
        let max_threads = self
            .attention_full_vjp_head_pipeline
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
            return Err(format!(
                "PET Metal full head VJP command buffer did not complete: {status:?}"
            ));
        }

        if !bar_projected_no_copy {
            let bar_projected_ptr = bar_projected_buffer.contents().cast::<f32>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bar_projected_ptr,
                    bar_projected.as_mut_ptr(),
                    bar_projected.len(),
                );
            }
        }

        let bar_cutoff_heads_ptr = bar_cutoff_heads_buffer.contents().cast::<f32>();
        let bar_cutoff_heads =
            unsafe { std::slice::from_raw_parts(bar_cutoff_heads_ptr, cutoff_head_len) };
        for atom in 0..n_atoms {
            let cutoff_start = atom * max_edges_per_atom;
            for head in 0..num_heads {
                let head_start = (atom * num_heads + head) * max_edges_per_atom;
                for edge in 0..max_edges_per_atom {
                    bar_cutoff[cutoff_start + edge] += bar_cutoff_heads[head_start + edge];
                }
            }
        }
        Ok(())
    }
}

fn backend() -> Result<&'static PetMetalBackend, String> {
    static BACKEND: OnceLock<Result<PetMetalBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(PetMetalBackend::new)
        .as_ref()
        .map_err(Clone::clone)
}

fn pet_metal_no_copy_input_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("LLMLIP_PET_METAL_NOCOPY").ok().as_deref(),
            Some("1")
                | Some("true")
                | Some("TRUE")
                | Some("yes")
                | Some("YES")
                | Some("on")
                | Some("ON")
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_forward_compact_projected_f32(
    projected: &[f32],
    active_tokens: &[usize],
    cutoff_factors: &[f32],
    n_atoms: usize,
    active_len: usize,
    max_edges_per_atom: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    attention: &mut [f32],
    probabilities: &mut [f32],
) -> Result<(), String> {
    let active_tokens_u32 = active_tokens
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| "PET Metal active token exceeds u32 range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    backend()?.attention_forward_compact_projected_f32(
        projected,
        &active_tokens_u32,
        cutoff_factors,
        n_atoms,
        active_len,
        max_edges_per_atom,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        attention,
        probabilities,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_forward_ragged_compact_projected_f32(
    projected: &[f32],
    active_tokens: &[usize],
    active_token_offsets: &[usize],
    compact_probability_offsets: &[usize],
    cutoff_factors: &[f32],
    n_atoms: usize,
    max_active_len: usize,
    max_edges_per_atom: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    attention: &mut [f32],
    probabilities: &mut [f32],
) -> Result<(), String> {
    let active_tokens_u32 = active_tokens
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| "PET Metal active token exceeds u32 range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let active_token_offsets_u32 = active_token_offsets
        .iter()
        .map(|&offset| {
            u32::try_from(offset)
                .map_err(|_| "PET Metal active token offset exceeds u32 range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let compact_probability_offsets_u32 = compact_probability_offsets
        .iter()
        .map(|&offset| {
            u32::try_from(offset)
                .map_err(|_| "PET Metal probability offset exceeds u32 range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    backend()?.attention_forward_ragged_compact_projected_f32(
        projected,
        &active_tokens_u32,
        &active_token_offsets_u32,
        &compact_probability_offsets_u32,
        cutoff_factors,
        n_atoms,
        max_active_len,
        max_edges_per_atom,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        attention,
        probabilities,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_vjp_compact_projected_f32(
    projected: &[f32],
    active_tokens: &[usize],
    cutoff_factors: &[f32],
    probabilities: &[f32],
    bar_attention: &[f32],
    n_atoms: usize,
    active_len: usize,
    max_edges_per_atom: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    bar_projected: &mut [f32],
    bar_cutoff: &mut [f32],
) -> Result<(), String> {
    let active_tokens_u32 = active_tokens
        .iter()
        .map(|&token| {
            u32::try_from(token)
                .map_err(|_| "PET Metal VJP active token exceeds u32 range".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    backend()?.attention_vjp_compact_projected_f32(
        projected,
        &active_tokens_u32,
        cutoff_factors,
        probabilities,
        bar_attention,
        n_atoms,
        active_len,
        max_edges_per_atom,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        bar_projected,
        bar_cutoff,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_forward_full_projected_f32(
    projected: &[f32],
    cutoff_factors: &[f32],
    n_atoms: usize,
    seq_len: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    attention: &mut [f32],
    probabilities: &mut [f32],
) -> Result<(), String> {
    backend()?.attention_forward_full_projected_f32(
        projected,
        cutoff_factors,
        n_atoms,
        seq_len,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        attention,
        probabilities,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_vjp_full_projected_f32(
    projected: &[f32],
    cutoff_factors: &[f32],
    probabilities: &[f32],
    bar_attention: &[f32],
    n_atoms: usize,
    seq_len: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    bar_projected: &mut [f32],
    bar_cutoff: &mut [f32],
) -> Result<(), String> {
    backend()?.attention_vjp_full_projected_f32(
        projected,
        cutoff_factors,
        probabilities,
        bar_attention,
        n_atoms,
        seq_len,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        bar_projected,
        bar_cutoff,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_vjp_full_projected_split_bar_f32(
    projected: &[f32],
    cutoff_factors: &[f32],
    probabilities: &[f32],
    bar_center_attention: &[f32],
    bar_edge_attention: &[f32],
    n_atoms: usize,
    seq_len: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    bar_projected: &mut [f32],
    bar_cutoff: &mut [f32],
) -> Result<(), String> {
    backend()?.attention_vjp_full_projected_split_bar_f32(
        projected,
        cutoff_factors,
        probabilities,
        bar_center_attention,
        bar_edge_attention,
        n_atoms,
        seq_len,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        bar_projected,
        bar_cutoff,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_vjp_full_projected_head_parallel_f32(
    projected: &[f32],
    cutoff_factors: &[f32],
    probabilities: &[f32],
    bar_attention: &[f32],
    n_atoms: usize,
    seq_len: usize,
    d_edge: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    attention_eps: f32,
    bar_projected: &mut [f32],
    bar_cutoff: &mut [f32],
) -> Result<(), String> {
    backend()?.attention_vjp_full_projected_head_parallel_f32(
        projected,
        cutoff_factors,
        probabilities,
        bar_attention,
        n_atoms,
        seq_len,
        d_edge,
        num_heads,
        head_dim,
        scale,
        attention_eps,
        bar_projected,
        bar_cutoff,
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn pet_metal_compact_projected_attention_forward_matches_cpu() {
        let n_atoms = 2;
        let seq_len = 4;
        let active_len = 3;
        let max_edges_per_atom = seq_len - 1;
        let d_edge = 128;
        let num_heads = 8;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let active_tokens = vec![0, 1, 3, 0, 2, 3];
        let padding_mask = [true, false, true, false, true, true];
        let cutoff_factors = vec![0.8, 0.0, 0.35, 0.0, 0.7, 0.4];
        let qkv_token_stride = 3 * d_edge;
        let projected = (0..n_atoms * active_len * qkv_token_stride)
            .map(|index| ((index % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();

        let mut expected_attention = vec![0.0; n_atoms * active_len * d_edge];
        let mut expected_probabilities = vec![0.0; n_atoms * num_heads * active_len * active_len];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * active_len * qkv_token_stride..(atom + 1) * active_len * qkv_token_stride];
            let active_tokens_atom = &active_tokens[atom * active_len..(atom + 1) * active_len];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let attention_atom = &mut expected_attention
                [atom * active_len * d_edge..(atom + 1) * active_len * d_edge];
            let probabilities_atom =
                &mut expected_probabilities[atom * num_heads * active_len * active_len
                    ..(atom + 1) * num_heads * active_len * active_len];
            super::super::attention_atom_forward_compact_projected_output_and_probabilities_f32(
                projected_atom,
                active_tokens_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                attention_atom,
                probabilities_atom,
                None,
            )
            .expect("CPU compact attention succeeds");
        }

        let mut actual_attention = vec![0.0; expected_attention.len()];
        let mut actual_probabilities = vec![0.0; expected_probabilities.len()];
        attention_forward_compact_projected_f32(
            &projected,
            &active_tokens,
            &cutoff_factors,
            n_atoms,
            active_len,
            max_edges_per_atom,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut actual_attention,
            &mut actual_probabilities,
        )
        .expect("Metal compact attention succeeds");

        for (index, (actual, expected)) in actual_attention
            .iter()
            .zip(expected_attention.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "attention[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in actual_probabilities
            .iter()
            .zip(expected_probabilities.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "probabilities[{index}] expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn pet_metal_ragged_compact_projected_attention_forward_matches_cpu() {
        let n_atoms = 2;
        let seq_len = 5;
        let max_edges_per_atom = seq_len - 1;
        let d_edge = 128;
        let num_heads = 8;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let active_token_offsets = vec![0, 4, 6];
        let active_tokens = vec![0, 1, 3, 4, 0, 2];
        let compact_probability_offsets =
            vec![0, num_heads * 4 * 4, num_heads * 4 * 4 + num_heads * 2 * 2];
        let padding_mask = [true, false, true, true, false, true, false, false];
        let cutoff_factors = vec![0.8, 0.0, 0.35, 0.6, 0.0, 0.7, 0.0, 0.0];
        let qkv_token_stride = 3 * d_edge;
        let projected = (0..active_tokens.len() * qkv_token_stride)
            .map(|index| ((index % 43) as f32 - 21.0) / 29.0)
            .collect::<Vec<_>>();

        let mut expected_attention = vec![0.0; active_tokens.len() * d_edge];
        let mut expected_probabilities = vec![
            0.0;
            *compact_probability_offsets
                .last()
                .expect("probability offset")
        ];
        for atom in 0..n_atoms {
            let active_start = active_token_offsets[atom];
            let active_end = active_token_offsets[atom + 1];
            let active_len = active_end - active_start;
            let projected_atom =
                &projected[active_start * qkv_token_stride..active_end * qkv_token_stride];
            let active_tokens_atom = &active_tokens[active_start..active_end];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let attention_atom =
                &mut expected_attention[active_start * d_edge..active_end * d_edge];
            let probabilities_atom = &mut expected_probabilities
                [compact_probability_offsets[atom]..compact_probability_offsets[atom + 1]];
            super::super::attention_atom_forward_compact_projected_output_and_probabilities_f32(
                projected_atom,
                active_tokens_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                attention_atom,
                probabilities_atom,
                None,
            )
            .expect("CPU ragged compact attention succeeds");
            assert_eq!(
                probabilities_atom.len(),
                num_heads * active_len * active_len
            );
        }

        let mut actual_attention = vec![0.0; expected_attention.len()];
        let mut actual_probabilities = vec![0.0; expected_probabilities.len()];
        attention_forward_ragged_compact_projected_f32(
            &projected,
            &active_tokens,
            &active_token_offsets,
            &compact_probability_offsets,
            &cutoff_factors,
            n_atoms,
            4,
            max_edges_per_atom,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut actual_attention,
            &mut actual_probabilities,
        )
        .expect("Metal ragged compact attention succeeds");

        for (index, (actual, expected)) in actual_attention
            .iter()
            .zip(expected_attention.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "attention[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in actual_probabilities
            .iter()
            .zip(expected_probabilities.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "probabilities[{index}] expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn pet_metal_full_projected_attention_forward_matches_cpu() {
        let n_atoms = 2;
        let seq_len = 5;
        let max_edges_per_atom = seq_len - 1;
        let d_edge = 128;
        let num_heads = 8;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let padding_mask = [true, false, true, true, true, true, false, true];
        let cutoff_factors = vec![0.8, 0.0, 0.35, 0.6, 0.9, 0.7, 0.0, 0.4];
        let qkv_token_stride = 3 * d_edge;
        let projected = (0..n_atoms * seq_len * qkv_token_stride)
            .map(|index| ((index % 47) as f32 - 23.0) / 31.0)
            .collect::<Vec<_>>();

        let mut expected_attention = vec![0.0; n_atoms * seq_len * d_edge];
        let mut expected_probabilities = vec![0.0; n_atoms * num_heads * seq_len * seq_len];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * seq_len * qkv_token_stride..(atom + 1) * seq_len * qkv_token_stride];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let attention_atom =
                &mut expected_attention[atom * seq_len * d_edge..(atom + 1) * seq_len * d_edge];
            let probabilities_atom = &mut expected_probabilities
                [atom * num_heads * seq_len * seq_len..(atom + 1) * num_heads * seq_len * seq_len];
            super::super::attention_atom_forward_f32(
                projected_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                attention_atom,
                Some(probabilities_atom),
                false,
                None,
            )
            .expect("CPU full attention succeeds");
        }

        let mut actual_attention = vec![0.0; expected_attention.len()];
        let mut actual_probabilities = vec![0.0; expected_probabilities.len()];
        attention_forward_full_projected_f32(
            &projected,
            &cutoff_factors,
            n_atoms,
            seq_len,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut actual_attention,
            &mut actual_probabilities,
        )
        .expect("Metal full attention succeeds");

        for (index, (actual, expected)) in actual_attention
            .iter()
            .zip(expected_attention.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "full attention[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in actual_probabilities
            .iter()
            .zip(expected_probabilities.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "full probabilities[{index}] expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn pet_metal_compact_projected_attention_vjp_matches_cpu() {
        let n_atoms = 2;
        let seq_len = 4;
        let active_len = 3;
        let max_edges_per_atom = seq_len - 1;
        let d_edge = 128;
        let num_heads = 8;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let active_tokens = vec![0, 1, 3, 0, 2, 3];
        let padding_mask = [true, false, true, false, true, true];
        let cutoff_factors = vec![0.8, 0.0, 0.35, 0.0, 0.7, 0.4];
        let qkv_token_stride = 3 * d_edge;
        let projected = (0..n_atoms * active_len * qkv_token_stride)
            .map(|index| ((index % 41) as f32 - 20.0) / 29.0)
            .collect::<Vec<_>>();
        let bar_attention = (0..n_atoms * active_len * d_edge)
            .map(|index| ((index % 31) as f32 - 15.0) / 37.0)
            .collect::<Vec<_>>();

        let mut probabilities = vec![0.0; n_atoms * num_heads * active_len * active_len];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * active_len * qkv_token_stride..(atom + 1) * active_len * qkv_token_stride];
            let active_tokens_atom = &active_tokens[atom * active_len..(atom + 1) * active_len];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let mut attention_atom = vec![0.0; active_len * d_edge];
            let probabilities_atom = &mut probabilities[atom * num_heads * active_len * active_len
                ..(atom + 1) * num_heads * active_len * active_len];
            super::super::attention_atom_forward_compact_projected_output_and_probabilities_f32(
                projected_atom,
                active_tokens_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                &mut attention_atom,
                probabilities_atom,
                None,
            )
            .expect("CPU compact attention forward succeeds");
        }

        let mut expected_bar_projected = vec![0.0; projected.len()];
        let mut expected_bar_cutoff = vec![0.0; cutoff_factors.len()];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * active_len * qkv_token_stride..(atom + 1) * active_len * qkv_token_stride];
            let active_tokens_atom = &active_tokens[atom * active_len..(atom + 1) * active_len];
            let probabilities_atom = &probabilities[atom * num_heads * active_len * active_len
                ..(atom + 1) * num_heads * active_len * active_len];
            let bar_attention_atom =
                &bar_attention[atom * active_len * d_edge..(atom + 1) * active_len * d_edge];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let bar_projected_atom = &mut expected_bar_projected
                [atom * active_len * qkv_token_stride..(atom + 1) * active_len * qkv_token_stride];
            let bar_cutoff_atom = &mut expected_bar_cutoff
                [atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            super::super::attention_atom_vjp_from_compact_probabilities_compact_projected_f32(
                projected_atom,
                active_tokens_atom,
                probabilities_atom,
                bar_attention_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                bar_projected_atom,
                bar_cutoff_atom,
                None,
            )
            .expect("CPU compact attention VJP succeeds");
        }

        let mut actual_bar_projected = vec![0.0; expected_bar_projected.len()];
        let mut actual_bar_cutoff = vec![0.0; expected_bar_cutoff.len()];
        attention_vjp_compact_projected_f32(
            &projected,
            &active_tokens,
            &cutoff_factors,
            &probabilities,
            &bar_attention,
            n_atoms,
            active_len,
            max_edges_per_atom,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut actual_bar_projected,
            &mut actual_bar_cutoff,
        )
        .expect("Metal compact attention VJP succeeds");

        for (index, (actual, expected)) in actual_bar_projected
            .iter()
            .zip(expected_bar_projected.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "bar_projected[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in actual_bar_cutoff
            .iter()
            .zip(expected_bar_cutoff.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "bar_cutoff[{index}] expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn pet_metal_full_projected_attention_vjp_matches_cpu() {
        let n_atoms = 2;
        let seq_len = 5;
        let max_edges_per_atom = seq_len - 1;
        let d_edge = 128;
        let num_heads = 8;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let active_tokens = (0..seq_len).collect::<Vec<_>>();
        let padding_mask = [true, false, true, true, true, true, false, true];
        let cutoff_factors = vec![0.8, 0.0, 0.35, 0.6, 0.9, 0.7, 0.0, 0.4];
        let qkv_token_stride = 3 * d_edge;
        let projected = (0..n_atoms * seq_len * qkv_token_stride)
            .map(|index| ((index % 43) as f32 - 21.0) / 31.0)
            .collect::<Vec<_>>();
        let bar_attention = (0..n_atoms * seq_len * d_edge)
            .map(|index| ((index % 29) as f32 - 14.0) / 41.0)
            .collect::<Vec<_>>();

        let mut probabilities = vec![0.0; n_atoms * num_heads * seq_len * seq_len];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * seq_len * qkv_token_stride..(atom + 1) * seq_len * qkv_token_stride];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let mut attention_atom = vec![0.0; seq_len * d_edge];
            let probabilities_atom = &mut probabilities
                [atom * num_heads * seq_len * seq_len..(atom + 1) * num_heads * seq_len * seq_len];
            super::super::attention_atom_forward_compact_output_and_probabilities_f32(
                projected_atom,
                &active_tokens,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                &mut attention_atom,
                probabilities_atom,
                None,
            )
            .expect("CPU full attention forward succeeds");
        }

        let mut expected_bar_projected = vec![0.0; projected.len()];
        let mut expected_bar_cutoff = vec![0.0; cutoff_factors.len()];
        for atom in 0..n_atoms {
            let projected_atom = &projected
                [atom * seq_len * qkv_token_stride..(atom + 1) * seq_len * qkv_token_stride];
            let probabilities_atom = &probabilities
                [atom * num_heads * seq_len * seq_len..(atom + 1) * num_heads * seq_len * seq_len];
            let bar_attention_atom =
                &bar_attention[atom * seq_len * d_edge..(atom + 1) * seq_len * d_edge];
            let padding_atom =
                &padding_mask[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let cutoff_atom =
                &cutoff_factors[atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            let bar_projected_atom = &mut expected_bar_projected
                [atom * seq_len * qkv_token_stride..(atom + 1) * seq_len * qkv_token_stride];
            let bar_cutoff_atom = &mut expected_bar_cutoff
                [atom * max_edges_per_atom..(atom + 1) * max_edges_per_atom];
            super::super::attention_atom_vjp_from_probabilities_f32(
                projected_atom,
                probabilities_atom,
                bar_attention_atom,
                padding_atom,
                cutoff_atom,
                seq_len,
                d_edge,
                num_heads,
                head_dim,
                scale,
                bar_projected_atom,
                bar_cutoff_atom,
                false,
                None,
            )
            .expect("CPU full attention VJP succeeds");
        }

        let mut actual_bar_projected = vec![0.0; expected_bar_projected.len()];
        let mut actual_bar_cutoff = vec![0.0; expected_bar_cutoff.len()];
        attention_vjp_full_projected_f32(
            &projected,
            &cutoff_factors,
            &probabilities,
            &bar_attention,
            n_atoms,
            seq_len,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut actual_bar_projected,
            &mut actual_bar_cutoff,
        )
        .expect("Metal full attention VJP succeeds");

        for (index, (actual, expected)) in actual_bar_projected
            .iter()
            .zip(expected_bar_projected.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full bar_projected[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in actual_bar_cutoff
            .iter()
            .zip(expected_bar_cutoff.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full bar_cutoff[{index}] expected {expected}, got {actual}"
            );
        }

        let mut head_bar_projected = vec![0.0; expected_bar_projected.len()];
        let mut head_bar_cutoff = vec![0.0; expected_bar_cutoff.len()];
        attention_vjp_full_projected_head_parallel_f32(
            &projected,
            &cutoff_factors,
            &probabilities,
            &bar_attention,
            n_atoms,
            seq_len,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut head_bar_projected,
            &mut head_bar_cutoff,
        )
        .expect("Metal full head-parallel attention VJP succeeds");

        for (index, (actual, expected)) in head_bar_projected
            .iter()
            .zip(expected_bar_projected.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full head bar_projected[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in head_bar_cutoff
            .iter()
            .zip(expected_bar_cutoff.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full head bar_cutoff[{index}] expected {expected}, got {actual}"
            );
        }

        let mut bar_center_attention = vec![0.0; n_atoms * d_edge];
        let mut bar_edge_attention = vec![0.0; n_atoms * max_edges_per_atom * d_edge];
        for atom in 0..n_atoms {
            let source_start = atom * seq_len * d_edge;
            bar_center_attention[atom * d_edge..(atom + 1) * d_edge]
                .copy_from_slice(&bar_attention[source_start..source_start + d_edge]);
            for edge in 0..max_edges_per_atom {
                let source = source_start + (edge + 1) * d_edge;
                let target = (atom * max_edges_per_atom + edge) * d_edge;
                bar_edge_attention[target..target + d_edge]
                    .copy_from_slice(&bar_attention[source..source + d_edge]);
            }
        }

        let mut split_bar_projected = vec![0.0; expected_bar_projected.len()];
        let mut split_bar_cutoff = vec![0.0; expected_bar_cutoff.len()];
        attention_vjp_full_projected_split_bar_f32(
            &projected,
            &cutoff_factors,
            &probabilities,
            &bar_center_attention,
            &bar_edge_attention,
            n_atoms,
            seq_len,
            d_edge,
            num_heads,
            head_dim,
            scale,
            super::super::PET_MAD_XS_ATTENTION_EPS as f32,
            &mut split_bar_projected,
            &mut split_bar_cutoff,
        )
        .expect("Metal full split attention VJP succeeds");

        for (index, (actual, expected)) in split_bar_projected
            .iter()
            .zip(expected_bar_projected.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full split bar_projected[{index}] expected {expected}, got {actual}"
            );
        }
        for (index, (actual, expected)) in split_bar_cutoff
            .iter()
            .zip(expected_bar_cutoff.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 4.0e-4,
                "full split bar_cutoff[{index}] expected {expected}, got {actual}"
            );
        }
    }
}
