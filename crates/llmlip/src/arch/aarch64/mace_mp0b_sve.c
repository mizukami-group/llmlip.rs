#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t x_base;
    size_t x_dim;
    size_t y_base;
    size_t out_base;
    size_t out_dim;
    size_t weight_base;
} LlmlipMp0bLayer1TpTemplate;

typedef struct {
    size_t i_m;
    size_t j_m;
    size_t k_m;
    float scale;
} LlmlipMp0bLayer1CgF32;

static inline void llmlip_mp0b_hidden_block_sources(
    size_t output_block,
    size_t *source_begin,
    size_t *source_end,
    size_t *component_dim,
    size_t *input_offset,
    size_t *output_offset
) {
    switch (output_block) {
        case 0:
            *source_begin = 0;
            *source_end = 2;
            *component_dim = 1;
            *input_offset = 0;
            *output_offset = 0;
            break;
        case 1:
            *source_begin = 2;
            *source_end = 5;
            *component_dim = 3;
            *input_offset = 256;
            *output_offset = 128;
            break;
        case 2:
            *source_begin = 5;
            *source_end = 8;
            *component_dim = 5;
            *input_offset = 1408;
            *output_offset = 512;
            break;
        default:
            *source_begin = 8;
            *source_end = 10;
            *component_dim = 7;
            *input_offset = 3328;
            *output_offset = 1152;
            break;
    }
}

void llmlip_mp0b_layer1_hidden_linear_sve_f32(
    const float *input,
    const float *density,
    const float *weight,
    float *out,
    size_t node_count
) {
    const size_t features = 128;
    const size_t input_row_len = 5120;
    const size_t output_row_len = 2048;
    const size_t block_weights = features * features;
    const size_t lanes = svcntw();
    for (size_t node = 0; node < node_count; ++node) {
        const float *input_row = input + node * input_row_len;
        float *out_row = out + node * output_row_len;
        const float inv_density = 1.0f / (density[node] + 1.0f);
        for (size_t output_block = 0; output_block < 4; ++output_block) {
            size_t source_begin;
            size_t source_end;
            size_t component_dim;
            size_t first_input_offset;
            size_t output_offset;
            llmlip_mp0b_hidden_block_sources(
                output_block,
                &source_begin,
                &source_end,
                &component_dim,
                &first_input_offset,
                &output_offset
            );
            for (size_t component = 0; component < component_dim; ++component) {
                for (size_t out_feature = 0; out_feature < features; out_feature += lanes) {
                    svbool_t pg =
                        svwhilelt_b32((uint32_t)out_feature, (uint32_t)features);
                    svfloat32_t acc = svdup_n_f32(0.0f);
                    size_t input_offset = first_input_offset;
                    for (size_t source = source_begin; source < source_end; ++source) {
                        const float *weight_block = weight + source * block_weights;
                        for (size_t in_feature = 0; in_feature < features; ++in_feature) {
                            const float x =
                                input_row[input_offset + in_feature * component_dim + component];
                            svfloat32_t wv =
                                svld1(pg, weight_block + in_feature * features + out_feature);
                            acc = svmla_n_f32_x(pg, acc, wv, x);
                        }
                        input_offset += features * component_dim;
                    }
                    acc = svmul_n_f32_x(pg, acc, inv_density);
                    svuint32_t indices = svindex_u32(
                        (uint32_t)(output_offset + component + out_feature * component_dim),
                        (uint32_t)component_dim
                    );
                    svst1_scatter_u32index_f32(pg, out_row, indices, acc);
                }
            }
        }
    }
}

void llmlip_mp0b_layer1_hidden_linear_vjp_sve_f32(
    const float *density,
    const float *linear_out,
    const float *weight,
    const float *bar_out,
    float *bar_input,
    float *bar_density,
    size_t node_count
) {
    const size_t features = 128;
    const size_t input_row_len = 5120;
    const size_t output_row_len = 2048;
    const size_t block_weights = features * features;
    const size_t lanes = svcntw();
    svbool_t all = svptrue_b32();
    for (size_t node = 0; node < node_count; ++node) {
        const float inv_density = 1.0f / (density[node] + 1.0f);
        const float *linear_out_row = linear_out + node * output_row_len;
        const float *bar_out_row = bar_out + node * output_row_len;
        float *bar_input_row = bar_input + node * input_row_len;
        svfloat32_t bd_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < output_row_len; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint32_t)index, (uint32_t)output_row_len);
            svfloat32_t lo = svld1(pg, linear_out_row + index);
            svfloat32_t bo = svld1(pg, bar_out_row + index);
            bd_v = svmla_f32_m(pg, bd_v, lo, bo);
        }
        bar_density[node] -= svaddv_f32(all, bd_v) * inv_density;

        for (size_t output_block = 0; output_block < 4; ++output_block) {
            size_t source_begin;
            size_t source_end;
            size_t component_dim;
            size_t first_input_offset;
            size_t output_offset;
            llmlip_mp0b_hidden_block_sources(
                output_block,
                &source_begin,
                &source_end,
                &component_dim,
                &first_input_offset,
                &output_offset
            );
            for (size_t component = 0; component < component_dim; ++component) {
                size_t input_offset = first_input_offset;
                for (size_t source = source_begin; source < source_end; ++source) {
                    const float *weight_block = weight + source * block_weights;
                    for (size_t in_feature = 0; in_feature < features; ++in_feature) {
                        svfloat32_t acc = svdup_n_f32(0.0f);
                        for (size_t out_feature = 0; out_feature < features; out_feature += lanes) {
                            svbool_t pg =
                                svwhilelt_b32((uint32_t)out_feature, (uint32_t)features);
                            svuint32_t indices = svindex_u32(
                                (uint32_t)(
                                    output_offset + component + out_feature * component_dim
                                ),
                                (uint32_t)component_dim
                            );
                            svfloat32_t bo =
                                svld1_gather_u32index_f32(pg, bar_out_row, indices);
                            svfloat32_t wv =
                                svld1(pg, weight_block + in_feature * features + out_feature);
                            acc = svmla_f32_m(pg, acc, bo, wv);
                        }
                        bar_input_row[input_offset + in_feature * component_dim + component] =
                            svaddv_f32(all, acc) * inv_density;
                    }
                    input_offset += features * component_dim;
                }
            }
        }
    }
}

static inline svfloat32_t llmlip_sve_load_strided_f32(
    const float *base,
    size_t offset,
    size_t stride,
    svbool_t pg
) {
    if (stride == 1) {
        return svld1(pg, base + offset);
    }
    svuint32_t indices = svindex_u32((uint32_t)offset, (uint32_t)stride);
    return svld1_gather_u32index_f32(pg, base, indices);
}

static inline void llmlip_sve_add_store_strided_f32(
    float *base,
    size_t offset,
    size_t stride,
    svfloat32_t value,
    svbool_t pg
) {
    if (stride == 1) {
        svfloat32_t current = svld1(pg, base + offset);
        svst1(pg, base + offset, svadd_f32_x(pg, current, value));
        return;
    }
    svuint32_t indices = svindex_u32((uint32_t)offset, (uint32_t)stride);
    svfloat32_t current = svld1_gather_u32index_f32(pg, base, indices);
    svst1_scatter_u32index_f32(pg, base, indices, svadd_f32_x(pg, current, value));
}

void llmlip_mp0b_layer1_scalar_vector_edge_sve_f32(
    const float *x,
    const float *y,
    const float *w,
    float *scatter,
    const LlmlipMp0bLayer1TpTemplate *templates,
    size_t template_count,
    const size_t *cg_offsets,
    const LlmlipMp0bLayer1CgF32 *cgs
) {
    const size_t features = 128;
    const size_t lanes = svcntw();
    for (size_t template_index = 0; template_index < template_count; ++template_index) {
        LlmlipMp0bLayer1TpTemplate template = templates[template_index];
        const size_t cg_start = cg_offsets[template_index];
        const size_t cg_end = cg_offsets[template_index + 1];
        for (size_t feature = 0; feature < features; feature += lanes) {
            svbool_t pg = svwhilelt_b32((uint32_t)feature, (uint32_t)features);
            svfloat32_t weight_v = svld1(pg, w + template.weight_base + feature);
            const size_t x_base = template.x_base + feature * template.x_dim;
            const size_t out_base = template.out_base + feature * template.out_dim;
            for (size_t cg_index = cg_start; cg_index < cg_end; ++cg_index) {
                LlmlipMp0bLayer1CgF32 cg = cgs[cg_index];
                svfloat32_t x_v = llmlip_sve_load_strided_f32(
                    x,
                    x_base + cg.i_m,
                    template.x_dim,
                    pg
                );
                svfloat32_t scale_v =
                    svdup_n_f32(cg.scale * y[template.y_base + cg.j_m]);
                svfloat32_t value =
                    svmul_f32_x(pg, svmul_f32_x(pg, weight_v, scale_v), x_v);
                llmlip_sve_add_store_strided_f32(
                    scatter,
                    out_base + cg.k_m,
                    template.out_dim,
                    value,
                    pg
                );
            }
        }
    }
}

void llmlip_mp0b_layer1_scalar_vector_vjp_edge_sve_f32(
    const float *x,
    const float *y,
    const float *w,
    const float *bar_scatter,
    float *bar_w,
    float *bar_x_values,
    float *bar_y,
    const LlmlipMp0bLayer1TpTemplate *templates,
    size_t template_count,
    const size_t *cg_offsets,
    const LlmlipMp0bLayer1CgF32 *cgs
) {
    const size_t features = 128;
    const size_t lanes = svcntw();
    svfloat32_t bar_y0 = svdup_n_f32(0.0f);
    svfloat32_t bar_y1 = svdup_n_f32(0.0f);
    svfloat32_t bar_y2 = svdup_n_f32(0.0f);
    svfloat32_t bar_y3 = svdup_n_f32(0.0f);
    svfloat32_t bar_y4 = svdup_n_f32(0.0f);
    svfloat32_t bar_y5 = svdup_n_f32(0.0f);
    svfloat32_t bar_y6 = svdup_n_f32(0.0f);
    svfloat32_t bar_y7 = svdup_n_f32(0.0f);
    svfloat32_t bar_y8 = svdup_n_f32(0.0f);
    svfloat32_t bar_y9 = svdup_n_f32(0.0f);
    svfloat32_t bar_y10 = svdup_n_f32(0.0f);
    svfloat32_t bar_y11 = svdup_n_f32(0.0f);
    svfloat32_t bar_y12 = svdup_n_f32(0.0f);
    svfloat32_t bar_y13 = svdup_n_f32(0.0f);
    svfloat32_t bar_y14 = svdup_n_f32(0.0f);
    svfloat32_t bar_y15 = svdup_n_f32(0.0f);

#define LLMLIP_ACCUM_BAR_Y(INDEX, VALUE) \
    do { \
        switch (INDEX) { \
            case 0: bar_y0 = svadd_f32_m(pg, bar_y0, VALUE); break; \
            case 1: bar_y1 = svadd_f32_m(pg, bar_y1, VALUE); break; \
            case 2: bar_y2 = svadd_f32_m(pg, bar_y2, VALUE); break; \
            case 3: bar_y3 = svadd_f32_m(pg, bar_y3, VALUE); break; \
            case 4: bar_y4 = svadd_f32_m(pg, bar_y4, VALUE); break; \
            case 5: bar_y5 = svadd_f32_m(pg, bar_y5, VALUE); break; \
            case 6: bar_y6 = svadd_f32_m(pg, bar_y6, VALUE); break; \
            case 7: bar_y7 = svadd_f32_m(pg, bar_y7, VALUE); break; \
            case 8: bar_y8 = svadd_f32_m(pg, bar_y8, VALUE); break; \
            case 9: bar_y9 = svadd_f32_m(pg, bar_y9, VALUE); break; \
            case 10: bar_y10 = svadd_f32_m(pg, bar_y10, VALUE); break; \
            case 11: bar_y11 = svadd_f32_m(pg, bar_y11, VALUE); break; \
            case 12: bar_y12 = svadd_f32_m(pg, bar_y12, VALUE); break; \
            case 13: bar_y13 = svadd_f32_m(pg, bar_y13, VALUE); break; \
            case 14: bar_y14 = svadd_f32_m(pg, bar_y14, VALUE); break; \
            case 15: bar_y15 = svadd_f32_m(pg, bar_y15, VALUE); break; \
            default: break; \
        } \
    } while (0)

    for (size_t template_index = 0; template_index < template_count; ++template_index) {
        LlmlipMp0bLayer1TpTemplate template = templates[template_index];
        const size_t cg_start = cg_offsets[template_index];
        const size_t cg_end = cg_offsets[template_index + 1];
        for (size_t feature = 0; feature < features; feature += lanes) {
            svbool_t pg = svwhilelt_b32((uint32_t)feature, (uint32_t)features);
            const size_t weight_index = template.weight_base + feature;
            svfloat32_t weight_v = svld1(pg, w + weight_index);
            svfloat32_t bar_weight_v = svdup_n_f32(0.0f);
            const size_t x_base = template.x_base + feature * template.x_dim;
            const size_t out_base = template.out_base + feature * template.out_dim;
            for (size_t cg_index = cg_start; cg_index < cg_end; ++cg_index) {
                LlmlipMp0bLayer1CgF32 cg = cgs[cg_index];
                const size_t x_index = x_base + cg.i_m;
                const size_t y_index = template.y_base + cg.j_m;
                const size_t out_index = out_base + cg.k_m;
                svfloat32_t x_v =
                    llmlip_sve_load_strided_f32(x, x_index, template.x_dim, pg);
                svfloat32_t bar_out_v = llmlip_sve_load_strided_f32(
                    bar_scatter,
                    out_index,
                    template.out_dim,
                    pg
                );
                svfloat32_t scale_v = svdup_n_f32(cg.scale);
                svfloat32_t y_v = svdup_n_f32(y[y_index]);
                svfloat32_t scaled_bar =
                    svmul_f32_x(pg, svmul_f32_x(pg, scale_v, weight_v), bar_out_v);
                llmlip_sve_add_store_strided_f32(
                    bar_x_values,
                    x_index,
                    template.x_dim,
                    svmul_f32_x(pg, scaled_bar, y_v),
                    pg
                );
                LLMLIP_ACCUM_BAR_Y(y_index, svmul_f32_x(pg, scaled_bar, x_v));
                svfloat32_t bar_weight_delta = svmul_f32_x(
                    pg,
                    svmul_f32_x(pg, scale_v, x_v),
                    svmul_f32_x(pg, y_v, bar_out_v)
                );
                bar_weight_v = svadd_f32_x(pg, bar_weight_v, bar_weight_delta);
            }
            svst1(pg, bar_w + weight_index, bar_weight_v);
        }
    }

#undef LLMLIP_ACCUM_BAR_Y

    svbool_t all = svptrue_b32();
    bar_y[0] += svaddv_f32(all, bar_y0);
    bar_y[1] += svaddv_f32(all, bar_y1);
    bar_y[2] += svaddv_f32(all, bar_y2);
    bar_y[3] += svaddv_f32(all, bar_y3);
    bar_y[4] += svaddv_f32(all, bar_y4);
    bar_y[5] += svaddv_f32(all, bar_y5);
    bar_y[6] += svaddv_f32(all, bar_y6);
    bar_y[7] += svaddv_f32(all, bar_y7);
    bar_y[8] += svaddv_f32(all, bar_y8);
    bar_y[9] += svaddv_f32(all, bar_y9);
    bar_y[10] += svaddv_f32(all, bar_y10);
    bar_y[11] += svaddv_f32(all, bar_y11);
    bar_y[12] += svaddv_f32(all, bar_y12);
    bar_y[13] += svaddv_f32(all, bar_y13);
    bar_y[14] += svaddv_f32(all, bar_y14);
    bar_y[15] += svaddv_f32(all, bar_y15);
}
