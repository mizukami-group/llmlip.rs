#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#ifdef LLMLIP_ENABLE_OMP_SVE
#include <omp.h>
#endif

static inline double llmlip_uma_silu_f64(double value) {
    if (value >= 0.0) {
        return value / (1.0 + exp(-value));
    }
    double exp_value = exp(value);
    return value * exp_value / (1.0 + exp_value);
}

static inline double llmlip_uma_silu_derivative_f64(double value) {
    double sigmoid;
    if (value >= 0.0) {
        sigmoid = 1.0 / (1.0 + exp(-value));
    } else {
        double exp_value = exp(value);
        sigmoid = exp_value / (1.0 + exp_value);
    }
    return sigmoid * (1.0 + value * (1.0 - sigmoid));
}

void llmlip_uma_spectral_scalar_gate_mlp_forward_sve_f32(
    size_t nodes,
    size_t input_node_stride,
    size_t sphere_channels,
    size_t gate_node_stride,
    const float *restrict input,
    const float *restrict weight,
    const float *restrict bias,
    float *restrict gate_pre,
    float *restrict gates
) {
    const size_t lanes = svcntw();
#ifdef LLMLIP_ENABLE_OMP_SVE
#pragma omp parallel for schedule(static)
#endif
    for (size_t node = 0; node < nodes; ++node) {
        const size_t input_base = node * input_node_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t gate = 0; gate < gate_node_stride; ++gate) {
            const size_t weight_base = gate * sphere_channels;
            svfloat32_t acc = svdup_n_f32(0.0f);
            for (size_t ch = 0; ch < sphere_channels; ch += lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)sphere_channels);
                svfloat32_t input_v = svld1(pg, input + input_base + ch);
                svfloat32_t weight_v = svld1(pg, weight + weight_base + ch);
                acc = svmla_f32_m(pg, acc, input_v, weight_v);
            }
            float value = bias[gate] + svaddv_f32(svptrue_b32(), acc);
            gate_pre[gate_base + gate] = value;
            gates[gate_base + gate] = (float)llmlip_uma_silu_f64((double)value);
        }
    }
}

void llmlip_uma_spectral_scalar_gate_mlp_vjp_sve_f32(
    size_t nodes,
    size_t input_node_stride,
    size_t sphere_channels,
    size_t gate_node_stride,
    const float *restrict gate_pre,
    const float *restrict weight,
    const float *restrict grad_gates,
    float *restrict grad_input
) {
    const size_t lanes = svcntw();
#ifdef LLMLIP_ENABLE_OMP_SVE
#pragma omp parallel for schedule(static)
#endif
    for (size_t node = 0; node < nodes; ++node) {
        const size_t input_base = node * input_node_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t gate = 0; gate < gate_node_stride; ++gate) {
            const float grad_pre = grad_gates[gate_base + gate] *
                (float)llmlip_uma_silu_derivative_f64((double)gate_pre[gate_base + gate]);
            const svfloat32_t grad_pre_v = svdup_n_f32(grad_pre);
            const size_t weight_base = gate * sphere_channels;
            for (size_t ch = 0; ch < sphere_channels; ch += lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)sphere_channels);
                svfloat32_t current = svld1(pg, grad_input + input_base + ch);
                svfloat32_t weight_v = svld1(pg, weight + weight_base + ch);
                current = svmla_f32_m(pg, current, weight_v, grad_pre_v);
                svst1(pg, grad_input + input_base + ch, current);
            }
        }
    }
}

static inline size_t llmlip_uma_l2_degree_for_coeff(size_t coeff) {
    if (coeff == 0) {
        return 0;
    }
    if (coeff < 4) {
        return 1;
    }
    return 2;
}

void llmlip_uma_spectral_so3_linear_forward_l2_sve_f32(
    size_t nodes,
    size_t channels,
    const float *restrict input,
    const float *restrict weight_in_by_out,
    const float *restrict bias,
    float *restrict out
) {
    const size_t node_stride = 9 * channels;
    const size_t lanes = svcntw();
#ifdef LLMLIP_ENABLE_OMP_SVE
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < nodes * 9; ++row) {
        const size_t node = row / 9;
        const size_t coeff = row - node * 9;
        const size_t degree = llmlip_uma_l2_degree_for_coeff(coeff);
        const float *restrict x = input + node * node_stride + coeff * channels;
        const float *restrict w = weight_in_by_out + degree * channels * channels;
        float *restrict y = out + node * node_stride + coeff * channels;
        for (size_t out_ch = 0; out_ch < channels; out_ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)out_ch, (uint64_t)channels);
            svfloat32_t acc = coeff == 0 ? svld1(pg, bias + out_ch) : svdup_n_f32(0.0f);
            for (size_t in_ch = 0; in_ch < channels; ++in_ch) {
                svfloat32_t weight_v = svld1(pg, w + in_ch * channels + out_ch);
                acc = svmla_n_f32_m(pg, acc, weight_v, x[in_ch]);
            }
            svst1(pg, y + out_ch, acc);
        }
    }
}

void llmlip_uma_spectral_so3_linear_vjp_l2_sve_f32(
    size_t nodes,
    size_t channels,
    const float *restrict weight_out_by_in,
    const float *restrict grad_out,
    float *restrict grad_input
) {
    const size_t node_stride = 9 * channels;
    const size_t lanes = svcntw();
#ifdef LLMLIP_ENABLE_OMP_SVE
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < nodes * 9; ++row) {
        const size_t node = row / 9;
        const size_t coeff = row - node * 9;
        const size_t degree = llmlip_uma_l2_degree_for_coeff(coeff);
        const float *restrict go = grad_out + node * node_stride + coeff * channels;
        const float *restrict w = weight_out_by_in + degree * channels * channels;
        float *restrict gi = grad_input + node * node_stride + coeff * channels;
        for (size_t in_ch = 0; in_ch < channels; in_ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)in_ch, (uint64_t)channels);
            svfloat32_t acc = svld1(pg, gi + in_ch);
            for (size_t out_ch = 0; out_ch < channels; ++out_ch) {
                svfloat32_t weight_v = svld1(pg, w + out_ch * channels + in_ch);
                acc = svmla_n_f32_m(pg, acc, weight_v, go[out_ch]);
            }
            svst1(pg, gi + in_ch, acc);
        }
    }
}

static inline svfloat32_t llmlip_uma_exp_approx_sve_f32(
    svbool_t pg,
    svfloat32_t x
) {
    const svfloat32_t min_x = svdup_n_f32(-88.0f);
    const svfloat32_t max_x = svdup_n_f32(88.0f);
    svfloat32_t y = svmin_f32_x(pg, svmax_f32_x(pg, x, min_x), max_x);
    svfloat32_t n_f = svrintn_f32_x(pg, svmul_n_f32_x(pg, y, 1.4426950408889634f));
    svfloat32_t r = svsub_f32_x(pg, y, svmul_n_f32_x(pg, n_f, 0.6931471805599453f));

    svfloat32_t poly = svdup_n_f32(1.0f / 120.0f);
    poly = svmla_f32_x(pg, svdup_n_f32(1.0f / 24.0f), poly, r);
    poly = svmla_f32_x(pg, svdup_n_f32(1.0f / 6.0f), poly, r);
    poly = svmla_f32_x(pg, svdup_n_f32(0.5f), poly, r);
    poly = svmla_f32_x(pg, svdup_n_f32(1.0f), poly, r);
    poly = svmla_f32_x(pg, svdup_n_f32(1.0f), poly, r);

    svint32_t n_i = svcvt_s32_f32_x(pg, n_f);
    svint32_t exponent =
        svlsl_n_s32_x(pg, svadd_n_s32_x(pg, n_i, 127), 23);
    svfloat32_t pow2 = svreinterpret_f32_s32(exponent);
    return svmul_f32_x(pg, poly, pow2);
}

static inline svfloat32_t llmlip_uma_sigmoid_sve_f32(
    svbool_t pg,
    svfloat32_t x
) {
    svfloat32_t one = svdup_n_f32(1.0f);
    svfloat32_t exp_neg = llmlip_uma_exp_approx_sve_f32(pg, svneg_f32_x(pg, x));
    return svdiv_f32_x(pg, one, svadd_f32_x(pg, one, exp_neg));
}

void llmlip_uma_silu_derivative_sve_f32(
    size_t len,
    float *values,
    float *derivatives
) {
    const size_t lanes = svcntw();
    svfloat32_t one = svdup_n_f32(1.0f);
    for (size_t index = 0; index < len; index += lanes) {
        svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)len);
        svfloat32_t x = svld1(pg, values + index);
        svfloat32_t sigmoid = llmlip_uma_sigmoid_sve_f32(pg, x);
        svst1(pg, values + index, svmul_f32_x(pg, x, sigmoid));
        svfloat32_t derivative = svmul_f32_x(
            pg,
            sigmoid,
            svadd_f32_x(
                pg,
                one,
                svmul_f32_x(pg, x, svsub_f32_x(pg, one, sigmoid))
            )
        );
        svst1(pg, derivatives + index, derivative);
    }
}

void llmlip_uma_layer_norm_silu_rows_sve_f32(
    size_t rows,
    size_t dim,
    const float *restrict input,
    const float *restrict weight,
    const float *restrict bias,
    float eps,
    float *restrict out,
    float *restrict derivatives
) {
    const size_t lanes = svcntw();
    const float dim_f = (float)dim;
    const svbool_t all = svptrue_b32();
    const svfloat32_t one = svdup_n_f32(1.0f);
    for (size_t row = 0; row < rows; ++row) {
        const float *x_row = input + row * dim;
        float *out_row = out + row * dim;
        float *deriv_row = derivatives + row * dim;

        svfloat32_t sum_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            sum_v = svadd_f32_m(pg, sum_v, svld1(pg, x_row + index));
        }
        const float mean = svaddv_f32(all, sum_v) / dim_f;

        const svfloat32_t mean_v = svdup_n_f32(mean);
        svfloat32_t variance_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            svfloat32_t centered =
                svsub_f32_x(pg, svld1(pg, x_row + index), mean_v);
            variance_v = svmla_f32_m(pg, variance_v, centered, centered);
        }
        const float variance = svaddv_f32(all, variance_v) / dim_f;
        const float inv_std = 1.0f / sqrtf(variance + eps);
        const svfloat32_t inv_std_v = svdup_n_f32(inv_std);

        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            svfloat32_t normalized = svmul_f32_x(
                pg,
                svsub_f32_x(pg, svld1(pg, x_row + index), mean_v),
                inv_std_v
            );
            svfloat32_t value = svmla_f32_x(
                pg,
                svld1(pg, bias + index),
                normalized,
                svld1(pg, weight + index)
            );
            svfloat32_t sigmoid = llmlip_uma_sigmoid_sve_f32(pg, value);
            svst1(pg, out_row + index, svmul_f32_x(pg, value, sigmoid));
            svfloat32_t derivative = svmul_f32_x(
                pg,
                sigmoid,
                svadd_f32_x(
                    pg,
                    one,
                    svmul_f32_x(pg, value, svsub_f32_x(pg, one, sigmoid))
                )
            );
            svst1(pg, deriv_row + index, derivative);
        }
    }
}

void llmlip_uma_layer_norm_silu_vjp_rows_sve_f32(
    size_t rows,
    size_t dim,
    const float *restrict input,
    const float *restrict weight,
    const float *restrict grad_out,
    const float *restrict silu_derivative,
    float eps,
    float *restrict out
) {
    const size_t lanes = svcntw();
    const float dim_f = (float)dim;
    const svbool_t all = svptrue_b32();
    for (size_t row = 0; row < rows; ++row) {
        const float *x_row = input + row * dim;
        const float *grad_row = grad_out + row * dim;
        const float *deriv_row = silu_derivative + row * dim;
        float *out_row = out + row * dim;

        svfloat32_t sum_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            sum_v = svadd_f32_m(pg, sum_v, svld1(pg, x_row + index));
        }
        const float mean = svaddv_f32(all, sum_v) / dim_f;

        const svfloat32_t mean_v = svdup_n_f32(mean);
        svfloat32_t variance_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            svfloat32_t centered =
                svsub_f32_x(pg, svld1(pg, x_row + index), mean_v);
            variance_v = svmla_f32_m(pg, variance_v, centered, centered);
        }
        const float variance = svaddv_f32(all, variance_v) / dim_f;
        const float inv_std = 1.0f / sqrtf(variance + eps);
        const svfloat32_t inv_std_v = svdup_n_f32(inv_std);

        svfloat32_t mean_grad_v = svdup_n_f32(0.0f);
        svfloat32_t mean_grad_normalized_v = svdup_n_f32(0.0f);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            svfloat32_t normalized = svmul_f32_x(
                pg,
                svsub_f32_x(pg, svld1(pg, x_row + index), mean_v),
                inv_std_v
            );
            svfloat32_t grad_normalized = svmul_f32_x(
                pg,
                svmul_f32_x(
                    pg,
                    svld1(pg, grad_row + index),
                    svld1(pg, deriv_row + index)
                ),
                svld1(pg, weight + index)
            );
            mean_grad_v = svadd_f32_m(pg, mean_grad_v, grad_normalized);
            mean_grad_normalized_v = svmla_f32_m(
                pg,
                mean_grad_normalized_v,
                grad_normalized,
                normalized
            );
        }
        const float mean_grad = svaddv_f32(all, mean_grad_v) / dim_f;
        const float mean_grad_normalized =
            svaddv_f32(all, mean_grad_normalized_v) / dim_f;
        const svfloat32_t mean_grad_v2 = svdup_n_f32(mean_grad);
        const svfloat32_t mean_grad_normalized_v2 =
            svdup_n_f32(mean_grad_normalized);

        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)index, (uint64_t)dim);
            svfloat32_t normalized = svmul_f32_x(
                pg,
                svsub_f32_x(pg, svld1(pg, x_row + index), mean_v),
                inv_std_v
            );
            svfloat32_t grad_normalized = svmul_f32_x(
                pg,
                svmul_f32_x(
                    pg,
                    svld1(pg, grad_row + index),
                    svld1(pg, deriv_row + index)
                ),
                svld1(pg, weight + index)
            );
            svfloat32_t centered_grad = svsub_f32_x(
                pg,
                svsub_f32_x(pg, grad_normalized, mean_grad_v2),
                svmul_f32_x(pg, normalized, mean_grad_normalized_v2)
            );
            svst1(
                pg,
                out_row + index,
                svmul_f32_x(pg, inv_std_v, centered_grad)
            );
        }
    }
}

void llmlip_uma_add_bias_rows_sve_f32(
    size_t rows,
    size_t cols,
    const float *restrict bias,
    float *restrict out
) {
    const size_t lanes = svcntw();
    for (size_t row = 0; row < rows; ++row) {
        const size_t base = row * cols;
        for (size_t col = 0; col < cols; col += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)col, (uint64_t)cols);
            svfloat32_t out_v = svld1(pg, out + base + col);
            svfloat32_t bias_v = svld1(pg, bias + col);
            svst1(pg, out + base + col, svadd_f32_x(pg, out_v, bias_v));
        }
    }
}

void llmlip_uma_so3_vjp_matrix_9x9_sve_f32(
    size_t items,
    size_t channels,
    const float *restrict input,
    const float *restrict grad_out,
    double *restrict grad_matrix,
    int accumulate
) {
    const size_t lanes = svcntw();
    const size_t coeffs = 9;
    const size_t input_item_stride = coeffs * channels;
    const size_t grad_out_item_stride = coeffs * channels;
    const size_t matrix_item_stride = coeffs * coeffs;

    for (size_t item = 0; item < items; ++item) {
        const float *input_item = input + item * input_item_stride;
        const float *grad_out_item = grad_out + item * grad_out_item_stride;
        double *matrix_item = grad_matrix + item * matrix_item_stride;

        for (size_t out_coeff = 0; out_coeff < coeffs; ++out_coeff) {
            const float *grad_out_row = grad_out_item + out_coeff * channels;
            svfloat32_t acc0 = svdup_n_f32(0.0f);
            svfloat32_t acc1 = svdup_n_f32(0.0f);
            svfloat32_t acc2 = svdup_n_f32(0.0f);
            svfloat32_t acc3 = svdup_n_f32(0.0f);
            svfloat32_t acc4 = svdup_n_f32(0.0f);
            svfloat32_t acc5 = svdup_n_f32(0.0f);
            svfloat32_t acc6 = svdup_n_f32(0.0f);
            svfloat32_t acc7 = svdup_n_f32(0.0f);
            svfloat32_t acc8 = svdup_n_f32(0.0f);

            for (size_t ch = 0; ch < channels; ch += lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
                svfloat32_t go = svld1(pg, grad_out_row + ch);
                acc0 = svmla_f32_m(pg, acc0, go, svld1(pg, input_item + 0 * channels + ch));
                acc1 = svmla_f32_m(pg, acc1, go, svld1(pg, input_item + 1 * channels + ch));
                acc2 = svmla_f32_m(pg, acc2, go, svld1(pg, input_item + 2 * channels + ch));
                acc3 = svmla_f32_m(pg, acc3, go, svld1(pg, input_item + 3 * channels + ch));
                acc4 = svmla_f32_m(pg, acc4, go, svld1(pg, input_item + 4 * channels + ch));
                acc5 = svmla_f32_m(pg, acc5, go, svld1(pg, input_item + 5 * channels + ch));
                acc6 = svmla_f32_m(pg, acc6, go, svld1(pg, input_item + 6 * channels + ch));
                acc7 = svmla_f32_m(pg, acc7, go, svld1(pg, input_item + 7 * channels + ch));
                acc8 = svmla_f32_m(pg, acc8, go, svld1(pg, input_item + 8 * channels + ch));
            }

            double *row = matrix_item + out_coeff * coeffs;
            const double sums[9] = {
                (double)svaddv_f32(svptrue_b32(), acc0),
                (double)svaddv_f32(svptrue_b32(), acc1),
                (double)svaddv_f32(svptrue_b32(), acc2),
                (double)svaddv_f32(svptrue_b32(), acc3),
                (double)svaddv_f32(svptrue_b32(), acc4),
                (double)svaddv_f32(svptrue_b32(), acc5),
                (double)svaddv_f32(svptrue_b32(), acc6),
                (double)svaddv_f32(svptrue_b32(), acc7),
                (double)svaddv_f32(svptrue_b32(), acc8),
            };
            if (accumulate) {
                for (size_t in_coeff = 0; in_coeff < coeffs; ++in_coeff) {
                    row[in_coeff] += sums[in_coeff];
                }
            } else {
                for (size_t in_coeff = 0; in_coeff < coeffs; ++in_coeff) {
                    row[in_coeff] = sums[in_coeff];
                }
            }
        }
    }
}

void llmlip_uma_so3_vjp_matrix_9x9_order_packed_sve_f32(
    size_t items,
    size_t channels,
    const float *restrict input,
    const float *restrict grad_out,
    double *restrict grad_matrix,
    int accumulate
) {
    const size_t lanes = svcntw();
    const size_t coeffs = 9;
    const size_t input_item_stride = coeffs * channels;
    const size_t grad_out_item_stride = coeffs * channels;
    const size_t matrix_item_stride = coeffs * coeffs;
    const size_t packed_to_l_major[9] = {0, 2, 6, 3, 7, 1, 5, 8, 4};

    for (size_t item = 0; item < items; ++item) {
        const float *input_item = input + item * input_item_stride;
        const float *grad_out_item = grad_out + item * grad_out_item_stride;
        double *matrix_item = grad_matrix + item * matrix_item_stride;

        for (size_t out_coeff = 0; out_coeff < coeffs; ++out_coeff) {
            const float *grad_out_row = grad_out_item + out_coeff * channels;
            svfloat32_t acc0 = svdup_n_f32(0.0f);
            svfloat32_t acc1 = svdup_n_f32(0.0f);
            svfloat32_t acc2 = svdup_n_f32(0.0f);
            svfloat32_t acc3 = svdup_n_f32(0.0f);
            svfloat32_t acc4 = svdup_n_f32(0.0f);
            svfloat32_t acc5 = svdup_n_f32(0.0f);
            svfloat32_t acc6 = svdup_n_f32(0.0f);
            svfloat32_t acc7 = svdup_n_f32(0.0f);
            svfloat32_t acc8 = svdup_n_f32(0.0f);

            for (size_t ch = 0; ch < channels; ch += lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
                svfloat32_t go = svld1(pg, grad_out_row + ch);
                acc0 = svmla_f32_m(pg, acc0, go, svld1(pg, input_item + 0 * channels + ch));
                acc1 = svmla_f32_m(pg, acc1, go, svld1(pg, input_item + 1 * channels + ch));
                acc2 = svmla_f32_m(pg, acc2, go, svld1(pg, input_item + 2 * channels + ch));
                acc3 = svmla_f32_m(pg, acc3, go, svld1(pg, input_item + 3 * channels + ch));
                acc4 = svmla_f32_m(pg, acc4, go, svld1(pg, input_item + 4 * channels + ch));
                acc5 = svmla_f32_m(pg, acc5, go, svld1(pg, input_item + 5 * channels + ch));
                acc6 = svmla_f32_m(pg, acc6, go, svld1(pg, input_item + 6 * channels + ch));
                acc7 = svmla_f32_m(pg, acc7, go, svld1(pg, input_item + 7 * channels + ch));
                acc8 = svmla_f32_m(pg, acc8, go, svld1(pg, input_item + 8 * channels + ch));
            }

            double *row = matrix_item + out_coeff * coeffs;
            const double sums[9] = {
                (double)svaddv_f32(svptrue_b32(), acc0),
                (double)svaddv_f32(svptrue_b32(), acc1),
                (double)svaddv_f32(svptrue_b32(), acc2),
                (double)svaddv_f32(svptrue_b32(), acc3),
                (double)svaddv_f32(svptrue_b32(), acc4),
                (double)svaddv_f32(svptrue_b32(), acc5),
                (double)svaddv_f32(svptrue_b32(), acc6),
                (double)svaddv_f32(svptrue_b32(), acc7),
                (double)svaddv_f32(svptrue_b32(), acc8),
            };
            if (accumulate) {
                for (size_t packed_coeff = 0; packed_coeff < coeffs; ++packed_coeff) {
                    row[packed_to_l_major[packed_coeff]] += sums[packed_coeff];
                }
            } else {
                for (size_t in_coeff = 0; in_coeff < coeffs; ++in_coeff) {
                    row[in_coeff] = 0.0;
                }
                for (size_t packed_coeff = 0; packed_coeff < coeffs; ++packed_coeff) {
                    row[packed_to_l_major[packed_coeff]] = sums[packed_coeff];
                }
            }
        }
    }
}

void llmlip_uma_so3_vjp_input_write_9x9_sve_f32(
    size_t items,
    size_t channels,
    const float *restrict matrix,
    const float *restrict grad_out,
    float *restrict grad_input
) {
    const size_t lanes = svcntw();
    const size_t coeffs = 9;
    const size_t feature_item_stride = coeffs * channels;
    const size_t matrix_item_stride = coeffs * coeffs;

    for (size_t item = 0; item < items; ++item) {
        const float *matrix_item = matrix + item * matrix_item_stride;
        const float *grad_out_item = grad_out + item * feature_item_stride;
        float *grad_input_item = grad_input + item * feature_item_stride;

        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t go0 = svld1(pg, grad_out_item + 0 * channels + ch);
            svfloat32_t go1 = svld1(pg, grad_out_item + 1 * channels + ch);
            svfloat32_t go2 = svld1(pg, grad_out_item + 2 * channels + ch);
            svfloat32_t go3 = svld1(pg, grad_out_item + 3 * channels + ch);
            svfloat32_t go4 = svld1(pg, grad_out_item + 4 * channels + ch);
            svfloat32_t go5 = svld1(pg, grad_out_item + 5 * channels + ch);
            svfloat32_t go6 = svld1(pg, grad_out_item + 6 * channels + ch);
            svfloat32_t go7 = svld1(pg, grad_out_item + 7 * channels + ch);
            svfloat32_t go8 = svld1(pg, grad_out_item + 8 * channels + ch);

            for (size_t in_coeff = 0; in_coeff < coeffs; ++in_coeff) {
                svfloat32_t acc =
                    svmul_n_f32_x(pg, go0, matrix_item[0 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go1, matrix_item[1 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go2, matrix_item[2 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go3, matrix_item[3 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go4, matrix_item[4 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go5, matrix_item[5 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go6, matrix_item[6 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go7, matrix_item[7 * coeffs + in_coeff]);
                acc = svmla_n_f32_x(pg, acc, go8, matrix_item[8 * coeffs + in_coeff]);
                svst1(pg, grad_input_item + in_coeff * channels + ch, acc);
            }
        }
    }
}

void llmlip_uma_so2_m0_gate_sve_f32(
    size_t nodes,
    size_t channels,
    size_t m0_out_dim,
    size_t conv1_out_node_stride,
    size_t gate_node_stride,
    size_t conv2_m0_in_node_stride,
    const float *conv1_m0_dense_out,
    float *conv1_out_l_major,
    float *gate_sigmoids,
    float *scalar_silu_derivatives,
    float *conv2_m0_input
) {
    const size_t lanes = svcntw();
    for (size_t node = 0; node < nodes; ++node) {
        const size_t dense_base = node * m0_out_dim;
        const size_t conv1_base = node * conv1_out_node_stride;
        const size_t gate_base = node * gate_node_stride;
        const size_t conv2_in_base = node * conv2_m0_in_node_stride;
        const size_t deriv_base = node * channels;

        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t one = svdup_n_f32(1.0f);

            svfloat32_t gate1_raw = svld1(pg, conv1_m0_dense_out + dense_base + ch);
            svfloat32_t gate1 = llmlip_uma_sigmoid_sve_f32(pg, gate1_raw);
            svst1(pg, gate_sigmoids + gate_base + ch, gate1);

            svfloat32_t gate2_raw =
                svld1(pg, conv1_m0_dense_out + dense_base + channels + ch);
            svfloat32_t gate2 = llmlip_uma_sigmoid_sve_f32(pg, gate2_raw);
            svst1(pg, gate_sigmoids + gate_base + channels + ch, gate2);

            svfloat32_t scalar =
                svld1(pg, conv1_m0_dense_out + dense_base + 2 * channels + ch);
            svst1(pg, conv1_out_l_major + conv1_base + ch, scalar);
            svfloat32_t scalar_sigmoid = llmlip_uma_sigmoid_sve_f32(pg, scalar);
            svst1(
                pg,
                conv2_m0_input + conv2_in_base + ch,
                svmul_f32_x(pg, scalar, scalar_sigmoid)
            );
            svfloat32_t scalar_derivative = svmul_f32_x(
                pg,
                scalar_sigmoid,
                svadd_f32_x(
                    pg,
                    one,
                    svmul_f32_x(pg, scalar, svsub_f32_x(pg, one, scalar_sigmoid))
                )
            );
            svst1(pg, scalar_silu_derivatives + deriv_base + ch, scalar_derivative);

            svfloat32_t l1_val =
                svld1(pg, conv1_m0_dense_out + dense_base + 3 * channels + ch);
            svst1(pg, conv1_out_l_major + conv1_base + channels + ch, l1_val);
            svst1(
                pg,
                conv2_m0_input + conv2_in_base + channels + ch,
                svmul_f32_x(pg, l1_val, gate1)
            );

            svfloat32_t l2_val =
                svld1(pg, conv1_m0_dense_out + dense_base + 4 * channels + ch);
            svst1(pg, conv1_out_l_major + conv1_base + 2 * channels + ch, l2_val);
            svst1(
                pg,
                conv2_m0_input + conv2_in_base + 2 * channels + ch,
                svmul_f32_x(pg, l2_val, gate2)
            );
        }
    }
}

void llmlip_uma_so2_m0_gather_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    const float *restrict input_l_major,
    const float *restrict radial,
    float *restrict conv1_m0_input
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 3 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);

            svfloat32_t src0 = svld1(pg, input_l_major + in_base + ch);
            svfloat32_t rad0 = svld1(pg, radial + rad_base + ch);
            svst1(
                pg,
                conv1_m0_input + blk_base + ch,
                svmul_f32_x(pg, src0, rad0)
            );

            svfloat32_t src2 =
                svld1(pg, input_l_major + in_base + 2 * conv1_in_channels + ch);
            svfloat32_t rad1 =
                svld1(pg, radial + rad_base + conv1_in_channels + ch);
            svst1(
                pg,
                conv1_m0_input + blk_base + conv1_in_channels + ch,
                svmul_f32_x(pg, src2, rad1)
            );

            svfloat32_t src6 =
                svld1(pg, input_l_major + in_base + 6 * conv1_in_channels + ch);
            svfloat32_t rad2 =
                svld1(pg, radial + rad_base + 2 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_m0_input + blk_base + 2 * conv1_in_channels + ch,
                svmul_f32_x(pg, src6, rad2)
            );
        }
    }
}

void llmlip_uma_so2_m0_gather_grad_conv2_sve_f32(
    size_t nodes,
    size_t channels,
    size_t out_node_stride,
    const float *restrict grad_conv2_out_l_major,
    float *restrict conv2_m0_grad_out
) {
    const size_t lanes = svcntw();
    const size_t dst_stride = 3 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t src_base = node * out_node_stride;
        const size_t dst_base = node * dst_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svst1(
                pg,
                conv2_m0_grad_out + dst_base + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + ch)
            );
            svst1(
                pg,
                conv2_m0_grad_out + dst_base + channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 2 * channels + ch)
            );
            svst1(
                pg,
                conv2_m0_grad_out + dst_base + 2 * channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 6 * channels + ch)
            );
        }
    }
}

void llmlip_uma_so2_order1_gather_grad_conv2_sve_f32(
    size_t nodes,
    size_t channels,
    size_t out_node_stride,
    const float *restrict grad_conv2_out_l_major,
    float *restrict conv2_blk_grad
) {
    const size_t lanes = svcntw();
    const size_t dst_stride = 4 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t src_base = node * out_node_stride;
        const size_t dst_base = node * dst_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svst1(
                pg,
                conv2_blk_grad + dst_base + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 3 * channels + ch)
            );
            svst1(
                pg,
                conv2_blk_grad + dst_base + channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 7 * channels + ch)
            );
            svst1(
                pg,
                conv2_blk_grad + dst_base + 2 * channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + channels + ch)
            );
            svst1(
                pg,
                conv2_blk_grad + dst_base + 3 * channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 5 * channels + ch)
            );
        }
    }
}

void llmlip_uma_so2_order2_gather_grad_conv2_sve_f32(
    size_t nodes,
    size_t channels,
    size_t out_node_stride,
    const float *restrict grad_conv2_out_l_major,
    float *restrict conv2_blk_grad
) {
    const size_t lanes = svcntw();
    const size_t dst_stride = 2 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t src_base = node * out_node_stride;
        const size_t dst_base = node * dst_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svst1(
                pg,
                conv2_blk_grad + dst_base + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 8 * channels + ch)
            );
            svst1(
                pg,
                conv2_blk_grad + dst_base + channels + ch,
                svld1(pg, grad_conv2_out_l_major + src_base + 4 * channels + ch)
            );
        }
    }
}

void llmlip_uma_so2_order1_gather_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    size_t radial_offset,
    const float *restrict input_l_major,
    const float *restrict radial,
    float *restrict conv1_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 4 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride + radial_offset;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);
            svfloat32_t radial0 = svld1(pg, radial + rad_base + ch);
            svfloat32_t radial1 =
                svld1(pg, radial + rad_base + conv1_in_channels + ch);

            svfloat32_t src3 =
                svld1(pg, input_l_major + in_base + 3 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + ch,
                svmul_f32_x(pg, src3, radial0)
            );

            svfloat32_t src7 =
                svld1(pg, input_l_major + in_base + 7 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + conv1_in_channels + ch,
                svmul_f32_x(pg, src7, radial1)
            );

            svfloat32_t src1 =
                svld1(pg, input_l_major + in_base + conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + 2 * conv1_in_channels + ch,
                svmul_f32_x(pg, src1, radial0)
            );

            svfloat32_t src5 =
                svld1(pg, input_l_major + in_base + 5 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + 3 * conv1_in_channels + ch,
                svmul_f32_x(pg, src5, radial1)
            );
        }
    }
}

void llmlip_uma_so2_order2_gather_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    size_t radial_offset,
    const float *restrict input_l_major,
    const float *restrict radial,
    float *restrict conv1_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 2 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride + radial_offset;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);
            svfloat32_t radial0 = svld1(pg, radial + rad_base + ch);

            svfloat32_t src8 =
                svld1(pg, input_l_major + in_base + 8 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + ch,
                svmul_f32_x(pg, src8, radial0)
            );

            svfloat32_t src4 =
                svld1(pg, input_l_major + in_base + 4 * conv1_in_channels + ch);
            svst1(
                pg,
                conv1_blk_in + blk_base + conv1_in_channels + ch,
                svmul_f32_x(pg, src4, radial0)
            );
        }
    }
}

void llmlip_uma_so2_order1_gate_sve_f32(
    size_t nodes,
    size_t channels,
    size_t gate_node_stride,
    const float *restrict conv1_blk_out,
    const float *restrict gate_sigmoids,
    float *restrict conv2_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 4 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t blk_base = node * blk_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate1 = svld1(pg, gate_sigmoids + gate_base + ch);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);

            svfloat32_t v3 = svld1(pg, conv1_blk_out + blk_base + ch);
            svst1(pg, conv2_blk_in + blk_base + ch, svmul_f32_x(pg, v3, gate1));
            svfloat32_t v7 =
                svld1(pg, conv1_blk_out + blk_base + channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + channels + ch,
                svmul_f32_x(pg, v7, gate2)
            );
            svfloat32_t v1 =
                svld1(pg, conv1_blk_out + blk_base + 2 * channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + 2 * channels + ch,
                svmul_f32_x(pg, v1, gate1)
            );
            svfloat32_t v5 =
                svld1(pg, conv1_blk_out + blk_base + 3 * channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + 3 * channels + ch,
                svmul_f32_x(pg, v5, gate2)
            );
        }
    }
}

void llmlip_uma_so2_order1_gate_lmajor_sve_f32(
    size_t nodes,
    size_t channels,
    size_t out_node_stride,
    size_t gate_node_stride,
    const float *restrict conv1_out_l_major,
    const float *restrict gate_sigmoids,
    float *restrict conv2_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 4 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t src_base = node * out_node_stride + 3 * channels;
        const size_t blk_base = node * blk_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate1 = svld1(pg, gate_sigmoids + gate_base + ch);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);

            svfloat32_t v3 = svld1(pg, conv1_out_l_major + src_base + ch);
            svst1(pg, conv2_blk_in + blk_base + ch, svmul_f32_x(pg, v3, gate1));
            svfloat32_t v7 =
                svld1(pg, conv1_out_l_major + src_base + channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + channels + ch,
                svmul_f32_x(pg, v7, gate2)
            );
            svfloat32_t v1 =
                svld1(pg, conv1_out_l_major + src_base + 2 * channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + 2 * channels + ch,
                svmul_f32_x(pg, v1, gate1)
            );
            svfloat32_t v5 =
                svld1(pg, conv1_out_l_major + src_base + 3 * channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + 3 * channels + ch,
                svmul_f32_x(pg, v5, gate2)
            );
        }
    }
}

void llmlip_uma_so2_order2_gate_sve_f32(
    size_t nodes,
    size_t channels,
    size_t gate_node_stride,
    const float *restrict conv1_blk_out,
    const float *restrict gate_sigmoids,
    float *restrict conv2_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 2 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t blk_base = node * blk_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t v8 = svld1(pg, conv1_blk_out + blk_base + ch);
            svst1(pg, conv2_blk_in + blk_base + ch, svmul_f32_x(pg, v8, gate2));
            svfloat32_t v4 =
                svld1(pg, conv1_blk_out + blk_base + channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + channels + ch,
                svmul_f32_x(pg, v4, gate2)
            );
        }
    }
}

void llmlip_uma_so2_order2_gate_lmajor_sve_f32(
    size_t nodes,
    size_t channels,
    size_t out_node_stride,
    size_t gate_node_stride,
    const float *restrict conv1_out_l_major,
    const float *restrict gate_sigmoids,
    float *restrict conv2_blk_in
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 2 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t src_base = node * out_node_stride + 7 * channels;
        const size_t blk_base = node * blk_stride;
        const size_t gate_base = node * gate_node_stride;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t v8 = svld1(pg, conv1_out_l_major + src_base + ch);
            svst1(pg, conv2_blk_in + blk_base + ch, svmul_f32_x(pg, v8, gate2));
            svfloat32_t v4 =
                svld1(pg, conv1_out_l_major + src_base + channels + ch);
            svst1(
                pg,
                conv2_blk_in + blk_base + channels + ch,
                svmul_f32_x(pg, v4, gate2)
            );
        }
    }
}

void llmlip_uma_so2_order1_gate_vjp_sve_f32(
    size_t nodes,
    size_t channels,
    size_t m0_out_dim,
    size_t out_node_stride,
    size_t gate_node_stride,
    const float *restrict act_blk_grad,
    const float *restrict conv1_out_l_major,
    const float *restrict gate_sigmoids,
    float *restrict conv1_blk_grad,
    float *restrict m0_grad
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 4 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t blk = node * blk_stride;
        const size_t conv1_base = node * out_node_stride + 3 * channels;
        const size_t gate_base = node * gate_node_stride;
        const size_t full_grad_base = node * m0_out_dim;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate1 = svld1(pg, gate_sigmoids + gate_base + ch);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t grad3 = svld1(pg, act_blk_grad + blk + ch);
            svfloat32_t grad7 =
                svld1(pg, act_blk_grad + blk + channels + ch);
            svfloat32_t grad1 =
                svld1(pg, act_blk_grad + blk + 2 * channels + ch);
            svfloat32_t grad5 =
                svld1(pg, act_blk_grad + blk + 3 * channels + ch);

            svst1(
                pg,
                conv1_blk_grad + blk + ch,
                svmul_f32_x(pg, grad3, gate1)
            );
            svst1(
                pg,
                conv1_blk_grad + blk + channels + ch,
                svmul_f32_x(pg, grad7, gate2)
            );
            svst1(
                pg,
                conv1_blk_grad + blk + 2 * channels + ch,
                svmul_f32_x(pg, grad1, gate1)
            );
            svst1(
                pg,
                conv1_blk_grad + blk + 3 * channels + ch,
                svmul_f32_x(pg, grad5, gate2)
            );

            svfloat32_t v3 = svld1(pg, conv1_out_l_major + conv1_base + ch);
            svfloat32_t v7 =
                svld1(pg, conv1_out_l_major + conv1_base + channels + ch);
            svfloat32_t v1 =
                svld1(pg, conv1_out_l_major + conv1_base + 2 * channels + ch);
            svfloat32_t v5 =
                svld1(pg, conv1_out_l_major + conv1_base + 3 * channels + ch);

            svfloat32_t gate1_acc = svld1(pg, m0_grad + full_grad_base + ch);
            gate1_acc = svmla_f32_x(pg, gate1_acc, grad3, v3);
            gate1_acc = svmla_f32_x(pg, gate1_acc, grad1, v1);
            svst1(pg, m0_grad + full_grad_base + ch, gate1_acc);

            svfloat32_t gate2_acc =
                svld1(pg, m0_grad + full_grad_base + channels + ch);
            gate2_acc = svmla_f32_x(pg, gate2_acc, grad7, v7);
            gate2_acc = svmla_f32_x(pg, gate2_acc, grad5, v5);
            svst1(pg, m0_grad + full_grad_base + channels + ch, gate2_acc);
        }
    }
}

void llmlip_uma_so2_order2_gate_vjp_sve_f32(
    size_t nodes,
    size_t channels,
    size_t m0_out_dim,
    size_t out_node_stride,
    size_t gate_node_stride,
    const float *restrict act_blk_grad,
    const float *restrict conv1_out_l_major,
    const float *restrict gate_sigmoids,
    float *restrict conv1_blk_grad,
    float *restrict m0_grad
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 2 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t blk = node * blk_stride;
        const size_t conv1_base = node * out_node_stride + 7 * channels;
        const size_t gate_base = node * gate_node_stride;
        const size_t full_grad_base = node * m0_out_dim;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t grad8 = svld1(pg, act_blk_grad + blk + ch);
            svfloat32_t grad4 =
                svld1(pg, act_blk_grad + blk + channels + ch);

            svst1(
                pg,
                conv1_blk_grad + blk + ch,
                svmul_f32_x(pg, grad8, gate2)
            );
            svst1(
                pg,
                conv1_blk_grad + blk + channels + ch,
                svmul_f32_x(pg, grad4, gate2)
            );

            svfloat32_t v8 = svld1(pg, conv1_out_l_major + conv1_base + ch);
            svfloat32_t v4 =
                svld1(pg, conv1_out_l_major + conv1_base + channels + ch);
            svfloat32_t gate2_acc =
                svld1(pg, m0_grad + full_grad_base + channels + ch);
            gate2_acc = svmla_f32_x(pg, gate2_acc, grad8, v8);
            gate2_acc = svmla_f32_x(pg, gate2_acc, grad4, v4);
            svst1(pg, m0_grad + full_grad_base + channels + ch, gate2_acc);
        }
    }
}

void llmlip_uma_so2_m0_gate_vjp_sve_f32(
    size_t nodes,
    size_t channels,
    size_t m0_out_dim,
    size_t out_node_stride,
    size_t gate_node_stride,
    const float *restrict activated_m0_grad,
    const float *restrict conv1_out_l_major,
    const float *restrict gate_sigmoids,
    const float *restrict scalar_silu_derivatives,
    float *restrict conv1_m0_full_grad_out
) {
    const size_t lanes = svcntw();
    const size_t act_stride = 3 * channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t act_base = node * act_stride;
        const size_t conv1_base = node * out_node_stride;
        const size_t gate_base = node * gate_node_stride;
        const size_t deriv_base = node * channels;
        const size_t grad_base = node * m0_out_dim;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);

            svfloat32_t grad_scalar = svld1(pg, activated_m0_grad + act_base + ch);
            svfloat32_t grad_l1 =
                svld1(pg, activated_m0_grad + act_base + channels + ch);
            svfloat32_t grad_l2 =
                svld1(pg, activated_m0_grad + act_base + 2 * channels + ch);
            svfloat32_t gate1 = svld1(pg, gate_sigmoids + gate_base + ch);
            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t conv1_l1 =
                svld1(pg, conv1_out_l_major + conv1_base + channels + ch);
            svfloat32_t conv1_l2 =
                svld1(pg, conv1_out_l_major + conv1_base + 2 * channels + ch);
            svfloat32_t scalar_deriv =
                svld1(pg, scalar_silu_derivatives + deriv_base + ch);

            float *gate1_grad = conv1_m0_full_grad_out + grad_base + ch;
            float *gate2_grad = conv1_m0_full_grad_out + grad_base + channels + ch;
            float *scalar_grad =
                conv1_m0_full_grad_out + grad_base + 2 * channels + ch;
            float *l1_grad = conv1_m0_full_grad_out + grad_base + 3 * channels + ch;
            float *l2_grad = conv1_m0_full_grad_out + grad_base + 4 * channels + ch;

            svst1(
                pg,
                gate1_grad,
                svmla_f32_x(pg, svld1(pg, gate1_grad), grad_l1, conv1_l1)
            );
            svst1(
                pg,
                gate2_grad,
                svmla_f32_x(pg, svld1(pg, gate2_grad), grad_l2, conv1_l2)
            );
            svst1(
                pg,
                scalar_grad,
                svmla_f32_x(pg, svld1(pg, scalar_grad), grad_scalar, scalar_deriv)
            );
            svst1(
                pg,
                l1_grad,
                svmla_f32_x(pg, svld1(pg, l1_grad), grad_l1, gate1)
            );
            svst1(
                pg,
                l2_grad,
                svmla_f32_x(pg, svld1(pg, l2_grad), grad_l2, gate2)
            );
        }
    }
}

void llmlip_uma_so2_m0_gate_derivative_sve_f32(
    size_t nodes,
    size_t channels,
    size_t m0_out_dim,
    size_t gate_node_stride,
    const float *restrict gate_sigmoids,
    float *restrict conv1_m0_full_grad_out
) {
    const size_t lanes = svcntw();
    svfloat32_t one = svdup_n_f32(1.0f);
    for (size_t node = 0; node < nodes; ++node) {
        const size_t gate_base = node * gate_node_stride;
        const size_t grad_base = node * m0_out_dim;
        for (size_t ch = 0; ch < channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)channels);

            svfloat32_t gate1 = svld1(pg, gate_sigmoids + gate_base + ch);
            svfloat32_t grad1 = svld1(pg, conv1_m0_full_grad_out + grad_base + ch);
            grad1 = svmul_f32_x(
                pg,
                grad1,
                svmul_f32_x(pg, gate1, svsub_f32_x(pg, one, gate1))
            );
            svst1(pg, conv1_m0_full_grad_out + grad_base + ch, grad1);

            svfloat32_t gate2 =
                svld1(pg, gate_sigmoids + gate_base + channels + ch);
            svfloat32_t grad2 =
                svld1(pg, conv1_m0_full_grad_out + grad_base + channels + ch);
            grad2 = svmul_f32_x(
                pg,
                grad2,
                svmul_f32_x(pg, gate2, svsub_f32_x(pg, one, gate2))
            );
            svst1(pg, conv1_m0_full_grad_out + grad_base + channels + ch, grad2);
        }
    }
}

void llmlip_uma_so2_m0_radial_vjp_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    const float *restrict conv1_m0_grad_mod,
    const float *restrict conv1_input_l_major,
    const float *restrict conv1_radial,
    float *restrict grad_conv1_input_l_major,
    float *restrict grad_conv1_radial
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 3 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);

            svfloat32_t g0 = svld1(pg, conv1_m0_grad_mod + blk_base + ch);
            svfloat32_t inp0 = svld1(pg, conv1_input_l_major + in_base + ch);
            svfloat32_t r0 = svld1(pg, conv1_radial + rad_base + ch);
            svst1(
                pg,
                grad_conv1_input_l_major + in_base + ch,
                svmul_f32_x(pg, g0, r0)
            );
            svst1(
                pg,
                grad_conv1_radial + rad_base + ch,
                svmul_f32_x(pg, g0, inp0)
            );

            svfloat32_t g1 =
                svld1(pg, conv1_m0_grad_mod + blk_base + conv1_in_channels + ch);
            svfloat32_t inp1 =
                svld1(pg, conv1_input_l_major + in_base + 2 * conv1_in_channels + ch);
            svfloat32_t r1 =
                svld1(pg, conv1_radial + rad_base + conv1_in_channels + ch);
            svst1(
                pg,
                grad_conv1_input_l_major + in_base + 2 * conv1_in_channels + ch,
                svmul_f32_x(pg, g1, r1)
            );
            svst1(
                pg,
                grad_conv1_radial + rad_base + conv1_in_channels + ch,
                svmul_f32_x(pg, g1, inp1)
            );

            svfloat32_t g2 =
                svld1(pg, conv1_m0_grad_mod + blk_base + 2 * conv1_in_channels + ch);
            svfloat32_t inp2 =
                svld1(pg, conv1_input_l_major + in_base + 6 * conv1_in_channels + ch);
            svfloat32_t r2 =
                svld1(pg, conv1_radial + rad_base + 2 * conv1_in_channels + ch);
            svst1(
                pg,
                grad_conv1_input_l_major + in_base + 6 * conv1_in_channels + ch,
                svmul_f32_x(pg, g2, r2)
            );
            svst1(
                pg,
                grad_conv1_radial + rad_base + 2 * conv1_in_channels + ch,
                svmul_f32_x(pg, g2, inp2)
            );
        }
    }
}

void llmlip_uma_so2_order1_radial_vjp_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    size_t radial_offset,
    const float *restrict conv1_blk_mod,
    const float *restrict conv1_input_l_major,
    const float *restrict conv1_radial,
    float *restrict grad_conv1_input_l_major,
    float *restrict grad_conv1_radial
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 4 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride + radial_offset;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);
            svfloat32_t radial0 = svld1(pg, conv1_radial + rad_base + ch);
            svfloat32_t radial1 =
                svld1(pg, conv1_radial + rad_base + conv1_in_channels + ch);
            svfloat32_t g0 = svld1(pg, conv1_blk_mod + blk_base + ch);
            svfloat32_t g1 =
                svld1(pg, conv1_blk_mod + blk_base + conv1_in_channels + ch);
            svfloat32_t g2 =
                svld1(pg, conv1_blk_mod + blk_base + 2 * conv1_in_channels + ch);
            svfloat32_t g3 =
                svld1(pg, conv1_blk_mod + blk_base + 3 * conv1_in_channels + ch);
            svfloat32_t inp0 =
                svld1(pg, conv1_input_l_major + in_base + 3 * conv1_in_channels + ch);
            svfloat32_t inp1 =
                svld1(pg, conv1_input_l_major + in_base + 7 * conv1_in_channels + ch);
            svfloat32_t inp2 =
                svld1(pg, conv1_input_l_major + in_base + conv1_in_channels + ch);
            svfloat32_t inp3 =
                svld1(pg, conv1_input_l_major + in_base + 5 * conv1_in_channels + ch);

            float *gi0 = grad_conv1_input_l_major + in_base + 3 * conv1_in_channels + ch;
            float *gi1 = grad_conv1_input_l_major + in_base + 7 * conv1_in_channels + ch;
            float *gi2 = grad_conv1_input_l_major + in_base + conv1_in_channels + ch;
            float *gi3 = grad_conv1_input_l_major + in_base + 5 * conv1_in_channels + ch;
            svst1(pg, gi0, svmul_f32_x(pg, g0, radial0));
            svst1(pg, gi1, svmul_f32_x(pg, g1, radial1));
            svst1(pg, gi2, svmul_f32_x(pg, g2, radial0));
            svst1(pg, gi3, svmul_f32_x(pg, g3, radial1));

            float *gr0 = grad_conv1_radial + rad_base + ch;
            float *gr1 = grad_conv1_radial + rad_base + conv1_in_channels + ch;
            svst1(
                pg,
                gr0,
                svmla_f32_x(pg, svmul_f32_x(pg, g0, inp0), g2, inp2)
            );
            svst1(
                pg,
                gr1,
                svmla_f32_x(pg, svmul_f32_x(pg, g1, inp1), g3, inp3)
            );
        }
    }
}

void llmlip_uma_so2_order2_radial_vjp_sve_f32(
    size_t nodes,
    size_t conv1_in_channels,
    size_t input_node_stride,
    size_t radial_node_stride,
    size_t radial_offset,
    const float *restrict conv1_blk_mod,
    const float *restrict conv1_input_l_major,
    const float *restrict conv1_radial,
    float *restrict grad_conv1_input_l_major,
    float *restrict grad_conv1_radial
) {
    const size_t lanes = svcntw();
    const size_t blk_stride = 2 * conv1_in_channels;
    for (size_t node = 0; node < nodes; ++node) {
        const size_t in_base = node * input_node_stride;
        const size_t rad_base = node * radial_node_stride + radial_offset;
        const size_t blk_base = node * blk_stride;
        for (size_t ch = 0; ch < conv1_in_channels; ch += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)ch, (uint64_t)conv1_in_channels);
            svfloat32_t radial0 = svld1(pg, conv1_radial + rad_base + ch);
            svfloat32_t g0 = svld1(pg, conv1_blk_mod + blk_base + ch);
            svfloat32_t g1 =
                svld1(pg, conv1_blk_mod + blk_base + conv1_in_channels + ch);
            svfloat32_t inp0 =
                svld1(pg, conv1_input_l_major + in_base + 8 * conv1_in_channels + ch);
            svfloat32_t inp1 =
                svld1(pg, conv1_input_l_major + in_base + 4 * conv1_in_channels + ch);

            float *gi0 = grad_conv1_input_l_major + in_base + 8 * conv1_in_channels + ch;
            float *gi1 = grad_conv1_input_l_major + in_base + 4 * conv1_in_channels + ch;
            svst1(pg, gi0, svmul_f32_x(pg, g0, radial0));
            svst1(pg, gi1, svmul_f32_x(pg, g1, radial0));

            float *gr0 = grad_conv1_radial + rad_base + ch;
            svst1(
                pg,
                gr0,
                svmla_f32_x(pg, svmul_f32_x(pg, g0, inp0), g1, inp1)
            );
        }
    }
}

#ifdef LLMLIP_ENABLE_ARMPL_AMATH
/* ---------------------------------------------------------------------------
 * f64 SVE2 activation kernels (Graviton / Neoverse-V2).
 *
 * The f64 UMA path otherwise runs scalar libm transcendentals; these kernels
 * vectorize the radial-MLP LayerNorm+SiLU using ArmPL's libamath SVE `exp`
 * (`armpl_svexp_f64_u10_x`, 1.0 ULP) so the result matches the scalar libm
 * reference to ~1 ULP. Argument order is (value, predicate).
 * ------------------------------------------------------------------------- */
svfloat64_t armpl_svexp_f64_u10_x(svfloat64_t, svbool_t);

static inline svfloat64_t llmlip_uma_sigmoid_sve_f64(svbool_t pg, svfloat64_t x) {
    const svfloat64_t one = svdup_n_f64(1.0);
    svfloat64_t exp_neg = armpl_svexp_f64_u10_x(svneg_f64_x(pg, x), pg);
    return svdiv_f64_x(pg, one, svadd_f64_x(pg, one, exp_neg));
}

void llmlip_uma_layer_norm_silu_rows_sve_f64(
    size_t rows,
    size_t dim,
    const double *restrict input,
    const double *restrict weight,
    const double *restrict bias,
    double eps,
    double *restrict out,
    double *restrict derivatives
) {
    const size_t lanes = svcntd();
    const double dim_f = (double)dim;
    const svbool_t all = svptrue_b64();
    const svfloat64_t one = svdup_n_f64(1.0);
    for (size_t row = 0; row < rows; ++row) {
        const double *x_row = input + row * dim;
        double *out_row = out + row * dim;
        double *deriv_row = derivatives + row * dim;

        svfloat64_t sum_v = svdup_n_f64(0.0);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            sum_v = svadd_f64_m(pg, sum_v, svld1(pg, x_row + index));
        }
        const double mean = svaddv_f64(all, sum_v) / dim_f;

        const svfloat64_t mean_v = svdup_n_f64(mean);
        svfloat64_t variance_v = svdup_n_f64(0.0);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            svfloat64_t centered = svsub_f64_x(pg, svld1(pg, x_row + index), mean_v);
            variance_v = svmla_f64_m(pg, variance_v, centered, centered);
        }
        const double variance = svaddv_f64(all, variance_v) / dim_f;
        const double inv_std = 1.0 / sqrt(variance + eps);
        const svfloat64_t inv_std_v = svdup_n_f64(inv_std);

        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            svfloat64_t normalized = svmul_f64_x(
                pg, svsub_f64_x(pg, svld1(pg, x_row + index), mean_v), inv_std_v);
            svfloat64_t value = svmla_f64_x(
                pg, svld1(pg, bias + index), normalized, svld1(pg, weight + index));
            svfloat64_t sigmoid = llmlip_uma_sigmoid_sve_f64(pg, value);
            svst1(pg, out_row + index, svmul_f64_x(pg, value, sigmoid));
            svfloat64_t derivative = svmul_f64_x(
                pg, sigmoid,
                svadd_f64_x(pg, one, svmul_f64_x(pg, value, svsub_f64_x(pg, one, sigmoid))));
            svst1(pg, deriv_row + index, derivative);
        }
    }
}

void llmlip_uma_layer_norm_silu_vjp_rows_sve_f64(
    size_t rows,
    size_t dim,
    const double *restrict input,
    const double *restrict weight,
    const double *restrict grad_out,
    const double *restrict silu_derivative,
    double eps,
    double *restrict out
) {
    const size_t lanes = svcntd();
    const double dim_f = (double)dim;
    const svbool_t all = svptrue_b64();
    for (size_t row = 0; row < rows; ++row) {
        const double *x_row = input + row * dim;
        const double *grad_row = grad_out + row * dim;
        const double *deriv_row = silu_derivative + row * dim;
        double *out_row = out + row * dim;

        svfloat64_t sum_v = svdup_n_f64(0.0);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            sum_v = svadd_f64_m(pg, sum_v, svld1(pg, x_row + index));
        }
        const double mean = svaddv_f64(all, sum_v) / dim_f;

        const svfloat64_t mean_v = svdup_n_f64(mean);
        svfloat64_t variance_v = svdup_n_f64(0.0);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            svfloat64_t centered = svsub_f64_x(pg, svld1(pg, x_row + index), mean_v);
            variance_v = svmla_f64_m(pg, variance_v, centered, centered);
        }
        const double variance = svaddv_f64(all, variance_v) / dim_f;
        const double inv_std = 1.0 / sqrt(variance + eps);
        const svfloat64_t inv_std_v = svdup_n_f64(inv_std);

        svfloat64_t mean_grad_v = svdup_n_f64(0.0);
        svfloat64_t mean_grad_normalized_v = svdup_n_f64(0.0);
        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            svfloat64_t normalized = svmul_f64_x(
                pg, svsub_f64_x(pg, svld1(pg, x_row + index), mean_v), inv_std_v);
            svfloat64_t grad_normalized = svmul_f64_x(
                pg,
                svmul_f64_x(pg, svld1(pg, grad_row + index), svld1(pg, deriv_row + index)),
                svld1(pg, weight + index));
            mean_grad_v = svadd_f64_m(pg, mean_grad_v, grad_normalized);
            mean_grad_normalized_v =
                svmla_f64_m(pg, mean_grad_normalized_v, grad_normalized, normalized);
        }
        const double mean_grad = svaddv_f64(all, mean_grad_v) / dim_f;
        const double mean_grad_normalized = svaddv_f64(all, mean_grad_normalized_v) / dim_f;
        const svfloat64_t mean_grad_v2 = svdup_n_f64(mean_grad);
        const svfloat64_t mean_grad_normalized_v2 = svdup_n_f64(mean_grad_normalized);

        for (size_t index = 0; index < dim; index += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)index, (uint64_t)dim);
            svfloat64_t normalized = svmul_f64_x(
                pg, svsub_f64_x(pg, svld1(pg, x_row + index), mean_v), inv_std_v);
            svfloat64_t grad_normalized = svmul_f64_x(
                pg,
                svmul_f64_x(pg, svld1(pg, grad_row + index), svld1(pg, deriv_row + index)),
                svld1(pg, weight + index));
            svfloat64_t centered_grad = svsub_f64_x(
                pg,
                svsub_f64_x(pg, grad_normalized, mean_grad_v2),
                svmul_f64_x(pg, normalized, mean_grad_normalized_v2));
            svst1(pg, out_row + index, svmul_f64_x(pg, inv_std_v, centered_grad));
        }
    }
}

#endif /* LLMLIP_ENABLE_ARMPL_AMATH */
