#include <stddef.h>
#include <string.h>

typedef float v8sf __attribute__((vector_size(32)));
typedef double v4df __attribute__((vector_size(32)));

extern v8sf _ZGVdN8v_expf(v8sf);
extern v4df _ZGVdN4v_exp(v4df);

static inline v8sf loadu8(const float *ptr) {
  v8sf value;
  __builtin_memcpy(&value, ptr, sizeof(value));
  return value;
}

static inline void storeu8(float *ptr, v8sf value) {
  __builtin_memcpy(ptr, &value, sizeof(value));
}

static inline v4df loadu4d(const double *ptr) {
  v4df value;
  __builtin_memcpy(&value, ptr, sizeof(value));
  return value;
}

static inline void storeu4d(double *ptr, v4df value) {
  __builtin_memcpy(ptr, &value, sizeof(value));
}

void llmlip_orb_f64_silu_derivative_mvec(double *values, double *derivative,
                                         size_t len) {
  const v4df one = {1.0, 1.0, 1.0, 1.0};
  const v4df zero = {0.0, 0.0, 0.0, 0.0};
  size_t offset = 0;
  for (; offset + 4 <= len; offset += 4) {
    const v4df input = loadu4d(values + offset);
    const v4df exp_neg = _ZGVdN4v_exp(zero - input);
    const v4df sigmoid = one / (one + exp_neg);
    storeu4d(derivative + offset, sigmoid * (one + input * (one - sigmoid)));
    storeu4d(values + offset, input * sigmoid);
  }
  for (; offset < len; ++offset) {
    const double input = values[offset];
    const double exp_neg = __builtin_exp(-input);
    const double sigmoid = 1.0 / (1.0 + exp_neg);
    derivative[offset] = sigmoid * (1.0 + input * (1.0 - sigmoid));
    values[offset] = input * sigmoid;
  }
}

void llmlip_orb_f32_silu_derivative_mvec(float *values, float *derivative,
                                         size_t len) {
  const v8sf one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const v8sf zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  size_t offset = 0;
  for (; offset + 8 <= len; offset += 8) {
    const v8sf input = loadu8(values + offset);
    const v8sf exp_neg = _ZGVdN8v_expf(zero - input);
    const v8sf sigmoid = one / (one + exp_neg);
    storeu8(derivative + offset, sigmoid * (one + input * (one - sigmoid)));
    storeu8(values + offset, input * sigmoid);
  }
  for (; offset < len; ++offset) {
    const float input = values[offset];
    const float exp_neg = __builtin_expf(-input);
    const float sigmoid = 1.0f / (1.0f + exp_neg);
    derivative[offset] = sigmoid * (1.0f + input * (1.0f - sigmoid));
    values[offset] = input * sigmoid;
  }
}

void llmlip_orb_f32_add_bias_silu_derivative_mvec(float *values,
                                                  float *derivative,
                                                  const float *bias,
                                                  size_t len) {
  const v8sf one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const v8sf zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  size_t offset = 0;
  for (; offset + 8 <= len; offset += 8) {
    const v8sf input = loadu8(values + offset) + loadu8(bias + offset);
    const v8sf exp_neg = _ZGVdN8v_expf(zero - input);
    const v8sf sigmoid = one / (one + exp_neg);
    storeu8(derivative + offset, sigmoid * (one + input * (one - sigmoid)));
    storeu8(values + offset, input * sigmoid);
  }
  for (; offset < len; ++offset) {
    const float input = values[offset] + bias[offset];
    const float exp_neg = __builtin_expf(-input);
    const float sigmoid = 1.0f / (1.0f + exp_neg);
    derivative[offset] = sigmoid * (1.0f + input * (1.0f - sigmoid));
    values[offset] = input * sigmoid;
  }
}
