#include <immintrin.h>

#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

namespace avs {

using vecf32_t = std::vector<float>;
using matf32_t = std::vector<std::vector<float>>;
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static bool is_amxbf16_supported() {
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "a"(7), "c"(0));
  return edx & (1 << 22);
}

static void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
  if (!src)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    ((uint8_t *)handle)[i] = src[i];
  }
}

static void write_to_dnnl_memory(void const *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
  if (!dst)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    dst[i] = ((uint8_t *)handle)[i];
  }
}

void avx512_substract(float const *a, float const *b, float *result, int32_t const &N) {
    size_t i;
    for (i = 0; i < N; i += 16) {
        __m512 vec_a = _mm512_loadu_ps(&a[i]);
        __m512 vec_b = _mm512_loadu_ps(&b[i]);
        __m512 vec_result = _mm512_sub_ps(vec_a, vec_b);
        _mm512_storeu_ps(&result[i], vec_result);
    }
}

static avs::matf32_t avx512_subtract_batch(avs::vecf32_t const &query,
                                           avs::matf32_t const &data) {
  int32_t const N = data.size();
  int32_t const dim = data[0].size();
  avs::matf32_t result(N, avs::vecf32_t(dim, 0.0f));
  for (int32_t i = 0; i < N; i++) {
    float PORTABLE_ALIGN64 tmp[dim];
    avx512_substract(query.data(), data[i].data(), tmp, query.size());
    for (int32_t k = 0; k < dim; k++) {
      result[i][k] = tmp[k];
    }
  }
  return result;
}

static avs::vecf32_t amx_matmul(const int32_t &r, const int32_t &c,
                                avs::vecf32_t const &m, avs::vecf32_t const &mt,
                                dnnl::engine &engine, dnnl::stream &stream) {
  avs::vecf32_t dst(r * r, 0.0f);

  dnnl::memory::dims a_dims = {r, c};
  dnnl::memory::dims b_dims = {c, r};
  dnnl::memory::dims c_dims = {r, r};

  auto a_in_md = dnnl::memory::desc(a_dims, dt::f32, tag::ab);
  auto b_in_md = dnnl::memory::desc(b_dims, dt::f32, tag::ab);
  auto c_out_md = dnnl::memory::desc(c_dims, dt::f32, tag::ab);
  auto a_in_mem = dnnl::memory(a_in_md, engine);
  auto b_in_mem = dnnl::memory(b_in_md, engine);
  write_to_dnnl_memory(m.data(), a_in_mem);
  write_to_dnnl_memory(mt.data(), b_in_mem);

  auto a_md = dnnl::memory::desc(a_dims, dt::bf16, tag::any);
  auto b_md = dnnl::memory::desc(b_dims, dt::bf16, tag::any);
  auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_out_md);

  auto a_mem = dnnl::memory(pd.src_desc(), engine);
  dnnl::reorder(a_in_mem, a_mem).execute(stream, a_in_mem, a_mem);
  auto b_mem = dnnl::memory(pd.weights_desc(), engine);
  dnnl::reorder(b_in_mem, b_mem).execute(stream, b_in_mem, b_mem);
  auto c_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::matmul(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, a_mem});
  args.insert({DNNL_ARG_WEIGHTS, b_mem});
  args.insert({DNNL_ARG_DST, c_mem});
  prim.execute(stream, args);
  stream.wait();

  read_from_dnnl_memory(dst.data(), c_mem);
  avs::vecf32_t result(r, 0.0f);
  for (int32_t i = 0; i < r; i++) {
    result[i] = dst[i * r + i];
  }
  return result;
}

static matf32_t amx_inner_product(int32_t const &n, int32_t const &oc,
                                  int32_t const &ic, const float *s,
                                  const float* w, dnnl::engine &engine,
                                  dnnl::stream &stream) {

  dnnl::memory::dims s_dims = {n, ic};
  dnnl::memory::dims w_dims = {oc, ic};
  dnnl::memory::dims dst_dims = {n, oc};

  auto s_in_md = dnnl::memory::desc(s_dims, dt::f32, tag::ab);
  auto w_in_md = dnnl::memory::desc(w_dims, dt::f32, tag::ab);
  auto dst_out_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);
  auto s_in_mem = dnnl::memory(s_in_md, engine);
  auto w_in_mem = dnnl::memory(w_in_md, engine);
  write_to_dnnl_memory(s, s_in_mem);
  write_to_dnnl_memory(w, w_in_mem);

  auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::any);
  auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::any);

  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_training, s_md, w_md, dst_out_md);

  auto s_mem = dnnl::memory(pd.src_desc(), engine);
  dnnl::reorder(s_in_mem, s_mem).execute(stream, s_in_mem, s_mem);
  auto w_mem = dnnl::memory(pd.weights_desc(), engine);
  dnnl::reorder(w_in_mem, w_mem).execute(stream, w_in_mem, w_mem);
  auto dst_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::inner_product_forward(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, s_mem});
  args.insert({DNNL_ARG_WEIGHTS, w_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  prim.execute(stream, args);
  stream.wait();

  avs::vecf32_t dst(n * oc, 0.0f);
  read_from_dnnl_memory(dst.data(), dst_mem);
  avs::matf32_t result(n, avs::vecf32_t(oc, 0.0f));
  for (int32_t i = 0; i < n; i++) {
    for (int32_t j = 0; j < oc; j++) {
      result[i][j] = dst[i * oc + j];
    }
  }
  return result;
}

void print_vector(const float *data, std::string prefix) {
  std::cout << prefix << " :";
  for (unsigned i = 0; i < 10; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

static avs::matf32_t ip_distance_amx(const float* queries,
                                     const float* data,
                                     int32_t queries_size,
                                     int32_t data_size,
                                     int32_t dim,
                                     dnnl::engine &engine,
                                     dnnl::stream &stream) {
  return amx_inner_product(
    queries_size, data_size, dim, queries, data, engine, stream);
}

static avs::vecf32_t l2_distance_amx(avs::vecf32_t const &query,
                                     avs::matf32_t const &batch,
                                     dnnl::engine &engine,
                                     dnnl::stream &stream) {
  int32_t const batch_size = batch.size();
  int32_t const dim = batch[0].size();
  avs::matf32_t dis_2d = avx512_subtract_batch(query, batch);
  avs::matf32_t dis_2d_t(dis_2d[0].size(), avs::vecf32_t(dis_2d.size(), 0.0f));
  for (int32_t i = 0; i < dis_2d_t.size(); i++) {
    for (int32_t j = 0; j < dis_2d_t[0].size(); j++) {
      dis_2d_t[i][j] = dis_2d[j][i];
    }
  }
  avs::vecf32_t dis_1d(batch_size * dim);
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < dim; j++) {
      dis_1d[i * dim + j] = dis_2d[i][j];
    }
  }
  avs::vecf32_t dis_1d_t(batch_size * dim);
  for (int32_t i = 0; i < dim; i++) {
    for (int32_t j = 0; j < batch_size; j++) {
      dis_1d_t[i * batch_size + j] = dis_2d_t[i][j];
    }
  }
  return amx_matmul(batch_size, dim, dis_1d, dis_1d_t, engine, stream);
}

static float L2Sqr(void const *vec1, void const *vec2, int32_t const &dim) {
  float *v1 = (float *)vec1;
  float *v2 = (float *)vec2;
  float result = 0;
  for (int32_t i = 0; i < dim; i++) {
    float t = *v1 - *v2;
    v1++;
    v2++;
    result += t * t;
  }
  return (result);
}

static float
L2SqrAVX512(const void *vec1, const void *vec2, int32_t const &dim) {
    float *pVect1 = (float *) vec1;
    float *pVect2 = (float *) vec2;
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = dim >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

static float InnerProduct(void const *vec1, void const *vec2,
                          int32_t const &dim) {
  float *v1 = (float *)vec1;
  float *v2 = (float *)vec2;
  float result = 0;
  for (int32_t i = 0; i < dim; i++) {
    result += ((float *)v1)[i] * ((float *)v2)[i];
  }
  return result;
}


static float
InnerProductAVX512(const void *vec1, const void *vec2, int32_t const &dim) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) vec1;
    float *pVect2 = (float *) vec2;

    size_t qty16 = dim / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static avs::vecf32_t l2_distance_avx512(avs::vecf32_t const &query,
                                        avs::matf32_t const &batch,
                                        dnnl::engine &engine,
                                        dnnl::stream &stream) {
  int32_t const dim = batch[0].size();
  avs::vecf32_t result(batch.size());
  for (int32_t i = 0; i < batch.size(); i++) {
    auto d = L2SqrAVX512(query.data(), batch[i].data(), dim);
    result[i] = d;
  }
  return result;
}

static avs::vecf32_t ip_distance_avx512(avs::vecf32_t const &query,
                                        avs::matf32_t const &batch,
                                        dnnl::engine &engine,
                                        dnnl::stream &stream) {
  int32_t const dim = batch[0].size();
  avs::vecf32_t result(batch.size());
  for (int32_t i = 0; i < batch.size(); i++) {
    auto d = InnerProductAVX512(query.data(), batch[i].data(), dim);
    result[i] = d;
  }
  return result;
}

} // namespace avs