#ifndef __TSQR_TC_DETAIL_CONSTANT_HPP__
#define __TSQR_TC_DETAIL_CONSTANT_HPP__
namespace mtk {
namespace tsqr_tc {
namespace compute_mode {
using type = unsigned;
static const type fp32_fp16_hmma_cor    = 1;
static const type fp32_tf32_hmma_cor    = 2;
static const type fp32_fp16_hmma_no_cor = 3;
static const type fp32_tf32_hmma_no_cor = 4;
static const type fp32_no_tc            = 5;
} // namespace compute_mode
} // namespace tsqr_tc
} // namespace mtk
#endif
