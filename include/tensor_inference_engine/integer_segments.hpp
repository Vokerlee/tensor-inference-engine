#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define TIE_INTEGER_HD __host__ __device__
#else
#define TIE_INTEGER_HD
#endif

namespace tensor_inference_engine {

struct IntegerGroupBound {
    std::uint64_t magnitude;
    std::int64_t minimum_exponent;
    std::int64_t maximum_exponent;
};

struct IntegerSegmentPlan {
    std::uint8_t ends;
    std::uint8_t oversized;
    bool valid;
};

namespace integer_segment_detail {

TIE_INTEGER_HD inline std::uint64_t bounded_shift(std::uint64_t magnitude, std::int64_t shift,
    unsigned bits, std::uint64_t limit)
{
    if (magnitude == 0)
        return 0;
    if (shift < 0 || shift >= bits - 1 || magnitude > (limit >> shift))
        return limit + 1;
    return magnitude << shift;
}

TIE_INTEGER_HD inline std::uint64_t bounded_add(std::uint64_t left, std::uint64_t right,
    std::uint64_t limit)
{
    return left > limit || right > limit || left > limit - right ? limit + 1 : left + right;
}

} // namespace integer_segment_detail

// Greedy contiguous partitions of at most eight conservative integer bounds.
// Group g has an absolute bound magnitude * 2^maximum_exponent; its common
// anchor must not exceed minimum_exponent. Zero groups need no anchor.
// A group that cannot fit alone is marked oversized for a caller's fallback.
// Signed capacity excludes the asymmetric most-negative representable value.
TIE_INTEGER_HD inline IntegerSegmentPlan plan_integer_segments(const IntegerGroupBound* groups,
    std::size_t count, unsigned bits)
{
    if (count > 8 || bits < 2 || bits > 63)
        return {0xff, 0xff, false};
    if (count == 0)
        return {0, 0, true};
    const auto all = static_cast<std::uint8_t>((1u << count) - 1);
    const auto limit = (std::uint64_t{1} << (bits - 1)) - 1;
    IntegerSegmentPlan plan{0, 0, true};
    std::uint64_t accumulated = 0;
    std::int64_t anchor = 0;
    bool populated = false;
    for (std::size_t group = 0; group < count; ++group) {
        const auto& term = groups[group];
        if (term.magnitude == 0)
            continue;
        // Exponents supplied by callers must leave subtraction inside int64.
        if (term.minimum_exponent > term.maximum_exponent ||
            term.minimum_exponent < -0x7fffffffLL || term.maximum_exponent > 0x7fffffffLL)
            return {all, all, false};
        const auto next_anchor =
            populated && anchor < term.minimum_exponent ? anchor : term.minimum_exponent;
        const auto previous =
            integer_segment_detail::bounded_shift(accumulated, anchor - next_anchor, bits, limit);
        const auto addition = integer_segment_detail::bounded_shift(term.magnitude,
            term.maximum_exponent - next_anchor, bits, limit);
        const auto candidate = integer_segment_detail::bounded_add(previous, addition, limit);
        const auto singleton = integer_segment_detail::bounded_shift(term.magnitude,
            term.maximum_exponent - term.minimum_exponent, bits, limit);
        if (populated && candidate > limit) {
            plan.ends |= static_cast<std::uint8_t>(1u << (group - 1));
            accumulated = singleton;
            anchor = term.minimum_exponent;
        } else {
            accumulated = candidate;
            anchor = next_anchor;
        }
        if (singleton > limit)
            plan.oversized |= static_cast<std::uint8_t>(1u << group);
        populated = true;
    }
    plan.ends |= static_cast<std::uint8_t>(1u << (count - 1));
    return plan;
}

} // namespace tensor_inference_engine

#undef TIE_INTEGER_HD
