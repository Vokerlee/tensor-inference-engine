#include "tensor_inference_engine/integer_segments.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

using tensor_inference_engine::IntegerGroupBound;
using tensor_inference_engine::plan_integer_segments;

void check(bool condition)
{
    if (!condition)
        throw std::runtime_error("integer segment check failed");
}

bool feasible(const std::array<IntegerGroupBound, 8>& groups, unsigned begin, unsigned end)
{
    std::int64_t anchor = 1000;
    unsigned active = 0;
    for (auto i = begin; i < end; ++i) {
        if (groups[i].magnitude) {
            ++active;
            if (groups[i].minimum_exponent < anchor)
                anchor = groups[i].minimum_exponent;
        }
    }
    if (active <= 1)
        return true;
    std::uint64_t sum = 0;
    for (auto i = begin; i < end; ++i)
        if (groups[i].magnitude)
            sum += groups[i].magnitude << (groups[i].maximum_exponent - anchor);
    return sum <= 63;
}

void exhaustive_partitions()
{
    std::mt19937 generator(831);
    for (unsigned sample = 0; sample < 200; ++sample) {
        std::array<IntegerGroupBound, 8> groups;
        for (auto& group : groups) {
            group.magnitude = generator() % 10;
            group.minimum_exponent = generator() % 8;
            group.maximum_exponent = group.minimum_exponent + generator() % 4;
        }
        unsigned minimum = 8;
        for (unsigned cuts = 128; cuts < 256; ++cuts) {
            unsigned begin = 0;
            bool valid = true;
            for (unsigned end = 1; end <= 8; ++end) {
                if (cuts & (1u << (end - 1))) {
                    valid &= feasible(groups, begin, end);
                    begin = end;
                }
            }
            const auto count = static_cast<unsigned>(std::popcount(cuts));
            if (valid && count < minimum)
                minimum = count;
        }
        const auto plan = plan_integer_segments(groups.data(), groups.size(), 7);
        check(plan.valid && std::popcount(static_cast<unsigned>(plan.ends)) == minimum);
        unsigned begin = 0;
        for (unsigned end = 1; end <= 8; ++end) {
            if (plan.ends & (1u << (end - 1))) {
                check(feasible(groups, begin, end));
                begin = end;
            }
        }
    }
}

} // namespace

int main()
{
    exhaustive_partitions();
    std::array<IntegerGroupBound, 4> groups{
        {
         {100, -1000, -1000},
         {100, 1000, 1000},
         {100, -1000, -1000},
         {100, 1000, 1000},
         }
    };
    check(plan_integer_segments(groups.data(), 4, 24).ends == 15);
    groups[0] = {std::numeric_limits<std::uint64_t>::max(), 0, 0};
    check(plan_integer_segments(groups.data(), 4, 63).oversized & 1);
    groups[0] = {1, 10, -10};
    check(!plan_integer_segments(groups.data(), 4, 24).valid);
    for (auto& group : groups)
        group = {0, std::numeric_limits<std::int64_t>::min(),
            std::numeric_limits<std::int64_t>::max()};
    check(plan_integer_segments(groups.data(), 4, 24).ends == 8);
    check(plan_integer_segments(nullptr, 0, 24).ends == 0);
    return 0;
}
