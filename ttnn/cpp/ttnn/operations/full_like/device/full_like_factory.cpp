// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include "full_like_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::full_like {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

// After the full modification and if there are no issues in the overall tests, it will be added to `bfloat16.hpp` and
// applied globally.
uint32_t get_bfloat16_rounded(const float val) {
    uint32_t float_bits = *reinterpret_cast<const uint32_t*>(&val);

    // upper 16 bits
    uint16_t bfloat16_bits = float_bits >> 16;

    // check Guard, Round, Sticky bits from lower 16 bits
    uint32_t lower_bits = float_bits & 0xFFFF;
    uint32_t guard_bit = (lower_bits >> 15) & 1;
    uint32_t round_bit = (lower_bits >> 14) & 1;
    uint32_t sticky_bit = (lower_bits & 0x3FFF) != 0;

    // Tie-to-even rounding rule
    if (guard_bit && (round_bit || sticky_bit || (bfloat16_bits & 1))) {
        bfloat16_bits += 1;
    }

    return static_cast<uint32_t>(bfloat16_bits) << 16;
}

union datatype {
    uint32_t u32;
    float f32;
} u;

ProgramDescriptor FullLikeOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto fill_value = operation_attributes.fill_value;
    const DataType dtype = operation_attributes.dtype;
    IDevice* device = input.device();

    auto num_tiles = input.physical_volume() / TILE_HW;
    auto data_format = datatype_to_dataformat_converter(dtype);
    uint32_t single_tile_size = tt::tile_size(data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    constexpr CBIndex cb_fill_value_id = CBIndex::c_24;

    KernelDescriptor::Defines writer_defines;
    switch (dtype) {
        case DataType::BFLOAT16: writer_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::INT32: writer_defines.emplace_back("OUTPUT_DTYPE_INT32", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        auto float_fill_value = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = get_bfloat16_rounded(float_fill_value);
        } else {
            u.f32 = float_fill_value;
        }
    }

    std::vector<uint32_t> writer_ct_args = {(uint32_t)cb_fill_value_id, TILE_HW, single_tile_size};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_fill_value_id,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    KernelDescriptor writer;
    writer.kernel_source = "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp";
    writer.core_ranges = all_cores;
    writer.compile_time_args = writer_ct_args;
    writer.defines = writer_defines;
    writer.config = WriterConfigDescriptor{};

    uint32_t tiles_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core(i / num_cores_y, i % num_cores_y);

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        writer.runtime_args.push_back({core, {output.buffer()->address(), u.u32, num_tiles_per_core, tiles_offset}});
        tiles_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::operations::full_like
