// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::conv_transpose2d_polyphase::detail {

namespace nb = nanobind;

void bind_experimental_conv_transpose2d_polyphase(nb::module_& mod);

}  // namespace ttnn::operations::experimental::conv_transpose2d_polyphase::detail
