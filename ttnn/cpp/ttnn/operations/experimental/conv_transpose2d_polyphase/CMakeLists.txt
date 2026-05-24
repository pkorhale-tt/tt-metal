add_library(ttnn_op_experimental_conv_transpose2d_polyphase ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::ConvTranspose2dPolyphase ALIAS ttnn_op_experimental_conv_transpose2d_polyphase)

tt_reuse_precompile_headers(ttnn_op_experimental_conv_transpose2d_polyphase TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_conv_transpose2d_polyphase)

set_target_properties(
    ttnn_op_experimental_conv_transpose2d_polyphase
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_conv_transpose2d_polyphase
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        conv_transpose2d_polyphase.cpp
)

target_include_directories(ttnn_op_experimental_conv_transpose2d_polyphase PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_conv_transpose2d_polyphase
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Conv
)

install(TARGETS ttnn_op_experimental_conv_transpose2d_polyphase LIBRARY COMPONENT tar)
