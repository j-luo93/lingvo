package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "one_billion_wds",
    srcs = ["one_billion_wds.py"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:layers",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
        "//lingvo/tasks/lm:input_generator",
        "//lingvo/tasks/lm:layers",
        "//lingvo/tasks/lm:model",
    ],
)

py_library(
    name = "ptb_word_level",
    srcs = ["ptb_word_level.py"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:layers",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/lm:input_generator",
        "//lingvo/tasks/lm:layers",
        "//lingvo/tasks/lm:model",
    ],
)

py_library(
    name = "one_billion_word_level",
    srcs = ["one_billion_word_level.py"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:layers",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/lm:input_generator",
        "//lingvo/tasks/lm:layers",
        "//lingvo/tasks/lm:model",
    ],
)

py_library(
    name = "params",
    srcs = ["__init__.py"],
    deps = [
        ":one_billion_wds",
        ":ptb_word_level",
        ":one_billion_word_level"
    ],
)
