"""Workspace file for lingvo."""

load(
    "//lingvo:repo.bzl",
    "cc_tf_configure",
    "lingvo_protoc_deps",
    "lingvo_testonly_deps",
)

git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    tag = "1.3.0",
)

git_repository(
    name = "io_bazel_rules_python",
    # NOT VALID!  Replace this with a Git commit SHA.
    commit = "8b5d0683a7d878b28fffe464779c8a53659fc645",
    remote = "https://github.com/bazelbuild/rules_python.git",
)



load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories")
load("@io_bazel_rules_python//python:pip.bzl", "pip_import")



pip_import(
    name = "notebooks",
    requirements = "//notebooks:requirements.txt",
)

load(
    "@notebooks//:requirements.bzl",
    _notebooks_install = "pip_install",
)

_notebooks_install()

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()
