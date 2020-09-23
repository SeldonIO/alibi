This directory contains license information of 3rd party dependencies.

For any changes in dependencies, the command `make licenses` should be run from project root to generate the updated licese information files. The command will be run automatically in CI and fail the build if there are any differences in the license files.
The command itself usex `tox` to recreate a clean environment with only the library and its dependencies installed.
