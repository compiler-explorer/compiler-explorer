# CMake can execute standalone CMake scripts with the -P parameter
# in the following manner: `cmake -P script.cmake`

set(flags ON OFF ON OFF OFF ON)
set(num_activated_flags)

foreach (value IN LISTS flags)
  if (value)
    math(EXPR num_activated_flags "${num_activated_flags} + 1")
  endif()
endforeach()

message("Activated Options: ${num_activated_flags}")
