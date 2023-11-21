doc = """Julia wrapper.

Usage:
  julia_wrapper.jl <input_code> <output_path> [--format=<fmt>] [--debuginfo=<info>] [--optimize=<opt>] [--verbose]
  julia_wrapper.jl --help

Options:
  -h --help                Show this screen.
  --format=<fmt>           Set output format (One of "lowered", "typed", "warntype", "llvm", "native") [default: native]
  --debuginfo=<info>       Controls amount of generated metadata (One of "default", "none") [default: default]
  --optimize={true*|false} Controls whether "llvm" or "typed" output should be optimized or not [default: true]
  --verbose                Prints some process info
"""

using InteractiveUtils

if first(ARGS) == "--"
    popfirst!(ARGS)
end

format = "native"
debuginfo = :default
optimize = true
verbose = false
show_help = false
arg_parser_error = false
positional_ARGS = String[]

for x in ARGS
	if     startswith(x, "--format=")
		global format = x[10:end]
    elseif startswith(x, "--debuginfo=")
		if x[13:end] == "none"
			global debuginfo = :none
		end
	elseif startswith(x, "--optimize=")
		# Do not error out if we can't parse the option
		global optimize = something(tryparse(Bool, x[12:end]), true)
    elseif x == "--verbose"
		global verbose = true
	elseif x == "--help" || x == "-h"
        global show_help = true
    elseif !startswith(x, "-")
        push!(positional_ARGS, x)
    else
        global arg_parser_error = true
        println("Unknown argument ", x)
    end
end

if show_help
    println(doc)
    exit(Int(arg_parser_error)) # exit(1) if failed to parse
end

if length(positional_ARGS) != 2
    arg_parser_error = true
    println("Expected two position args", positional_ARGS)
end

if arg_parser_error
    println(doc)
    exit(1)
end

input_file = popfirst!(positional_ARGS)
output_path = popfirst!(positional_ARGS)

# Include user code into module
m = Module(:Godbolt)
Base.include(m, input_file)

# Find functions and method specializations
m_methods = Any[]
for name in names(m, all=true, imported=true)
    local fun = getfield(m, name)
    if fun isa Function
        if verbose
            println("Function: ", fun)
        end
        # only show methods found in input module
        for me in methods(fun, m)
            for s in me.specializations
                if s != nothing
                    spec_types = s.specTypes
                    # In case of a parametric type, see https://docs.julialang.org/en/v1/devdocs/types/#UnionAll-types
                    while typeof(spec_types) == UnionAll
                        spec_types = spec_types.body
                    end
                    me_types = getindex(spec_types.parameters, 2:length(spec_types.parameters))
                    push!(m_methods, (fun, me_types, me))
                    if verbose
                        println("    Method types: ", me_types)
                    end
                end
            end
        end
    end
end

# Open output file
open(output_path, "w") do io
    # For all found methods
    for (me_fun, me_types, me) in m_methods
        io_buf = IOBuffer() # string buffer
        if format == "typed"
            ir, retval = InteractiveUtils.code_typed(me_fun, me_types; optimize, debuginfo)[1]
            Base.IRShow.show_ir(io_buf, ir)
        elseif format == "lowered"
            cl = Base.code_lowered(me_fun, me_types; debuginfo)
            print(io_buf, cl)
        elseif format == "llvm"
            InteractiveUtils.code_llvm(io_buf, me_fun, me_types; optimize, debuginfo)
        elseif format == "native"
            InteractiveUtils.code_native(io_buf, me_fun, me_types; debuginfo)
        elseif format == "warntype"
            InteractiveUtils.code_warntype(io_buf, me_fun, me_types; debuginfo)
        end
        code = String(take!(io_buf))
        line_num = count("\n",code)
        # Print first line: <[source code line] [number of output lines] [function name] [method types]>
        write(io, "<")
        print(io, me.line)
        print(io, " ")
        print(io, line_num)
        print(io, " ")
        print(io, me_fun)
        write(io, " ")
        for i in 1:length(me_types)
            print(io, me_types[i])
            if i < length(me_types)
                write(io, ", ")
            end
        end
        write(io, ">\n")
        # Print code for this method
        write(io, code)
        write(io, "\n")
    end
end
exit(0)
