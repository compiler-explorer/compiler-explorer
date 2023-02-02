doc = """Julia wrapper.

Usage:
  julia_wrapper.jl <input_code> <output_path> [--format=<fmt>] [--debuginfo=<info>] [--optimize] [--verbose]
  julia_wrapper.jl --help

Options:
  -h --help           Show this screen.
  --format=<fmt>      Set output format (One of "lowered", "typed", "warntype", "llvm", "native") [default: native]
                      lowered	
                      typed
                      warntype
                      llvm
                      native
  --debuginfo=<info>  Controls amount of generated metadata (One of "default", "none") [default: default]
  --optimize          Sets whether "llvm" or "typed" output should be optimized or not.
  --verbose           Prints some process info
"""

using InteractiveUtils

if length(ARGS) < 2
	println(doc)
	exit(1)
end

if length(ARGS) > 3 && ARGS[3] == "--help"
	println(doc)
end

input_file = popfirst!(ARGS)
output_path = popfirst!(ARGS)
format = "native"
debuginfo = :default
optimize = false
verbose = false

for x in ARGS
	if startswith(x, "--format=")
		global format = x[10:end]
	end
	if startswith(x, "--debuginfo=")
		if x[13:end] == "none"
			global debuginfo = :none
		end
	end
	if x == "--optimize"
		global optimize = true
	end
	if x == "--verbose"
		global verbose = true
	end
end

# Include user code into module
m = Module(:Godbolt)
Base.include(m, input_file)

# Find functions and method specializations
m_methods = Any[]
for name in names(m, all=true)
    local fun = getfield(m, name)
    if fun isa Function
        if verbose
            println("Function: ", fun)
        end
        for me in methods(fun)
            for s in me.specializations
                if s != nothing
                    me_types = getindex(s.specTypes.parameters, 2:length(s.specTypes.parameters))
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
