const doc = """Julia wrapper.

Usage:
  julia_wrapper.jl <input_code> <output_path> [--format=<fmt>] [--optimize=<opt>] [--verbose]
  julia_wrapper.jl --help

Options:
  -h --help                Show this screen.
  --format=<fmt>           Set output format (One of "lowered", "typed", "warntype", "llvm", "native") [default: native]
  --optimize={true*|false} Controls whether "llvm" or "typed" output should be optimized or not [default: true]
  --verbose                Prints some process info
"""

using InteractiveUtils

function main()

    if first(ARGS) == "--"
        popfirst!(ARGS)
    end

    format = "native"
    debuginfo = :source
    optimize = true
    verbose = false
    show_help = false
    arg_parser_error = false
    positional_ARGS = String[]

    for x in ARGS
        if startswith(x, "--format=")
            format = x[10:end]
        elseif startswith(x, "--optimize=")
            # Do not error out if we can't parse the option
            optimize = something(tryparse(Bool, x[12:end]), true)
        elseif x == "--verbose"
            verbose = true
        elseif x == "--help" || x == "-h"
            show_help = true
        elseif !startswith(x, "-")
            push!(positional_ARGS, x)
        else
            arg_parser_error = true
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
                # In julia v1.7-1.9 `me.specializations` is always a `Core.SimpleVector`, but in
                # Julia v1.10+ it can also be a single instance of `Core.MethodInstance`, which
                # isn't iterable, so we put it in a tuple to be able to do the for loop below.
                specs = if me.specializations isa Core.SimpleVector
                    me.specializations
                elseif me.specializations isa Core.MethodInstance
                    (me.specializations,)
                else
                    error("Cannot handle specializations of type $(typeof(me.specializations))")
                end
                for s in specs
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

    # open output file
    open(output_path, "w") do io
        # For all found methods
        for (me_fun, me_types, me) in m_methods
            if format == "typed"
                ir, retval = InteractiveUtils.code_typed(me_fun, me_types; optimize, debuginfo)[1]
                Base.IRShow.show_ir(io, ir)
            elseif format == "lowered"
                cl = Base.code_lowered(me_fun, me_types; debuginfo)
                print(io, cl)
            elseif format == "llvm"
                InteractiveUtils.code_llvm(io, me_fun, me_types; optimize, debuginfo)
            elseif format == "llvm-module"
                @static if VERSION >= v"1.11.0-"
                    # Hide safepoint on entry.  Only in Julia v1.11+ `code_llvm` exposes
                    # codegen parameters.
                    InteractiveUtils.code_llvm(io, me_fun, me_types; optimize, debuginfo=:source, raw=true, dump_module=true, params=Base.CodegenParams(; debug_info_kind=Cint(1), safepoint_on_entry=false, debug_info_level=Cint(2)))
                else
                    InteractiveUtils.code_llvm(io, me_fun, me_types; optimize, debuginfo=:source, raw=true, dump_module=true)
                end
            elseif format == "native"
                # In Julia v1.10- `code_native` doesn't expose codegen parameters.
                @static if VERSION >= v"1.11.0-"
                    # With kind==1 we get full debug info:
                    # <https://github.com/JuliaLang/julia/blob/bf9079afb05829f51e60db888cb29a7c45296ee1/base/reflection.jl#L1393>.
                    # Also hide safepoint on entry.  Codegen parameters only available in
                    # Julia v1.11+.
                    InteractiveUtils.code_native(io, me_fun, me_types; debuginfo, params=Base.CodegenParams(; debug_info_kind=Cint(1), safepoint_on_entry=false, debug_info_level=Cint(2)))
                else
                    InteractiveUtils.code_native(io, me_fun, me_types; debuginfo)
                end
            elseif format == "warntype"
                InteractiveUtils.code_warntype(io, me_fun, me_types; debuginfo)
            end
            # Add extra newline, because some of the above tools don't add a final newline,
            # and when we have multiple functions to be shown, they'd be mixed up.
            println(io)
        end
    end
    exit(0)
end

main()
