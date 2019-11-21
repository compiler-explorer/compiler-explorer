doc = """Compiler wrapper.

Usage:
  compiler_wrapper.jl <input_code> --output=<output_path> [--format=<fmt>] [--optimize]
  compiler_wrapper.jl -h | --help
  compiler_wrapper.jl --version

Options:
  -h --help         Show this screen.
  --version         Show version.
  --format=<fmt>    Set output format (One of "typed", "llvm", "native") [default: native]
  --optimize        Sets whether llvm output should be optimized or not. [default: true]
                    Has no effect on "ast" or "native" output.
"""

using InteractiveUtils, DocOpt

# Read options in from command-line
args = docopt(doc, version=v"1.0.0")

user_code = String(read(args["<input_code>"]))
output_path = args["--output"]
optimize = args["--optimize"]
format = args["--format"]

# Get the last expression, figure out what the user is asking for
function parse_all(code::String)
    exprs = Any[]
    starts = Int[]
    start = 1
    last_start = 1
    while true
        push!(starts, start)
        expr, start = try
            Meta.parse(code, start; greedy=true)
        catch
            # If we can't parse it all, fail hard
            push!(exprs, Expr(:incomplete, "unparsable input from index $(start)"))
            return exprs
        end

        # If we've reached the end, break
        if expr === nothing
            pop!(starts)
            break
        end
        push!(exprs, expr)
    end
    return exprs, starts
end
exprs, starts = parse_all(user_code)
expr_to_compile = last(exprs)
code_to_include = user_code[1:last(starts)]

# If it's not even an Expr (like it's just a constant expression) just return
if !isa(expr_to_compile, Expr)
    exit(0)
end

# If it's incomplete, that's fine, let 'em finish
if expr_to_compile.head == :incomplete
    exit(0)
end

# If it's a call, compile it within a module
if expr_to_compile.head == :call
    # Include it into a module (all except the last line)
    m = Module(:Godbolt)
    Base.include_string(m, code_to_include)

    # Pull the function out of the module
    f = getfield(m, expr_to_compile.args[1])

    # Get the types of the arguments:
    arg_types = typeof.(Core.eval.(Ref(m), expr_to_compile.args[2:end]))

    # Call the appropriate code_* function
    open(output_path, "w") do io
        if format == "typed"
            ir, retval = InteractiveUtils.code_typed(f, arg_types)[1]
            Base.IRShow.show_ir(io, ir)
        elseif format == "llvm"
            InteractiveUtils.code_llvm(io, f, arg_types; optimize=optimize)
        elseif format == "native"
            InteractiveUtils.code_native(io, f, arg_types)
        end
    end
    exit(0)
end

error("Unable to compile expression type $(expr_to_compile.head)")