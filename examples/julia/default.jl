function square(x)
    return x * x
end

precompile(square, (Int32,))
