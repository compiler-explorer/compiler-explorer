function square(x)
    return x * x
end

function dsquare(x)
    return autodiff(square, Active(x))
end

precompile(dsquare, (Float32,))
