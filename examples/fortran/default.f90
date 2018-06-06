! Type your code here, or load an example.
real function square(x)
    implicit none
    real, intent(in) :: x
    square = x * x
    return
end function square
