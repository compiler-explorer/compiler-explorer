! Type your code here, or load an example.
integer function square(x)
    implicit none
    integer, intent(in) :: x
    square = x * x
end function square
