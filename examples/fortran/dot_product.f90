! Compute the dot product of two vectors w/ 4 entries with explicit loops.
!   Try changing the target architecture to AVX (-mavx)
!   and enabling optimisation (-O3) and compare the results.
real function dot_prod(a, b)
    implicit none
    real, intent(in), dimension(4) :: a, b
    real, dimension(4) :: temp
    integer :: i
    
    do i = 1, 4
        temp(i) = a(i) * b(i)
    end do
    
    dot_prod = 0.
    
    do i = 1,4
        dot_prod = dot_prod + temp(i)
    end do
    
    return
end function dot_prod
