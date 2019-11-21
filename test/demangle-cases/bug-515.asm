_Z3barv:
        movl    $_ZL3foo, %eax
        ret
_Z4bar2v:
        movl    $foo2, %eax
        ret
foo2:
        .long   1
_ZL3foo:
        .long   2