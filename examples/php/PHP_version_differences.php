<?php

declare(strict_types=1);

// PHP 7.1 new ISSET_ISEMPTY_STATIC_PROP code
isset(A::$a);

// PHP 7.2 new COUNT code
count([1,2,3]);

// PHP 7.2 new FUNC_GET_ARGS opcode
function first_arg($a, $b) {
    array_slice(func_get_args(), 0, 1);
}
