<?php

declare(strict_types=1);

function a() {
    function b() {
        function c() {
            return "c";
        }
        return "b" . c();
    }
    return "a" . b();
};

echo a(); // returns 'abc'