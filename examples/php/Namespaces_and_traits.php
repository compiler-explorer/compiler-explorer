<?php

declare(strict_types=1);

namespace NS {
	trait canAdd {
		static function add_two_numbers(int $num1, int $num2): int {
			return $num1 + $num2;
		}
	}

	class A {
		use canAdd;
		public function multiple_two_numbers(int $a, int $b): int {
			return $a * $b;
		}
	}
}

namespace {
	echo NS\A::add_two_numbers(1, 2);
	$a = new NS\A();
	echo $a->multiple_two_numbers(1, 2);
}