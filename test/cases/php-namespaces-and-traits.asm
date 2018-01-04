	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Namespaces_and_traits.php
	functionname:(null)
	numberofops:18
	compiledvars:	!0=$a
	line#*opfetchextreturnoperands
	6	0E>NOP
	12	1DECLARE_CLASS		$2	'ns%5Ca'
	13	2ADD_TRAIT			$2	,	'NS%5CcanAdd'
	12	3BIND_TRAITS			$2
	21	4INIT_STATIC_METHOD_CALL			'NS%5CA'	,	'add_two_numbers'
		5SEND_VAL_EX			1
		6SEND_VAL_EX			2
		7DO_FCALL	0	$3
		8ECHO			$3
	22	9NEW		$4	:5
		10DO_FCALL	0
		11ASSIGN			!0	,	$4
	23	12INIT_METHOD_CALL			!0	,	'multiple_two_numbers'
		13SEND_VAL_EX			1
		14SEND_VAL_EX			2
		15DO_FCALL	0	$7
		16ECHO			$7
	25	17>RETURN			1

	ClassNS\canAdd:
	Functionadd_two_numbers:
	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Namespaces_and_traits.php
	functionname:add_two_numbers
	numberofops:7
	compiledvars:	!0=$num1,	!1=$num2
	line#*opfetchextreturnoperands
	7	0E>RECV		!0
		1RECV		!1
	8	2ADD		~2	!0	,	!1
		3VERIFY_RETURN_TYPE			~2
		4>RETURN			~2
	9	5*VERIFY_RETURN_TYPE
		6*>RETURN			null

	Endoffunctionadd_two_numbers

	EndofclassNS\canAdd.

	ClassNS\A:
	Functionmultiple_two_numbers:
	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Namespaces_and_traits.php
	functionname:multiple_two_numbers
	numberofops:7
	compiledvars:	!0=$a,	!1=$b
	line#*opfetchextreturnoperands
	14	0E>RECV		!0
		1RECV		!1
	15	2MUL		~2	!0	,	!1
		3VERIFY_RETURN_TYPE			~2
		4>RETURN			~2
	16	5*VERIFY_RETURN_TYPE
		6*>RETURN			null

	Endoffunctionmultiple_two_numbers

	EndofclassNS\A.
