	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Functions_in_functions.php
	functionname:(null)
	numberofops:5
	compiledvars:	none
	line#*opfetchextreturnoperands
	5	0E>NOP
	15	1INIT_FCALL			'a'
		2DO_FCALL	0	$0
		3ECHO			$0
		4>RETURN			1

	Function%00b%2Fcompiler-explorer%2Fexamples%2Fphp%2FFunctions_in_functions.php0x7fb5a455009a:
	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Functions_in_functions.php
	functionname:b
	numberofops:6
	compiledvars:	none
	line#*opfetchextreturnoperands
	7	0E>DECLARE_FUNCTION			'c'
	10	1INIT_FCALL_BY_NAME			'c'
		2DO_FCALL	0	$0
		3CONCAT		~1	'b'	,	$0
		4>RETURN			~1
	11	5*>RETURN			null

	Endoffunction%00b%2Fcompiler-explorer%2Fexamples%2Fphp%2FFunctions_in_functions.php0x7fb5a455009a

	Function%00c%2Fcompiler-explorer%2Fexamples%2Fphp%2FFunctions_in_functions.php0x7fb5a455007a:
	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Functions_in_functions.php
	functionname:c
	numberofops:2
	compiledvars:	none
	line#*opfetchextreturnoperands
	8	0E>>RETURN			'c'
	9	1*>RETURN			null

	Endoffunction%00c%2Fcompiler-explorer%2Fexamples%2Fphp%2FFunctions_in_functions.php0x7fb5a455007a

	Functiona:
	Findingentrypoints
	Branchanalysisfromposition:0
	1jumpsfound.(Code=62)	Position1=-2
	filename:/compiler-explorer/examples/php/Functions_in_functions.php
	functionname:a
	numberofops:6
	compiledvars:	none
	line#*opfetchextreturnoperands
	6	0E>DECLARE_FUNCTION			'b'
	12	1INIT_FCALL_BY_NAME			'b'
		2DO_FCALL	0	$0
		3CONCAT		~1	'a'	,	$0
		4>RETURN			~1
	13	5*>RETURN			null

	Endoffunctiona
