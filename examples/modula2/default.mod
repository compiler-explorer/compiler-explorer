MODULE PrintHelloWorld;

(*This program prints "Hello world!" on the standard output device*)

FROM InOut IMPORT WriteString, WriteLn;

BEGIN
	WriteString('Hello world!');
	WriteLn;
END PrintHelloWorld.
