.assembly extern System.Console { }
.assembly extern System.Runtime { }

.class private auto ansi beforefieldinit Program extends [System.Runtime]System.Object
{
    .method private hidebysig static int32 Square(int32 num) cil managed
    {
        .maxstack 8

        ldarg.0
        ldarg.0
        mul
        ret
    }
    .method private hidebysig static void Main() cil managed
    {
        .maxstack 8
        .entrypoint

        ldc.i4.s 42
        call int32 Program::Square(int32)
        call void [System.Console]System.Console::WriteLine(int32)
        ret
    }
    .method public hidebysig specialname rtspecialname instance void .ctor() cil managed
    {
        .maxstack 8

        ldarg.0
        call instance void [System.Runtime]System.Object::.ctor()
        ret
    }
}
