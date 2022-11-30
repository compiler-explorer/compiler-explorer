Imports System

Module Program
    Function Square(num As Integer) As Integer
        Return num * num
    End Function
    
    Sub Main()
        Console.WriteLine(Square(42))
    End Sub
End Module
