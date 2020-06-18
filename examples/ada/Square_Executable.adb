with Ada.Command_Line;
with Ada.Text_IO;

function Example return Integer is

    function Square(num : Integer) return Integer is
    begin
        return num**2;
    end Square;

    function ReadCmdArgumentOrDefault(default: Integer) return Integer is
    begin
        if Ada.Command_Line.Argument_Count > 0 then
            return Integer'Value(Ada.Command_Line.Argument(1));
        else
            return Default;
        end if;
    end ReadCmdArgumentOrDefault;

    NumberToSquare: Integer;
    Answer: Integer;
begin
    NumberToSquare := ReadCmdArgumentOrDefault(4);
    Ada.Text_IO.Put_Line("Number to square: " & NumberToSquare'Image);

    Answer := Square(NumberToSquare);
    Ada.Text_IO.Put_Line("Square answer: " & Answer'Image);

    return Answer;
end Example;
