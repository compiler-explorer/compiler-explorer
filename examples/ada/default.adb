-- This pragma will remove the warning produced by the default
-- CE filename and the procedure name differing,
-- see : https://gcc.gnu.org/onlinedocs/gcc-8.2.0/gnat_rm/Pragma-Source_005fFile_005fName.html#Pragma-Source_005fFile_005fName
pragma Source_File_Name (Square, Body_File_Name => "example.adb");
with Ada.Command_Line;

function Square return Integer is
    Num: Integer;
begin
    if Ada.Command_Line.Argument_Count > 0 then
      num := Integer'Value(Ada.Command_Line.Argument(1));
    else
      num := 3;
    end if;

    return num**2;
end Square;
