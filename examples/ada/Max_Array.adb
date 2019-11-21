-- This pragma will remove the warning produced by the default 
-- CE filename and the procedure name differing,
-- see : https://gcc.gnu.org/onlinedocs/gcc-8.2.0/gnat_rm/Pragma-Source_005fFile_005fName.html#Pragma-Source_005fFile_005fName
-- pragma Source_File_Name (Max_Array, Body_File_Name => "example.adb");
procedure Max_Array is 
    type Integer_Array is array(Natural range <>) of Integer;
    procedure Max_Array(x,y : in out Integer_Array) is
    begin
        for i in x'range loop
            x(i) := (if (y(i) > x(i)) then y(i) else x(i));
        end loop;
    end Max_Array;
begin
    null;
end Max_Array;
