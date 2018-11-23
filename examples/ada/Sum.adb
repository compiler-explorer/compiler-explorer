-- This pragma will remove the warning produced by the default 
-- CE filename and the procedure name differing,
-- see : https://gcc.gnu.org/onlinedocs/gcc-8.2.0/gnat_rm/Pragma-Source_005fFile_005fName.html#Pragma-Source_005fFile_005fName
-- pragma Source_File_Name (Sum, Body_File_Name => "example.adb");
procedure Sum is
    type Integer_Array is array(Natural range <>) of Integer;
    function Sum(input : in Integer_Array) return Natural is
        sum : Natural := 0;
    begin
        for i in input'range loop
            sum := sum + input(i);
        end loop;
        return sum;
    end Sum;
begin
    null;
end Sum;
