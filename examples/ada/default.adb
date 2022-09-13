-- This pragma will remove the warning produced by the default
-- CE filename and the procedure name differing,
-- see : https://gcc.gnu.org/onlinedocs/gcc-8.2.0/gnat_rm/Pragma-Source_005fFile_005fName.html#Pragma-Source_005fFile_005fName
pragma Source_File_Name (Square, Body_File_Name => "example.adb");

-- Type your code here, or load an example.
function Square(num : Integer) return Integer is
begin
    return num**2;
end Square;

-- Ada 2012 also provides Expression Functions
-- (http://www.ada-auth.org/standards/12rm/html/RM-6-8.html)
-- as a short hand for functions whose body consists of a
-- single return statement. However they cannot be used as a
-- compilation unit.
-- function Square(num : Integer) return Integer is (num**2);
